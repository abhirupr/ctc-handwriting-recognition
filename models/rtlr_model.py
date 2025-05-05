import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from joblib import Parallel, delayed
from pyctcdecode import BeamSearchDecoderCTC

# ----- Positional Encoding -----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# ----- CNN Backbone with Space-to-Depth -----
class CNNBackbone(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.s2d = nn.PixelUnshuffle(4)  # Output: (N, 16*input_channels, H/4, W/4)

        # First special block: Conv(3x3x128) → Conv(1x1x64)
        self.init_block = nn.Sequential(
            nn.Conv2d(input_channels * 16, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU6(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),
        )

        # Repeat block: Conv(3x3x512) → Conv(1x1x64)
        def repeat_block():
            return nn.Sequential(
                nn.Conv2d(64, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU6(inplace=True),
                nn.Conv2d(512, 64, kernel_size=1),
                nn.BatchNorm2d(64),
                nn.ReLU6(inplace=True),
            )

        self.fused_blocks = nn.Sequential(*[repeat_block() for _ in range(10)])

        # Final block: reduce height to 1 and project to 256 channels
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU6(inplace=True),
            nn.Conv2d(512, 256, kernel_size=(1, 1)),
        )

    def forward(self, x):
        x = self.s2d(x)
        x = self.init_block(x)
        x = self.fused_blocks(x)
        x = self.final_conv(x)
        # Perform global average pooling over the height dimension
        x = F.adaptive_avg_pool2d(x, (1, x.size(3)))
        x = x.squeeze(2)         # (N, 256, 1, W) → (N, 256, W)
        x = x.permute(0, 2, 1)   # (N, 256, W) → (N, W, 256)
        return x


# ----- Transformer Encoder Block -----
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        x2 = self.norm1(x)
        attn_output, _ = self.attn(x2, x2, x2, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout(attn_output)
        x2 = self.norm2(x)
        x = x + self.dropout(self.ffn(x2))
        return x


# ----- Transformer Encoder with 16 Layers -----
class SelfAttentionEncoder(nn.Module):
    def __init__(self, d_model=256, num_layers=16, nhead=4, dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])

    def forward(self, x, src_key_padding_mask=None):
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return x


# ----- CTC Decoder Head -----
class CTCDecoderHead(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.output_layer = nn.Linear(d_model, vocab_size + 1)  # +1 for blank

    def forward(self, x):
        return F.log_softmax(self.output_layer(x), dim=-1)  # (B, T, V+1)


# ----- Full Model -----    
class CTCRecognitionModel(nn.Module):
    def __init__(self, vocab_size, chunk_width=320, pad=32):
        super().__init__()
        self.chunk_w  = chunk_width      # 320 px (for example)
        self.pad      = pad              # bidirectional padding
        self.backbone = CNNBackbone()
        self.encoder  = SelfAttentionEncoder(num_layers=16)
        self.ctc_head = CTCDecoderHead(256, vocab_size)
        self.resize_height = 40

    def forward(self, img):
        # img: (B,1,40,W)  — height normalized to 40 px
        B, C, H_in, W_in = img.size()
        if H_in != self.resize_height:
            # Ensure W_in doesn't change its scaling relative to H_in
            new_W = W_in * self.resize_height // H_in
            img = F.interpolate(img, size=(self.resize_height, new_W),
                                mode='bilinear', align_corners=False)
            _, _, _, W_in = img.size()  # Update W_in after resize
            
        features = []

        # Compute how many chunks
        step = self.chunk_w - 2*self.pad
        for start in range(0, W_in, step):
            end = min(start + self.chunk_w, W_in)
            # pad both sides
            left = max(start - self.pad, 0)
            right = min(end + self.pad, W_in)
            chunk = img[:, :, :, left:right]

            # if near right edge, pad to full chunk size
            if chunk.size(3) < self.chunk_w + 2*self.pad:
                pad_w = self.chunk_w + 2*self.pad - chunk.size(3)
                chunk = F.pad(chunk, (0, pad_w), "constant", 0)

            # run through backbone+encoder
            f = self.backbone(chunk)      # (B, Tc, D)
            f = self.encoder(f)           # (B, Tc, D)

            # remove padded timesteps
            valid_start = max(0, self.pad - (start - left))
            valid_end = valid_start + min(step, W_in - start)
            if valid_end > valid_start and valid_end <= f.size(1):
                features.append(f[:, valid_start:valid_end, :])

        # concat all valid features along time dim
        if features:
            feats = torch.cat(features, dim=1)  # (B, T_total, D)
            return self.ctc_head(feats)         # (B, T_total, V+1)
        else:
            # Handle edge case where no valid features were extracted
            return torch.zeros(B, 1, self.ctc_head.output_layer.out_features, device=img.device)


# ----- PyctcDecode CTC Beam Decoder Wrapper (for inference) -----
class PyCTCBeamDecoder:
    def __init__(self,
                 vocab: List[str],
                 lm_path: Optional[str] = None,
                 alpha: float = 0.5,
                 beta: float = 1.0,
                 beam_width: int = 100,
                 n_jobs: int = -1):
        """
        Initializes the PyCTCBeamDecoder.

        Args:
            vocab (List[str]): List of characters in the vocabulary.
            lm_path (Optional[str]): Path to KenLM language model.
            alpha (float): Weight for language model.
            beta (float): Weight for word insertion.
            beam_width (int): Width of the beam search.
            n_jobs (int): Number of parallel jobs (default: -1 for all cores).
        """
        self.vocab = vocab
        self.decoder = BeamSearchDecoderCTC(vocab, kenlm_model_path=lm_path)
        self.alpha = alpha
        self.beta = beta
        self.beam_width = beam_width
        self.n_jobs = n_jobs

    def _decode_single(self, log_prob: np.ndarray) -> str:
        """
        Decodes a single log-probability tensor.

        Args:
            log_prob (np.ndarray): Log probability array of shape (T, C).

        Returns:
            str: Decoded output string.
        """
        return self.decoder.decode_beams(log_prob,
                                         beam_width=self.beam_width,
                                         alpha=self.alpha,
                                         beta=self.beta)[0][0]

    def decode(self, log_probs: torch.Tensor, input_lengths: torch.Tensor) -> List[str]:
        """
        Decodes a batch of log-probability tensors.

        Args:
            log_probs (torch.Tensor): Log probabilities (B, T, C).
            input_lengths (torch.Tensor): Actual sequence lengths (B).

        Returns:
            List[str]: List of decoded strings.
        """
        log_probs = log_probs.detach().cpu().numpy()
        input_lengths = input_lengths.cpu().numpy()
        sequences = [log_probs[i, :input_lengths[i]] for i in range(log_probs.shape[0])]
        return Parallel(n_jobs=self.n_jobs)(delayed(self._decode_single)(seq) for seq in sequences)


class CTCDecoder(nn.Module):
    def __init__(self, input_dim: int, vocab_size: int):
        """
        CTC decoder head with linear projection.

        Args:
            input_dim (int): Dimension of input features.
            vocab_size (int): Number of output classes.
        """
        super(CTCDecoder, self).__init__()
        self.fc = nn.Linear(input_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the decoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, input_dim).

        Returns:
            torch.Tensor: Log probabilities (B, T, vocab_size).
        """
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)
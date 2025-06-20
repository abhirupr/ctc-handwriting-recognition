import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from joblib import Parallel, delayed

# Import CTCBeamDecoder with fallback
try:
    from ctcdecode import CTCBeamDecoder
    HAS_CTCDECODE = True
except ImportError:
    HAS_CTCDECODE = False
    print("Warning: ctcdecode not available. Falling back to greedy decoding.")

# ----- Positional Encoding -----
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

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
            nn.Conv2d(64, 256, kernel_size=(10, 1), padding=0),  # Reduce height to 1
            nn.BatchNorm2d(256),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.s2d(x)          # (B, 16*C, H/4, W/4)
        x = self.init_block(x)   # (B, 64, H/4, W/4)
        x = self.fused_blocks(x) # (B, 64, H/4, W/4)
        x = self.final_conv(x)   # (B, 256, 1, W/4)
        x = x.squeeze(2)         # (B, 256, W/4)
        x = x.permute(0, 2, 1)   # (B, W/4, 256)
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
        # CRITICAL FIX: Return raw logits, not log_softmax
        # The loss function will handle the softmax/log_softmax
        return self.output_layer(x)  # (B, T, V+1) - raw logits

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
        
        # Ensure img requires grad for gradient flow
        if self.training and not img.requires_grad:
            img.requires_grad_(True)
            
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
            output = self.ctc_head(feats)       # (B, T_total, V+1)
            
            # Ensure output has gradients when in training mode
            if self.training and not output.requires_grad:
                # This should not happen, but if it does, this is a safeguard
                print("Warning: Model output missing gradients, attempting to fix...")
                output.requires_grad_(True)
                
            return output
        else:
            # Handle edge case where no valid features were extracted
            empty_output = torch.zeros(B, 1, self.ctc_head.output_layer.out_features, 
                                     device=img.device, requires_grad=self.training)
            return empty_output

# ----- PyctcDecode CTC Beam Decoder Wrapper (for inference) -----
class PyCTCBeamDecoder:
    def __init__(self, labels, model_path=None, alpha=0.5, beta=1.0, cutoff_top_n=40, cutoff_prob=1.0, beam_width=100, num_processes=4, blank_id=0, log_probs_input=False):
        if HAS_CTCDECODE:
            try:
                # Try to initialize CTCBeamDecoder
                self.decoder = CTCBeamDecoder(labels, model_path, alpha, beta, cutoff_top_n, cutoff_prob, beam_width, num_processes, blank_id, log_probs_input)
                self.available = True
            except Exception as e:
                print(f"CTCBeamDecoder initialization failed: {e}")
                self.decoder = None
                self.available = False
        else:
            self.decoder = None
            self.available = False
            
        self.labels = labels
        self.blank_id = blank_id

    def decode(self, probs, seq_lens=None):
        if self.available and self.decoder:
            beam_results, beam_scores, timesteps, out_seq_len = self.decoder.decode(probs, seq_lens)
            return beam_results, beam_scores, timesteps, out_seq_len
        else:
            # Fallback to greedy decoding if beam search is not available
            return self._greedy_decode(probs, seq_lens)

    def _greedy_decode(self, probs, seq_lens=None):
        # Simple greedy decoding as fallback
        if len(probs.shape) == 3:
            batch_size, max_len, _ = probs.shape
            batch_results = []
            for b in range(batch_size):
                seq_len = seq_lens[b] if seq_lens is not None else max_len
                # Get the most probable character at each time step
                indices = torch.argmax(probs[b, :seq_len], dim=-1)
                # Remove blanks and consecutive duplicates
                decoded = []
                prev = None
                for idx in indices:
                    if idx != self.blank_id and idx != prev:
                        decoded.append(idx.item())
                    prev = idx
                batch_results.append(decoded)
            return batch_results, None, None, None
        else:
            raise ValueError("Expected 3D tensor for probs")

class CTCDecoder(nn.Module):
    def __init__(self, labels, beam_width=10, blank_id=0):
        super().__init__()
        self.labels = labels
        self.beam_width = beam_width
        self.blank_id = blank_id
        self.beam_decoder = PyCTCBeamDecoder(labels, beam_width=beam_width, blank_id=blank_id)

    def decode(self, log_probs):
        # Convert log probabilities to probabilities
        probs = torch.exp(log_probs)
        seq_lens = torch.full((probs.size(0),), probs.size(1), dtype=torch.long)
        
        beam_results, beam_scores, timesteps, out_seq_len = self.beam_decoder.decode(probs, seq_lens)
        
        # Convert results to strings
        decoded_strings = []
        if beam_results is not None:
            for beam_result in beam_results:
                # Take the best beam (first one)
                indices = beam_result[0][:out_seq_len[0][0]]
                decoded_string = ''.join([self.labels[idx] for idx in indices if idx < len(self.labels)])
                decoded_strings.append(decoded_string)
        else:
            # Fallback to simple greedy decoding
            for prob in probs:
                indices = torch.argmax(prob, dim=-1)
                decoded = []
                prev = None
                for idx in indices:
                    if idx != self.blank_id and idx != prev:
                        if idx < len(self.labels):
                            decoded.append(self.labels[idx])
                    prev = idx
                decoded_strings.append(''.join(decoded))
        
        return decoded_strings

class CTCBeamDecoderWrapper:
    def __init__(self, vocab, beam_width=10):
        self.vocab = vocab
        self.beam_width = beam_width
        self.blank_id = 0  # Assuming blank is at index 0
        
    def decode(self, log_probs_batch):
        """
        Decode batch of log probabilities to text
        
        Args:
            log_probs_batch: torch.Tensor of shape (B, T, V+1) with log probabilities
            
        Returns:
            List of decoded strings
        """
        batch_size = log_probs_batch.size(0)
        decoded_texts = []
        
        for b in range(batch_size):
            log_probs = log_probs_batch[b]  # (T, V+1)
            
            # Greedy decoding (can be replaced with beam search later)
            pred_indices = torch.argmax(log_probs, dim=-1)  # (T,)
            
            # Remove consecutive duplicates and blanks
            decoded_sequence = []
            prev_idx = None
            
            for idx in pred_indices:
                idx = idx.item()
                if idx != self.blank_id and idx != prev_idx:
                    if idx > 0 and idx <= len(self.vocab):  # Valid character index
                        decoded_sequence.append(self.vocab[idx-1])  # -1 because blank is at 0
                prev_idx = idx
            
            decoded_texts.append(''.join(decoded_sequence))
        
        return decoded_texts
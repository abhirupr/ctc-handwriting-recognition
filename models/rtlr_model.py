import torch
import torch.nn as nn
import torch.nn.functional as F

# Import CTCBeamDecoder with fallback
try:  # pragma: no cover - optional dependency
    from ctcdecode import CTCBeamDecoder
    HAS_CTCDECODE = True
except ImportError:  # pragma: no cover
    HAS_CTCDECODE = False
    print("Warning: ctcdecode not available. Beam search will fall back to greedy decoding.")

################################################################################
# Active Model: CNN + BiLSTM + Linear (CTC)
################################################################################

# ----- Full Model -----    
class CTCRecognitionModel(nn.Module):
    def __init__(self, vocab_size, chunk_width=320, pad=32, dropout_rate=0.3):
        super().__init__()
        self.vocab_size = vocab_size
        self.chunk_width = chunk_width
        self.pad = pad
        
    # CNN backbone (kept simple & efficient)
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # Increase channels
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Second conv block  
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Third conv block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # Preserve width
            
            # Fourth conv block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # More features
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )
        
        # Calculate feature dimensions
        self.feature_height = 2  # 40 -> 20 -> 10 -> 5 -> 2
        self.feature_dim = 512 * self.feature_height  # 512 * 2 = 1024
        
        # Enhanced LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=512,        # Increase hidden size
            num_layers=2,           # Reduce layers
            bidirectional=True,
            dropout=dropout_rate if dropout_rate > 0 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        
        # Better classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),  # Reduce dimensionality
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, vocab_size)
        )

    def forward(self, x):
        # CNN feature extraction
        features = self.cnn(x)  # (B, 512, H', W')
        
        # Reshape for LSTM
        B, C, H, W = features.size()
        features = features.permute(0, 3, 1, 2)  # (B, W', C, H')
        features = features.contiguous().view(B, W, C * H)
        
        # LSTM processing
        lstm_out, _ = self.lstm(features)
        lstm_out = self.dropout(lstm_out)
        
        # Classification
        output = self.classifier(lstm_out)
        
        return output

# ----- PyctcDecode CTC Beam Decoder Wrapper (for inference) -----
################################################################################
# Beam Search / LM Wrapper
################################################################################
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
    def __init__(self, labels, beam_width=10, blank_id=0, lm_path=None, alpha=0.5, beta=1.0):
        super().__init__()
        self.labels = labels
        self.beam_width = beam_width
        self.blank_id = blank_id
        
        # Only use LM if path is provided and file exists
        import os
        use_lm = lm_path and os.path.exists(lm_path) if lm_path else False
        
        self.beam_decoder = PyCTCBeamDecoder(
            labels, 
            model_path=lm_path if use_lm else None,
            alpha=alpha if use_lm else 0.0,
            beta=beta if use_lm else 0.0,
            beam_width=beam_width, 
            blank_id=blank_id
        )

    def decode(self, log_probs):
        # Convert log probabilities to probabilities
        probs = torch.exp(log_probs)
        seq_lens = torch.full((probs.size(0),), probs.size(1), dtype=torch.long)
        
        beam_results, beam_scores, timesteps, out_seq_len = self.beam_decoder.decode(probs, seq_lens)

        decoded_strings = []
        if beam_results is not None:
            # beam_results: (B, beam_width, max_len) ; out_seq_len: (B, beam_width)
            for b in range(len(beam_results)):
                best_len = out_seq_len[b][0] if out_seq_len is not None else len(beam_results[b][0])
                indices = beam_results[b][0][:best_len]
                decoded = []
                prev = None
                for idx in indices:
                    idx_val = int(idx)
                    if idx_val != self.blank_id and idx_val != prev and idx_val < len(self.labels):
                        decoded.append(self.labels[idx_val])
                    prev = idx_val
                decoded_strings.append(''.join(decoded))
        else:  # Greedy fallback
            for prob in probs:
                indices = torch.argmax(prob, dim=-1)
                decoded = []
                prev = None
                for idx in indices:
                    idx_val = int(idx)
                    if idx_val != self.blank_id and idx_val != prev and idx_val < len(self.labels):
                        decoded.append(self.labels[idx_val])
                    prev = idx_val
                decoded_strings.append(''.join(decoded))

        return decoded_strings

class CTCBeamDecoderWrapper:
    """Simplified greedy fallback decoder (kept for compatibility)."""
    def __init__(self, vocab, beam_width=10):
        self.vocab = vocab
        self.blank_id = 0

    def decode(self, log_probs_batch):
        batch_size = log_probs_batch.size(0)
        texts = []
        for b in range(batch_size):
            preds = torch.argmax(log_probs_batch[b], dim=-1)
            seq = []
            prev = None
            for idx in preds:
                idx_val = int(idx)
                if idx_val != self.blank_id and idx_val != prev and 0 < idx_val <= len(self.vocab):
                    seq.append(self.vocab[idx_val-1])
                prev = idx_val
            texts.append(''.join(seq))
        return texts
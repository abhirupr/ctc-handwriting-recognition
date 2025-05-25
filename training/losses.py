import torch
import torch.nn as nn

class LabelSmoothingCTCLoss(nn.Module):
    def __init__(self, blank_id=0, smoothing=0.1):
        super().__init__()
        self.blank_id = blank_id
        self.smoothing = smoothing
        self.ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)
        
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # Apply label smoothing
        if self.smoothing > 0:
            vocab_size = log_probs.size(-1)
            smooth_factor = self.smoothing / (vocab_size - 1)
            
            # Create smoothed probabilities
            smoothed_probs = log_probs.clone()
            smoothed_probs.fill_(smooth_factor)
            
            # Apply original probabilities with (1 - smoothing) weight
            original_probs = torch.exp(log_probs)
            smoothed_probs = (1 - self.smoothing) * original_probs + self.smoothing * smoothed_probs
            log_probs = torch.log(smoothed_probs + 1e-8)
        
        return self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
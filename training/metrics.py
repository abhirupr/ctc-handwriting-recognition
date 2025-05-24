import torch
import torch.nn.functional as F
import editdistance
from typing import List, Tuple

def calculate_cer(predictions: List[str], targets: List[str]) -> float:
    """Calculate Character Error Rate (CER)"""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    total_chars = 0
    total_errors = 0
    
    for pred, target in zip(predictions, targets):
        # Convert to character level
        pred_chars = list(pred)
        target_chars = list(target)
        
        # Calculate edit distance
        errors = editdistance.eval(pred_chars, target_chars)
        total_errors += errors
        total_chars += len(target_chars)
    
    return (total_errors / total_chars * 100) if total_chars > 0 else 100.0

def calculate_word_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Calculate Word-level Accuracy"""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    correct = sum(1 for pred, target in zip(predictions, targets) if pred.strip() == target.strip())
    total = len(targets)
    
    return (correct / total * 100) if total > 0 else 0.0

def greedy_decode(log_probs: torch.Tensor, converter, blank_id: int = 0) -> List[str]:
    """Simple greedy decoding for CTC outputs"""
    batch_size = log_probs.size(0)
    decoded_texts = []
    
    for b in range(batch_size):
        # Get the most probable indices
        indices = torch.argmax(log_probs[b], dim=-1)  # (T,)
        
        # Remove consecutive duplicates and blanks
        decoded_sequence = []
        prev_idx = None
        
        for idx in indices:
            idx = idx.item()
            if idx != blank_id and idx != prev_idx:
                decoded_sequence.append(idx)
            prev_idx = idx
        
        # Convert indices to text using converter
        try:
            if decoded_sequence:
                text = converter.decode([decoded_sequence])
                decoded_texts.append(text[0] if isinstance(text, list) else str(text))
            else:
                decoded_texts.append("")
        except:
            decoded_texts.append("")
    
    return decoded_texts

def calculate_sequence_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Calculate exact sequence match accuracy"""
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length")
    
    correct = sum(1 for pred, target in zip(predictions, targets) 
                  if pred.strip().lower() == target.strip().lower())
    total = len(targets)
    
    return (correct / total * 100) if total > 0 else 0.0
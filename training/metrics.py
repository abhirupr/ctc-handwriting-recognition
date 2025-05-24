import torch
import torch.nn.functional as F
import editdistance
from typing import List

def greedy_decode(log_probs, converter, blank_id=0):
    """
    Greedy CTC decoding
    
    Args:
        log_probs: (batch_size, seq_len, vocab_size) log probabilities
        converter: LabelConverter instance
        blank_id: ID of blank token (usually 0)
    
    Returns:
        List of decoded strings
    """
    batch_size = log_probs.size(0)
    decoded_texts = []
    
    for batch_idx in range(batch_size):
        # Get predictions for this sequence
        seq_log_probs = log_probs[batch_idx]  # (seq_len, vocab_size)
        
        # Get most likely tokens
        predictions = torch.argmax(seq_log_probs, dim=1)  # (seq_len,)
        
        # Remove consecutive duplicates and blanks
        decoded_indices = []
        prev_token = None
        
        for token in predictions:
            token_val = token.item()
            # Skip blanks and consecutive duplicates
            if token_val != blank_id and token_val != prev_token:
                decoded_indices.append(token_val)
            prev_token = token_val
        
        # Convert indices to text
        try:
            decoded_text = converter.decode([decoded_indices])[0]
            decoded_texts.append(decoded_text)
        except Exception as e:
            print(f"Warning: Decode error for batch {batch_idx}: {e}")
            decoded_texts.append("")  # Return empty string on error
    
    return decoded_texts

def calculate_cer(predictions: List[str], targets: List[str]) -> float:
    """Calculate Character Error Rate"""
    if not predictions or not targets:
        return 100.0
    
    total_chars = 0
    total_errors = 0
    
    for pred, target in zip(predictions, targets):
        # Handle empty predictions
        if not pred.strip() and target.strip():
            total_errors += len(target)
            total_chars += len(target)
        elif pred.strip() and not target.strip():
            total_errors += len(pred)
            total_chars += max(1, len(target))  # Avoid division by zero
        elif pred.strip() or target.strip():
            errors = editdistance.eval(pred, target)
            total_errors += errors
            total_chars += max(len(target), 1)  # Avoid division by zero
    
    if total_chars == 0:
        return 100.0
    
    cer = (total_errors / total_chars) * 100
    return min(cer, 100.0)  # Cap at 100%

def calculate_sequence_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Calculate sequence-level accuracy (exact match)"""
    if not predictions or not targets:
        return 0.0
    
    correct = 0
    total = len(predictions)
    
    for pred, target in zip(predictions, targets):
        if pred.strip() == target.strip():
            correct += 1
    
    return (correct / total) * 100 if total > 0 else 0.0

def calculate_word_accuracy(predictions: List[str], targets: List[str]) -> float:
    """Calculate word-level accuracy"""
    if not predictions or not targets:
        return 0.0
    
    total_words = 0
    correct_words = 0
    
    for pred, target in zip(predictions, targets):
        pred_words = pred.strip().split()
        target_words = target.strip().split()
        
        total_words += len(target_words)
        
        # Find matching words
        for target_word in target_words:
            if target_word in pred_words:
                correct_words += 1
    
    return (correct_words / total_words) * 100 if total_words > 0 else 0.0
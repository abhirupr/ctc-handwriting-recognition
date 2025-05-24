import torch
import torch.nn.functional as F
from typing import List

def debug_model_output(model, sample_batch, converter, device):
    """Debug what the model is actually outputting"""
    model.eval()
    
    with torch.no_grad():
        images, texts, _ = sample_batch
        
        print("🔍 Model Debug Information:")
        print(f"Batch size: {len(images)}")
        print(f"Sample texts: {texts[:3]}")
        
        # Process first image
        img = images[0].unsqueeze(0).to(device)
        text = texts[0]
        
        print(f"\nImage shape: {img.shape}")
        print(f"Target text: '{text}'")
        
        # Get model output
        logits = model(img)
        print(f"Model output shape: {logits.shape}")
        print(f"Vocab size: {logits.shape[-1]}")
        
        # Check output distribution
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        print(f"Max probability: {probs.max().item():.4f}")
        print(f"Min probability: {probs.min().item():.4f}")
        print(f"Entropy: {-(probs * log_probs).sum(dim=-1).mean().item():.4f}")
        
        # Check if model is outputting mostly blanks
        blank_prob = probs[0, :, 0].mean().item()  # Assuming blank is index 0
        print(f"Average blank probability: {blank_prob:.4f}")
        
        # Try to decode
        try:
            from .metrics import greedy_decode
            decoded = greedy_decode(log_probs, converter, blank_id=0)
            print(f"Decoded text: '{decoded[0]}'")
        except Exception as e:
            print(f"Decoding error: {e}")
        
        # Check label encoding
        try:
            encoded = converter.encode([text])
            print(f"Encoded target: {encoded}")
            print(f"Target length: {len(encoded[0]) if encoded and len(encoded) > 0 else 0}")
        except Exception as e:
            print(f"Encoding error: {e}")

def check_convergence_issues(model):
    """Check for common convergence issues"""
    print("\n🔍 Convergence Check:")
    
    # Check gradient flow
    total_norm = 0
    param_count = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
        else:
            print(f"⚠️ No gradient for {name}")
    
    if param_count > 0:
        total_norm = total_norm ** (1. / 2)
        print(f"Total gradient norm: {total_norm:.6f}")
        print(f"Parameters with gradients: {param_count}")
    
    # Check parameter magnitudes
    large_params = 0
    small_params = 0
    for name, param in model.named_parameters():
        if param.data.abs().max() > 10:
            large_params += 1
        if param.data.abs().max() < 1e-5:
            small_params += 1
    
    print(f"Parameters with large values (>10): {large_params}")
    print(f"Parameters with small values (<1e-5): {small_params}")
import torch
import torch.nn as nn
import os
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

def collate_fn(batch):
    """Collate function that processes each image individually (no padding needed)"""
    images, labels = zip(*batch)
    
    # Don't stack images - process them individually since the model handles chunking
    # Just return them as a list for now, we'll process one by one in the training loop
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    return list(images), labels, label_lengths

def train_model(model, train_dataloader, val_dataloader, converter, device, optimizer, config):
    """Main training function"""
    model.to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)  # blank=0 for CTC blank token
    
    # Create directories for saving models
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.BEST_MODEL_DIR, exist_ok=True)
    
    best_loss = float('inf')
    saved_checkpoints = []
    
    print(f"Starting training on {device}")
    print(f"Training samples: {len(train_dataloader.dataset)}")
    print(f"Validation samples: {len(val_dataloader.dataset)}")
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, texts, text_lengths) in enumerate(train_dataloader):
            # Process images one by one due to variable widths and chunking strategy
            batch_losses = []
            
            for img_idx, (img, text) in enumerate(zip(images, texts)):
                # Move single image to device and add batch dimension
                img = img.unsqueeze(0).to(device)  # (1, C, H, W)
                
                # Convert text label to indices
                try:
                    encoded = converter.encode([text])[0]  # encode returns a list
                    if len(encoded) == 0:
                        continue  # Skip empty labels
                    
                    labels_tensor = encoded.to(device)
                    label_length = torch.tensor([len(encoded)], dtype=torch.long).to(device)
                    
                    # Forward pass through model (handles chunking internally)
                    optimizer.zero_grad()
                    logits = model(img)  # (1, T, V+1) where T depends on image width
                    
                    # CTC expects (T, B, V+1)
                    log_probs = logits.log_softmax(2).permute(1, 0, 2)  # (T, 1, V+1)
                    
                    # Input lengths (sequence length for the single image)
                    input_length = torch.tensor([logits.size(1)], dtype=torch.long).to(device)
                    
                    # Reshape labels for CTC loss
                    labels_padded = labels_tensor.unsqueeze(0)  # (1, L)
                    
                    # Calculate CTC loss
                    loss = criterion(log_probs, labels_padded, input_length, label_length)
                    
                    # Accumulate gradients
                    loss.backward()
                    batch_losses.append(loss.item())
                    
                except Exception as e:
                    print(f"Error processing sample {img_idx} in batch {batch_idx}: {e}")
                    continue
            
            # Update weights after processing all images in the batch
            if batch_losses:
                optimizer.step()
                avg_batch_loss = sum(batch_losses) / len(batch_losses)
                total_loss += avg_batch_loss
                num_batches += 1
                
                if batch_idx % 10 == 0:  # Print every 10 batches
                    print(f"Epoch {epoch+1}/{config.EPOCHS}, Batch {batch_idx}, Avg Loss: {avg_batch_loss:.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f"Epoch {epoch+1}/{config.EPOCHS} - Average Loss: {avg_loss:.4f}")
        
        # Validation
        if epoch % config.EVAL_STEP == 0:
            val_loss = evaluate_model(model, val_dataloader, converter, criterion, device)
            print(f"Validation Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_path = os.path.join(config.BEST_MODEL_DIR, config.BEST_MODEL_NAME)
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved to {best_model_path}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(config.SAVE_DIR, f"{config.CHECKPOINT_PREFIX}_{epoch+1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        
        # Manage saved checkpoints
        saved_checkpoints.append(checkpoint_path)
        if len(saved_checkpoints) > config.SAVE_LIMIT:
            oldest_checkpoint = saved_checkpoints.pop(0)
            if os.path.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)
                print(f"Removed old checkpoint: {oldest_checkpoint}")

def evaluate_model(model, dataloader, converter, criterion, device):
    """Evaluate the model on validation set"""
    model.eval()
    total_loss = 0
    num_samples = 0
    
    with torch.no_grad():
        for images, texts, text_lengths in dataloader:
            for img, text in zip(images, texts):
                try:
                    img = img.unsqueeze(0).to(device)  # (1, C, H, W)
                    
                    # Convert text label to indices
                    encoded = converter.encode([text])[0]
                    if len(encoded) == 0:
                        continue
                    
                    labels_tensor = encoded.to(device)
                    label_length = torch.tensor([len(encoded)], dtype=torch.long).to(device)
                    
                    # Forward pass
                    logits = model(img)
                    log_probs = logits.log_softmax(2).permute(1, 0, 2)
                    
                    input_length = torch.tensor([logits.size(1)], dtype=torch.long).to(device)
                    labels_padded = labels_tensor.unsqueeze(0)
                    
                    loss = criterion(log_probs, labels_padded, input_length, label_length)
                    total_loss += loss.item()
                    num_samples += 1
                    
                except Exception as e:
                    print(f"Error in validation: {e}")
                    continue
    
    return total_loss / num_samples if num_samples > 0 else float('inf')

import torch
import torch.nn as nn
import os
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """Collate function for DataLoader"""
    images, labels = zip(*batch)
    images = torch.stack(images)
    
    # Convert labels to indices using the converter
    # For now, just return the raw labels - we'll process them in the training loop
    label_lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    return images, labels, label_lengths

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
            images = images.to(device)
            
            # Convert text labels to indices
            encoded_labels = []
            for text in texts:
                encoded = converter.encode([text])[0]  # encode returns a list
                encoded_labels.append(encoded)
            
            # Pad the encoded labels
            if encoded_labels:
                labels_padded = pad_sequence(encoded_labels, batch_first=True, padding_value=0)
                label_lengths = torch.tensor([len(label) for label in encoded_labels], dtype=torch.long)
            else:
                continue  # Skip batch if no valid labels
            
            labels_padded = labels_padded.to(device)
            label_lengths = label_lengths.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(images)  # (B, T, V+1)
            
            # CTC expects (T, B, V+1)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)
            
            # Input lengths (sequence length for each item in batch)
            input_lengths = torch.full(
                size=(logits.size(0),), 
                fill_value=logits.size(1), 
                dtype=torch.long
            ).to(device)
            
            # Calculate CTC loss
            loss = criterion(log_probs, labels_padded, input_lengths, label_lengths)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Epoch {epoch+1}/{config.EPOCHS}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
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
    num_batches = 0
    
    with torch.no_grad():
        for images, texts, text_lengths in dataloader:
            images = images.to(device)
            
            # Convert text labels to indices
            encoded_labels = []
            for text in texts:
                encoded = converter.encode([text])[0]
                encoded_labels.append(encoded)
            
            if encoded_labels:
                labels_padded = pad_sequence(encoded_labels, batch_first=True, padding_value=0)
                label_lengths = torch.tensor([len(label) for label in encoded_labels], dtype=torch.long)
            else:
                continue
            
            labels_padded = labels_padded.to(device)
            label_lengths = label_lengths.to(device)
            
            # Forward pass
            logits = model(images)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)
            
            input_lengths = torch.full(
                size=(logits.size(0),), 
                fill_value=logits.size(1), 
                dtype=torch.long
            ).to(device)
            
            loss = criterion(log_probs, labels_padded, input_lengths, label_lengths)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else float('inf')

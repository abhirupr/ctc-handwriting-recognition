import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import time
from .metrics import calculate_cer, calculate_word_accuracy, calculate_sequence_accuracy, greedy_decode
from models.rtlr_model import CTCDecoder  # LM / beam decoder
from .early_stopping import EarlyStopping

def train_model(model, train_dataloader, val_dataloader, converter, device, optimizer, config, scheduler=None):
    """Main training function with comprehensive metrics and early stopping"""
    model.to(device)
    
    # Ensure model parameters require gradients
    for param in model.parameters():
        param.requires_grad = True
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
    
    # Create directories for saving models
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.BEST_MODEL_DIR, exist_ok=True)
    
    best_metric = 0.0 if config.EVAL_STRATEGY == "accuracy" else float('inf')
    saved_checkpoints = []
    
    # Initialize early stopping
    early_stopping = None
    if hasattr(config, 'EARLY_STOPPING') and config.EARLY_STOPPING.get('enabled', False):
        early_stopping_config = config.EARLY_STOPPING
        early_stopping = EarlyStopping(
            patience=early_stopping_config.get('patience', 7),
            min_delta=early_stopping_config.get('min_delta', 0.0001),
            restore_best_weights=early_stopping_config.get('restore_best_weights', True),
            mode=early_stopping_config.get('mode', 'max'),
            baseline=early_stopping_config.get('baseline', None),
            verbose=early_stopping_config.get('verbose', True)
        )
        print(f"🎯 Early stopping enabled: patience={early_stopping.patience}, mode={early_stopping.mode}")
    
    print(f"Starting training on {device}")
    print(f"Training samples: {len(train_dataloader.dataset)}")
    print(f"Validation samples: {len(val_dataloader.dataset)}")
    print(f"Evaluation strategy: {config.EVAL_STRATEGY}")
    print(f"Maximum epochs: {config.EPOCHS}")
    
    # Debug: Check if model parameters require gradients
    grad_params = sum(p.requires_grad for p in model.parameters())
    total_params = sum(1 for _ in model.parameters())
    print(f"Model parameters requiring gradients: {grad_params}/{total_params}")
    print("-" * 80)
    
    # Debug: Check first batch
    print("\n🔍 Initial model debugging...")
    first_batch = next(iter(train_dataloader))
    from .debug_helpers import debug_model_output, check_convergence_issues
    debug_model_output(model, first_batch, converter, device)
    
    training_start_time = time.time()
    
    # Optional beam decoder (only for validation to save time)
    beam_decoder = None
    if getattr(config, 'USE_LANGUAGE_MODEL', False):
        try:
            beam_decoder = CTCDecoder(labels=converter.vocab, beam_width=getattr(config, 'BEAM_WIDTH', 50), lm_path=getattr(config, 'LM_MODEL_PATH', None), alpha=getattr(config, 'LM_ALPHA', 0.5), beta=getattr(config, 'LM_BETA', 1.0))
            print("Beam decoder initialized.")
        except Exception as e:
            print(f"Beam decoder init failed, falling back to greedy: {e}")
            beam_decoder = None

    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()
        
        # Training phase
        train_metrics = train_epoch(model, train_dataloader, converter, device, optimizer, criterion, epoch, config)
        
        # Validation phase
        if epoch % config.EVAL_STEP == 0:
            val_metrics = evaluate_model(model, val_dataloader, converter, criterion, device, beam_decoder=beam_decoder)
            
            # Print comprehensive metrics
            print_metrics(epoch + 1, config.EPOCHS, train_metrics, val_metrics, time.time() - epoch_start_time)
            
            # Save best model based on evaluation strategy
            current_metric = val_metrics['accuracy'] if config.EVAL_STRATEGY == "accuracy" else val_metrics['loss']
            is_best = False
            
            if config.EVAL_STRATEGY == "accuracy" and current_metric > best_metric:
                best_metric = current_metric
                is_best = True
            elif config.EVAL_STRATEGY == "loss" and current_metric < best_metric:
                best_metric = current_metric
                is_best = True
            
            if is_best:
                best_model_path = os.path.join(config.BEST_MODEL_DIR, config.BEST_MODEL_NAME)
                torch.save(model.state_dict(), best_model_path)
                print(f"✅ Best model saved! {config.EVAL_STRATEGY.capitalize()}: {best_metric:.4f}")
            
            # Check early stopping - Use CER instead of accuracy
            if early_stopping is not None:
                # Use CER for early stopping (lower is better)
                early_stop_metric = val_metrics['cer']  # Use CER instead of accuracy
                
                should_stop = early_stopping(early_stop_metric, model)
                if should_stop:
                    print(f"\n🛑 Training stopped early at epoch {epoch + 1}")
                    print(f"   Best CER: {early_stopping.get_best_metric():.2f}%")
                    
                    # Restore best weights if configured
                    if early_stopping.restore_best_weights:
                        early_stopping.restore_weights(model)
                        
                        # Also save the final restored model
                        final_model_path = os.path.join(config.BEST_MODEL_DIR, "final_model_early_stopped.pth")
                        torch.save(model.state_dict(), final_model_path)
                        print(f"   Final model saved: {final_model_path}")
                    
                    break
        else:
            # Print training metrics only
            print_training_metrics(epoch + 1, config.EPOCHS, train_metrics, time.time() - epoch_start_time)
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, train_metrics['loss'], config, saved_checkpoints)
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                # ReduceLROnPlateau needs validation metric
                scheduler.step(val_metrics['cer'])
                print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
            else:
                # Time-based schedulers
                scheduler.step()
                print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        print("-" * 80)
    
    # Training completed
    total_training_time = time.time() - training_start_time
    
    if early_stopping is not None and early_stopping.wait < early_stopping.patience:
        print(f"\n🎉 Training completed normally after {epoch + 1} epochs")
        print(f"   Final {config.EVAL_STRATEGY}: {current_metric:.4f}")
        print(f"   Best {config.EVAL_STRATEGY}: {early_stopping.get_best_metric():.4f}")
    else:
        print(f"\n🎉 Training completed after {config.EPOCHS} epochs")
    
    print(f"⏱️ Total training time: {total_training_time/3600:.1f} hours ({total_training_time/60:.1f} minutes)")
    
    return model

def train_epoch(model, dataloader, converter, device, optimizer, criterion, epoch: int, config) -> Dict[str, float]:
    """Train for one epoch and return metrics"""
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    for batch_idx, (images, texts, text_lengths) in enumerate(dataloader):
        optimizer.zero_grad()
        batch_losses = []
        batch_predictions = []
        batch_targets = []
        
        for img_idx, (img, text) in enumerate(zip(images, texts)):
            try:
                # Prepare input
                img = img.unsqueeze(0).to(device)
                img.requires_grad_(True)
                
                # Prepare labels
                encoded = converter.encode([text])[0]
                if len(encoded) == 0:
                    continue
                
                if isinstance(encoded, torch.Tensor):
                    labels_np = encoded.detach().cpu().numpy()
                    labels_tensor = torch.from_numpy(labels_np).long().to(device)
                else:
                    labels_tensor = torch.tensor(encoded, dtype=torch.long, device=device)
                
                if labels_tensor.dim() > 1:
                    labels_tensor = labels_tensor.flatten()
                
                label_length = torch.tensor([len(labels_tensor)], dtype=torch.long, device=device)
                
                # Forward pass
                logits = model(img)  # (1, T, V+1) where T depends on image width
                
                # Debug: Print sample details for less batches
                if batch_idx < 1 and img_idx == 0:  # Only first batch
                    print(f"\n🔍 Sample Debug (Batch {batch_idx}):")
                    print(f"   Image shape: {img.shape}")
                    print(f"   Logits shape: {logits.shape}")
                    print(f"   Text: '{text}' (len: {len(text)})")
                    print(f"   Encoded: {encoded} (len: {len(encoded)})")
                    print(f"   Logits range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")
                    print(f"   Logits std: {logits.std().item():.3f}")
                
                # Verify gradients are flowing
                if not logits.requires_grad:
                    print(f"Warning: Model output still doesn't require grad in batch {batch_idx}, sample {img_idx}")
                    continue
                
                # Validate input/label alignment
                if logits.size(1) < len(labels_tensor):
                    print(f"⚠️ Sequence too long: input_len={logits.size(1)}, target_len={len(labels_tensor)}")
                    continue

                # Also add minimum sequence length check
                if logits.size(1) < 2:  # CTC needs at least 2 time steps
                    print(f"⚠️ Sequence too short: {logits.size(1)} time steps")
                    continue
                
                # CTC expects log probabilities in (T, B, V+1) format
                log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)  # (T, 1, V+1)
                
                # Input lengths (sequence length for the single image)
                input_length = torch.tensor([logits.size(1)], dtype=torch.long, device=device)
                
                # Debug CTC inputs for first few samples
                if batch_idx < 3 and img_idx == 0:
                    print(f"   CTC Input length: {input_length.item()}")
                    print(f"   Target length: {label_length.item()}")
                    print(f"   Log probs shape: {log_probs.shape}")
                    print(f"   Log probs range: [{log_probs.min().item():.3f}, {log_probs.max().item():.3f}]")
                
                # Validate tensor shapes before CTC loss
                if log_probs.size(0) == 0 or labels_tensor.size(0) == 0:
                    print(f"Empty sequence detected in batch {batch_idx}, sample {img_idx}")
                    continue
                
                # Check if input sequence is long enough for target
                if input_length.item() < label_length.item():
                    print(f"⚠️ Sequence too short: input_len={input_length.item()}, target_len={label_length.item()}")
                    continue
                
                # Calculate CTC loss
                loss = criterion(log_probs, labels_tensor, input_length, label_length)
                
                # Debug loss for first few samples
                if batch_idx < 3 and img_idx == 0:
                    print(f"   CTC Loss: {loss.item():.4f}")
                
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                # Backward pass
                loss.backward()
                batch_losses.append(loss.item())
                
                # Debug gradients for first few batches
                if batch_idx < 3 and img_idx == 0:
                    total_grad_norm = 0
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_grad_norm += param_norm.item() ** 2
                    total_grad_norm = total_grad_norm ** (1. / 2)
                    print(f"   Total gradient norm: {total_grad_norm:.6f}")
                
                # Decode for metrics (every 10th batch to save computation)
                if batch_idx % 10 == 0:
                    with torch.no_grad():
                        pred_texts = greedy_decode(log_probs.permute(1, 0, 2), converter)
                        batch_predictions.extend(pred_texts)
                        batch_targets.append(text)
                
                # Quick decode check for first few samples
                if batch_idx == 0 and img_idx == 0:  # Only first sample of first batch
                    with torch.no_grad():
                        sample_decoded = greedy_decode(log_probs.permute(1, 0, 2), converter)
                        print(f"📝 Sample decode: '{sample_decoded[0][:50]}...' (target: '{text[:50]}...')")
                
            except Exception as e:
                if batch_idx < 3:  # Only print first few errors
                    print(f"Error in batch {batch_idx}, sample {img_idx}: {e}")
                continue
        
        # Update weights with gradient scaling
        if batch_losses:
            # Scale gradients if they're too small
            total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # Check if gradients are too small
            if total_grad_norm < 0.1:
                # Scale up gradients
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data *= 2.0
            
            optimizer.step()
            
            avg_batch_loss = sum(batch_losses) / len(batch_losses)
            total_loss += avg_batch_loss * len(batch_losses)
            total_samples += len(batch_losses)
            
            # Collect predictions for metrics
            all_predictions.extend(batch_predictions)
            all_targets.extend(batch_targets)
            
            if batch_idx % 20 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx:3d}, Loss: {avg_batch_loss:.4f}, "
                      f"Processed: {len(batch_losses)}/{len(images)}")
    
    # Calculate epoch metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    
    # Calculate accuracy metrics (on subset for speed)
    if all_predictions and all_targets:
        train_cer = calculate_cer(all_predictions, all_targets)
        train_accuracy = calculate_sequence_accuracy(all_predictions, all_targets)
    else:
        train_cer = 100.0
        train_accuracy = 0.0
    
    # Debug: Check model after each epoch
    if epoch % 10 == 0:  # Every 10 epochs instead of 5
        print(f"\n🔍 Debug after epoch {epoch + 1}:")
        from .debug_helpers import check_convergence_issues, debug_model_output
        check_convergence_issues(model)
        if 'images' in locals() and 'texts' in locals():
            sample_images = images[:1]
            sample_texts = texts[:1]
            debug_model_output(model, (sample_images, sample_texts, [len(sample_texts[0])]), converter, device)
    
    return {
        'loss': avg_loss,
        'cer': train_cer,
        'accuracy': train_accuracy,
        'samples': total_samples
    }

def evaluate_model(model, dataloader, converter, criterion, device, beam_decoder=None) -> Dict[str, float]:
    """Comprehensive evaluation with all metrics"""
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, texts, text_lengths) in enumerate(dataloader):
            for img, text in zip(images, texts):
                try:
                    img = img.unsqueeze(0).to(device)
                    
                    # Prepare labels
                    encoded = converter.encode([text])[0]
                    if len(encoded) == 0:
                        continue
                    
                    if isinstance(encoded, torch.Tensor):
                        labels_np = encoded.detach().cpu().numpy()
                        labels_tensor = torch.from_numpy(labels_np).long().to(device)
                    else:
                        labels_tensor = torch.tensor(encoded, dtype=torch.long, device=device)
                    
                    if labels_tensor.dim() > 1:
                        labels_tensor = labels_tensor.flatten()
                    
                    label_length = torch.tensor([len(labels_tensor)], dtype=torch.long, device=device)
                    
                    # Forward pass
                    logits = model(img)
                    log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)
                    input_length = torch.tensor([logits.size(1)], dtype=torch.long, device=device)
                    
                    # Calculate loss
                    loss = criterion(log_probs, labels_tensor, input_length, label_length)
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        total_loss += loss.item()
                        total_samples += 1
                        
                        # Decode prediction (beam if available)
                        if beam_decoder is not None:
                            with torch.no_grad():
                                # beam decoder expects log_probs in (B,T,V) log space -> convert to probs
                                probs = torch.exp(log_probs.permute(1,0,2))  # (1,T,V)
                                pred_texts = beam_decoder.decode(torch.log(probs + 1e-8))  # reuse interface
                        else:
                            pred_texts = greedy_decode(log_probs.permute(1, 0, 2), converter)
                        all_predictions.extend(pred_texts)
                        all_targets.append(text)
                
                except Exception as e:
                    continue
    
    # Calculate metrics
    avg_loss = total_loss / total_samples if total_samples > 0 else float('inf')
    
    if all_predictions and all_targets:
        val_cer = calculate_cer(all_predictions, all_targets)
        val_accuracy = calculate_sequence_accuracy(all_predictions, all_targets)
        word_accuracy = calculate_word_accuracy(all_predictions, all_targets)
    else:
        val_cer = 100.0
        val_accuracy = 0.0
        word_accuracy = 0.0
    
    return {
        'loss': avg_loss,
        'cer': val_cer,
        'accuracy': val_accuracy,
        'word_accuracy': word_accuracy,
        'samples': total_samples
    }

def print_metrics(epoch: int, total_epochs: int, train_metrics: Dict, val_metrics: Dict, epoch_time: float):
    """Print comprehensive training and validation metrics"""
    print(f"Epoch {epoch:2d}/{total_epochs}")
    print(f"Time: {epoch_time:.1f}s")
    print(f"Train - Loss: {train_metrics['loss']:7.4f} | CER: {train_metrics['cer']:5.1f}% | Acc: {train_metrics['accuracy']:5.1f}% | Samples: {train_metrics['samples']}")
    print(f"Valid - Loss: {val_metrics['loss']:7.4f} | CER: {val_metrics['cer']:5.1f}% | Acc: {val_metrics['accuracy']:5.1f}% | Word Acc: {val_metrics['word_accuracy']:5.1f}% | Samples: {val_metrics['samples']}")

def print_training_metrics(epoch: int, total_epochs: int, train_metrics: Dict, epoch_time: float):
    """Print training metrics only"""
    print(f"Epoch {epoch:2d}/{total_epochs} - Time: {epoch_time:.1f}s")
    print(f"Train - Loss: {train_metrics['loss']:7.4f} | CER: {train_metrics['cer']:5.1f}% | Acc: {train_metrics['accuracy']:5.1f}% | Samples: {train_metrics['samples']}")

def save_checkpoint(model, optimizer, epoch: int, loss: float, config, saved_checkpoints: List[str]):
    """Save model checkpoint"""
    checkpoint_path = os.path.join(config.SAVE_DIR, f"{config.CHECKPOINT_PREFIX}_{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    # Manage saved checkpoints
    saved_checkpoints.append(checkpoint_path)
    if len(saved_checkpoints) > config.SAVE_LIMIT:
        oldest_checkpoint = saved_checkpoints.pop(0)
        if os.path.exists(oldest_checkpoint):
            os.remove(oldest_checkpoint)

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import time
from .metrics import calculate_cer, calculate_word_accuracy, calculate_sequence_accuracy, greedy_decode
# LM / beam decoder placeholder (to be reintroduced later)
CTCDecoder = None  # type: ignore
from .early_stopping import EarlyStopping

def train_model(model, train_dataloader, val_dataloader, converter, device, optimizer, config, scheduler=None):
    """Main training function with comprehensive metrics and early stopping"""
    model.to(device)
    
    # Ensure model parameters require gradients
    for param in model.parameters():
        param.requires_grad = True
    
    criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')
    use_amp = bool(getattr(config, 'MIXED_PRECISION', False) and device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    grad_accum = max(1, int(getattr(config, 'GRAD_ACCUM_STEPS', 1)))
    
    # Create directories for saving models
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.BEST_MODEL_DIR, exist_ok=True)
    
    best_metric = 0.0 if config.EVAL_STRATEGY == "accuracy" else float('inf')
    best_val_loss = float('inf')  # Always track best validation loss separately
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
    
    # Optional OneCycleLR override (creates per-step scheduler)
    if getattr(config, 'USE_ONECYCLE', False):
        steps_per_epoch = max(1, len(train_dataloader))
        total_steps = steps_per_epoch * config.EPOCHS
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=getattr(config, 'ONECYCLE_MAX_LR', optimizer.param_groups[0]['lr']),
            total_steps=total_steps,
            pct_start=getattr(config, 'ONECYCLE_PCT_START', 0.15),
            div_factor=getattr(config, 'ONECYCLE_DIV_FACTOR', 25.0),
            final_div_factor=getattr(config, 'ONECYCLE_FINAL_DIV_FACTOR', 100.0),
            anneal_strategy='cos'
        )
        print(f"OneCycleLR enabled: steps_per_epoch={steps_per_epoch}, total_steps={total_steps}")

    # Optional beam decoder (only for validation to save time)
    beam_decoder = None
    if getattr(config, 'USE_LANGUAGE_MODEL', False):
        try:
            beam_decoder = CTCDecoder(labels=converter.vocab, beam_width=getattr(config, 'BEAM_WIDTH', 50), lm_path=getattr(config, 'LM_MODEL_PATH', None), alpha=getattr(config, 'LM_ALPHA', 0.5), beta=getattr(config, 'LM_BETA', 1.0))
            print("Beam decoder initialized.")
        except Exception as e:
            print(f"Beam decoder init failed, falling back to greedy: {e}")
            beam_decoder = None

    # Track global optimization steps (after optimizer.step()) for warmup & logging
    global_step = 0
    # Stash base lr for warmup (if not already present)
    for pg in optimizer.param_groups:
        pg.setdefault('initial_lr', pg['lr'])

    for epoch in range(config.EPOCHS):
        epoch_start_time = time.time()

        # Training phase
        train_metrics, global_step = train_epoch(
            model,
            train_dataloader,
            converter,
            device,
            optimizer,
            criterion,
            epoch,
            config,
            scheduler if getattr(config, 'USE_ONECYCLE', False) else None,
            scaler=scaler,
            use_amp=use_amp,
            grad_accum=grad_accum,
            global_step=global_step
        )
        
        # Validation phase
        if epoch % config.EVAL_STEP == 0:
            val_metrics = evaluate_model(model, val_dataloader, converter, criterion, device, beam_decoder=beam_decoder)
            
            # Print comprehensive metrics
            print_metrics(epoch + 1, config.EPOCHS, train_metrics, val_metrics, time.time() - epoch_start_time)
            
            # Save best model based on evaluation strategy
            current_metric = val_metrics['accuracy'] if config.EVAL_STRATEGY == "accuracy" else val_metrics['loss']
            # Track best validation loss always
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
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
        if scheduler is not None and not getattr(config, 'USE_ONECYCLE', False):
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['cer'])
            else:
                scheduler.step()
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        print("-" * 80)
    
    # Training completed
    total_training_time = time.time() - training_start_time
    
    print("\n================ Training Summary ================")
    print(f"Epochs run: {epoch + 1}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    if early_stopping is not None:
        print(f"Best validation CER (early stopping metric): {early_stopping.get_best_metric():.2f}%")
    print(f"Final epoch validation loss: {val_metrics['loss']:.4f}")
    print(f"Final epoch validation CER: {val_metrics['cer']:.2f}%")
    print(f"Final epoch validation accuracy: {val_metrics['accuracy']:.2f}%")
    print(f"Total training time: {total_training_time/3600:.2f} hours ({total_training_time/60:.1f} minutes)")
    print("=================================================")
    
    return model

def train_epoch(model, dataloader, converter, device, optimizer, criterion, epoch: int, config, scheduler=None, scaler=None, use_amp: bool=False, grad_accum: int=1, global_step: int = 0) -> Tuple[Dict[str, float], int]:
    """Train for one epoch and return (metrics, updated_global_step).
    Adds: label smoothing for CTC, linear LR warmup, grad norm logging, configurable clipping."""
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    max_batches = getattr(config, 'MAX_BATCHES_PER_EPOCH', None)
    log_interval = getattr(config, 'LOG_INTERVAL', 10)
    batch_start = time.time()
    token_count = 0
    
    max_grad_norm = float(getattr(config, 'MAX_GRAD_NORM', 5.0))
    label_smooth = float(getattr(config, 'CTC_LABEL_SMOOTH', 0.0))
    warmup_steps = int(getattr(config, 'WARMUP_STEPS', 0)) if not getattr(config, 'USE_ONECYCLE', False) else 0
    grad_norm_log_interval = int(getattr(config, 'GRAD_NORM_LOG_INTERVAL', 0))

    for batch_idx, (images, texts, text_lengths) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)

        if getattr(model, 'supports_batch_ctc', False):
            # Batched path for paper model
            widths = [img.shape[-1] for img in images]
            max_w = max(widths)
            padded_imgs = []
            for img in images:
                if img.shape[-1] < max_w:
                    pad_w = max_w - img.shape[-1]
                    img = F.pad(img, (0, pad_w, 0, 0))
                padded_imgs.append(img.unsqueeze(0))
            batch_imgs = torch.cat(padded_imgs, dim=0).to(device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(batch_imgs)
                log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)
            input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=device)
            encoded = converter.encode(texts)
            target_lengths = torch.tensor([e.numel() for e in encoded], device=device, dtype=torch.long)
            targets = torch.cat(encoded).to(device) if encoded else torch.tensor([], dtype=torch.long, device=device)
            if targets.numel() == 0:
                continue
            # Label smoothing (convert log_probs -> probs, smooth, back to log) keeping gradients
            if label_smooth > 0.0:
                probs = log_probs.exp()
                vocab = probs.size(-1)
                probs = (1.0 - label_smooth) * probs + label_smooth / vocab
                log_probs = probs.clamp_min(1e-12).log()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = criterion(log_probs, targets, input_lengths, target_lengths) / grad_accum
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if (batch_idx + 1) % grad_accum == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                # Compute grad norm (pre-clip)
                if grad_norm_log_interval > 0 and (global_step % grad_norm_log_interval == 0):
                    total_norm = 0.0
                    for p in model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    total_norm **= 0.5
                    print(f"GradNorm (pre-clip) @ step {global_step}: {total_norm:.2f}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                # Linear warmup (adjust lr before stepping scaler/optimizer)
                if warmup_steps > 0 and global_step < warmup_steps:
                    warm_scale = float(global_step + 1) / float(max(1, warmup_steps))
                    for pg in optimizer.param_groups:
                        base_lr = pg.get('initial_lr', pg['lr'])
                        pg['lr'] = base_lr * warm_scale
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                global_step += 1
                if scheduler is not None and getattr(config, 'USE_ONECYCLE', False):
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += loss.item() * grad_accum
            total_samples += 1
            token_count += int(target_lengths.sum().item())
            if batch_idx % log_interval == 0:
                with torch.no_grad():
                    preds = greedy_decode(log_probs.permute(1, 0, 2), converter)
                    all_predictions.extend(preds[:len(texts)])
                    all_targets.extend(texts)
                elapsed = time.time() - batch_start
                tps = token_count / elapsed if elapsed > 0 else 0.0
                print(f"Epoch {epoch+1}, Batch {batch_idx:3d}, Loss: {loss.item()*grad_accum:.4f}, B={len(texts)}, tokens/s={tps:.1f}, seq_len={logits.size(1)}")
                batch_start = time.time(); token_count = 0
        else:
            # Original per-sample path
            batch_losses = []
            batch_predictions = []
            batch_targets = []
            for img_idx, (img, text) in enumerate(zip(images, texts)):
                try:
                    img = img.unsqueeze(0).to(device)
                    img.requires_grad_(True)
                    encoded = converter.encode([text])[0]
                    if len(encoded) == 0:
                        continue
                    if isinstance(encoded, torch.Tensor):
                        labels_tensor = encoded.long().to(device)
                    else:
                        labels_tensor = torch.tensor(encoded, dtype=torch.long, device=device)
                    if labels_tensor.dim() > 1:
                        labels_tensor = labels_tensor.flatten()
                    label_length = torch.tensor([len(labels_tensor)], dtype=torch.long, device=device)
                    logits = model(img)
                    if logits.size(1) < len(labels_tensor) or logits.size(1) < 2:
                        continue
                    log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)
                    input_length = torch.tensor([logits.size(1)], dtype=torch.long, device=device)
                    # (Per-sample path) optional label smoothing (retain gradients)
                    if label_smooth > 0.0:
                        probs = log_probs.exp()
                        vocab = probs.size(-1)
                        probs = (1.0 - label_smooth) * probs + label_smooth / vocab
                        log_probs = probs.clamp_min(1e-12).log()
                    loss = criterion(log_probs, labels_tensor, input_length, label_length)
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    loss.backward()
                    batch_losses.append(loss.item())
                    if batch_idx % 10 == 0:
                        with torch.no_grad():
                            pred_texts = greedy_decode(log_probs.permute(1, 0, 2), converter)
                            batch_predictions.extend(pred_texts)
                            batch_targets.append(text)
                except Exception:
                    continue
            if batch_losses:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()
                if scheduler is not None and getattr(config, 'USE_ONECYCLE', False):
                    scheduler.step()
                global_step += 1
                avg_batch_loss = sum(batch_losses) / len(batch_losses)
                total_loss += avg_batch_loss * len(batch_losses)
                total_samples += len(batch_losses)
                all_predictions.extend(batch_predictions)
                all_targets.extend(batch_targets)
                if batch_idx % 20 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx:3d}, Loss: {avg_batch_loss:.4f}, Processed: {len(batch_losses)}/{len(images)}")
    
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
    
    return ({
        'loss': avg_loss,
        'cer': train_cer,
        'accuracy': train_accuracy,
        'samples': total_samples
    }, global_step)

def evaluate_model(model, dataloader, converter, criterion, device, beam_decoder=None) -> Dict[str, float]:
    """Comprehensive evaluation with all metrics"""
    model.eval()
    
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, texts, text_lengths) in enumerate(dataloader):
            if getattr(model, 'supports_batch_ctc', False):
                # Vectorized path
                widths = [img.shape[-1] for img in images]
                max_w = max(widths)
                padded = []
                for img in images:
                    if img.shape[-1] < max_w:
                        pad_w = max_w - img.shape[-1]
                        img = F.pad(img, (0, pad_w, 0, 0))
                    padded.append(img.unsqueeze(0))
                batch_imgs = torch.cat(padded, dim=0).to(device)
                logits = model(batch_imgs)  # (B,T,V)
                log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)
                input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=device)
                encoded = converter.encode(texts)
                target_lengths = torch.tensor([e.numel() for e in encoded], device=device, dtype=torch.long)
                if sum(t.item() for t in target_lengths) == 0:
                    continue
                targets = torch.cat(encoded).to(device)
                loss = criterion(log_probs, targets, input_lengths, target_lengths)
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                total_loss += loss.item()
                total_samples += 1
                # Decode
                if beam_decoder is not None:
                    probs = torch.exp(log_probs.permute(1, 0, 2))
                    pred_texts = beam_decoder.decode(torch.log(probs + 1e-8))
                else:
                    pred_texts = greedy_decode(log_probs.permute(1, 0, 2), converter)
                all_predictions.extend(pred_texts[:len(texts)])
                all_targets.extend(texts)
            else:
                # Fallback per-sample path
                for img, text in zip(images, texts):
                    try:
                        img = img.unsqueeze(0).to(device)
                        encoded = converter.encode([text])[0]
                        if len(encoded) == 0:
                            continue
                        labels_tensor = encoded.long().to(device) if isinstance(encoded, torch.Tensor) else torch.tensor(encoded, dtype=torch.long, device=device)
                        if labels_tensor.dim() > 1:
                            labels_tensor = labels_tensor.flatten()
                        label_length = torch.tensor([len(labels_tensor)], dtype=torch.long, device=device)
                        logits = model(img)
                        log_probs = F.log_softmax(logits, dim=2).permute(1, 0, 2)
                        input_length = torch.tensor([logits.size(1)], dtype=torch.long, device=device)
                        loss = criterion(log_probs, labels_tensor, input_length, label_length)
                        if torch.isnan(loss) or torch.isinf(loss):
                            continue
                        total_loss += loss.item(); total_samples += 1
                        if beam_decoder is not None:
                            probs = torch.exp(log_probs.permute(1, 0, 2))
                            pred_texts = beam_decoder.decode(torch.log(probs + 1e-8))
                        else:
                            pred_texts = greedy_decode(log_probs.permute(1, 0, 2), converter)
                        all_predictions.extend(pred_texts)
                        all_targets.append(text)
                    except Exception:
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

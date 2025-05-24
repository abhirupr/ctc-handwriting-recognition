import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from models.rtlr_model import CTCRecognitionModel
from utils.label_converter import LabelConverter
import os

def collate_fn(batch):
    images, labels = zip(*batch)
    images = torch.stack(images)
    label_tensors = [torch.tensor(l, dtype=torch.long) for l in labels]
    labels_padded = pad_sequence(label_tensors, batch_first=True, padding_value=0)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)
    return images, labels_padded, label_lengths

def train(
    model, dataloader, val_dataloader, converter, device, optimizer, config
):
    model.to(device)
    criterion = nn.CTCLoss(blank=converter.blank_idx, zero_infinity=True)

    # Create directories for saving models
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.BEST_MODEL_DIR, exist_ok=True)

    best_metric = float("inf") if config.EVAL_STRATEGY == "loss" else 0
    saved_checkpoints = []

    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        for step, (images, labels_padded, label_lengths) in enumerate(dataloader):
            images = images.to(device)
            labels_padded = labels_padded.to(device)
            label_lengths = label_lengths.to(device)

            logits = model(images)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)  # (T, N, C)
            input_lengths = torch.full(
                size=(logits.size(0),), fill_value=logits.size(1), dtype=torch.long
            ).to(device)

            loss = criterion(log_probs, labels_padded, input_lengths, label_lengths)
            loss = loss / config.GRADIENT_ACCUMULATION_STEPS
            loss.backward()

            if (step + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

        # Save the best model
        if epoch % config.EVAL_STEP == 0:
            val_metric = evaluate(model, val_dataloader, criterion, device, config)
            print(f"Validation {config.EVAL_STRATEGY}: {val_metric:.4f}")

            if (
                (config.EVAL_STRATEGY == "loss" and val_metric < best_metric)
                or (config.EVAL_STRATEGY == "accuracy" and val_metric > best_metric)
            ):
                best_metric = val_metric
                best_model_path = os.path.join(config.BEST_MODEL_DIR, config.BEST_MODEL_NAME)
                torch.save(model.state_dict(), best_model_path)
                print(f"Best model saved to {best_model_path}")

        # Save model checkpoint for the current epoch
        checkpoint_path = os.path.join(
            config.SAVE_DIR, f"{config.CHECKPOINT_PREFIX}_{epoch + 1}.pth"
        )
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

        # Manage saved checkpoints to enforce save limit
        saved_checkpoints.append(checkpoint_path)
        if len(saved_checkpoints) > config.SAVE_LIMIT:
            oldest_checkpoint = saved_checkpoints.pop(0)
            if os.path.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)
                print(f"Removed old checkpoint: {oldest_checkpoint}")


def evaluate(model, dataloader, criterion, device, config):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels_padded, label_lengths in dataloader:
            images = images.to(device)
            labels_padded = labels_padded.to(device)
            label_lengths = label_lengths.to(device)

            logits = model(images)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)  # (T, N, C)
            input_lengths = torch.full(
                size=(logits.size(0),), fill_value=logits.size(1), dtype=torch.long
            ).to(device)

            loss = criterion(log_probs, labels_padded, input_lengths, label_lengths)
            total_loss += loss.item()

            if config.EVAL_STRATEGY == "accuracy":
                predictions = log_probs.argmax(dim=-1).permute(1, 0)  # (N, T)
                for pred, label in zip(predictions, labels_padded):
                    pred_text = converter.decode(pred.tolist())
                    label_text = converter.decode(label.tolist())
                    if pred_text == label_text:
                        total_correct += 1
                    total_samples += 1

    if config.EVAL_STRATEGY == "loss":
        return total_loss / len(dataloader)
    elif config.EVAL_STRATEGY == "accuracy":
        return total_correct / total_samples

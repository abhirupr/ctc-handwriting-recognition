import sys
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.iam_dataset import IAMDataset
from models.model import CTCTransformerModel
from utils.label_converter import LabelConverter
from training.train_loop import train_model
from training.metrics import calculate_cer, calculate_word_accuracy
from config import (DATA_DIR, XML_PATH, VOCAB, BATCH_SIZE, EPOCHS, DEVICE, 
                   LEARNING_RATE, OPTIMIZER, MODEL_CONFIG, TEST_SIZE)
from sklearn.model_selection import train_test_split
import config

def collate_fn(batch):
    """Custom collate function for IAM dataset"""
    # Handle both (image, text) and (image, text, length) formats
    if len(batch[0]) == 2:
        # Dataset returns (image, text) pairs
        images, texts = zip(*batch)
        # Calculate text lengths
        lengths = [len(text) for text in texts]
        return list(images), list(texts), list(lengths)
    elif len(batch[0]) == 3:
        # Dataset returns (image, text, length) tuples
        images, texts, lengths = zip(*batch)
        return list(images), list(texts), list(lengths)
    else:
        raise ValueError(f"Unexpected batch item format. Expected 2 or 3 elements, got {len(batch[0])}")

# Setup dataset and dataloader
dataset = IAMDataset(DATA_DIR, XML_PATH)
converter = LabelConverter(VOCAB)

# Split dataset into train and validation sets
train_samples, val_samples = train_test_split(dataset.samples, test_size=TEST_SIZE, random_state=42)
train_dataset = IAMDataset(DATA_DIR, XML_PATH, samples=train_samples, 
                          augment=True, augment_config=config.DATA_AUGMENTATION)
val_dataset = IAMDataset(DATA_DIR, XML_PATH, samples=val_samples, 
                        augment=False)  # No augmentation for validation

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

pm = config.PAPER_MODEL
model = CTCTransformerModel(vocab_size=MODEL_CONFIG['vocab_size'],
                            depth=pm.get('depth', 16),
                            dropout=pm.get('dropout', 0.1),
                            chunk_width=pm.get('chunk_width', 320),
                            pad=pm.get('pad', 32))
print("Using model: CTCTransformerModel (paper-aligned)")

# Select optimizer
if OPTIMIZER == "sgd":
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
elif OPTIMIZER == "adamw":
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Choose which scheduler to use
USE_PLATEAU_SCHEDULER = True  # Set to True for ReduceLROnPlateau

if USE_PLATEAU_SCHEDULER:
    # ReduceLROnPlateau - adaptive based on validation CER
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.7,      # Less aggressive reduction
    patience=8,      # More patience
    min_lr=1e-5,     # Higher minimum LR
    verbose=True
    )
else:
    # CosineAnnealingWarmRestarts - time-based
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,        # Restart every 10 epochs
        T_mult=2,      # Double the restart period each time
        eta_min=1e-6   # Minimum learning rate
    )

# Train the model
if __name__ == "__main__":
    device = torch.device(DEVICE)
    
    print("🚀 Starting training with early stopping...")
    if hasattr(config, 'EARLY_STOPPING') and config.EARLY_STOPPING.get('enabled', False):
        es_config = config.EARLY_STOPPING
        print(f"   Patience: {es_config.get('patience', 7)} epochs")
        print(f"   Min delta: {es_config.get('min_delta', 0.0001)}")
        print(f"   Mode: {es_config.get('mode', 'max')}")
        print(f"   Restore best weights: {es_config.get('restore_best_weights', True)}")
    
    # Start training with comprehensive metrics and early stopping
    model = train_model(model, train_dataloader, val_dataloader, converter, device, optimizer, config, scheduler)
    
    print("\n🎉 Training complete!")

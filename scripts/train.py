import sys
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.iam_dataset import IAMDataset
from models.rtlr_model import CTCRecognitionModel
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
train_dataset = IAMDataset(DATA_DIR, XML_PATH, samples=train_samples)
val_dataset = IAMDataset(DATA_DIR, XML_PATH, samples=val_samples)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# Initialize model
model = CTCRecognitionModel(**MODEL_CONFIG)

# Select optimizer
if OPTIMIZER == "sgd":
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
elif OPTIMIZER == "adamw":
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

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
    model = train_model(model, train_dataloader, val_dataloader, converter, device, optimizer, config)
    
    print("\n🎉 Training complete!")

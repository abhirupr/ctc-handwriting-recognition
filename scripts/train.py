import sys
import os
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from dataset.iam_dataset import IAMDataset
from models.rtlr_model import CTCRecognitionModel
from utils.label_converter import LabelConverter
from train.train_loop import train_model, collate_fn  # Updated import
from config import DATA_DIR, XML_PATH, VOCAB, BATCH_SIZE, EPOCHS, DEVICE, LEARNING_RATE, OPTIMIZER, MODEL_CONFIG, TEST_SIZE
from sklearn.model_selection import train_test_split
import config

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
    raise ValueError(f"Unsupported optimizer: {OPTIMIZER}")

# Train the model
train_model(model, train_dataloader, val_dataloader, converter, DEVICE, optimizer, config)  # Updated function call

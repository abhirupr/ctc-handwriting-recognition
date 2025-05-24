import os
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, AdamW
from dataset.iam_dataset import IAMDataset
from models.rtlr_model import CTCRecognitionModel
from utils.label_converter import LabelConverter
from train.train_loop import train
from config import DATA_DIR, XML_PATH, VOCAB, BATCH_SIZE, EPOCHS, DEVICE, LEARNING_RATE, OPTIMIZER, MODEL_CONFIG

# Setup dataset and dataloader
dataset = IAMDataset(DATA_DIR, XML_PATH)
converter = LabelConverter(VOCAB)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: (zip(*b)))

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
train(model, dataloader, converter, DEVICE, optimizer, epochs=EPOCHS)

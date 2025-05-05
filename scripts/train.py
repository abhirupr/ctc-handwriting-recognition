import os
import torch
from torch.utils.data import DataLoader
from dataset.iam_dataset import IAMDataset
from models.rtlr_model import CTCRecognitionModel
from utils.label_converter import LabelConverter
from train.train_loop import train

# Configuration
DATA_DIR = "path/to/iam_dataset"
XML_PATH = "path/to/annotations.xml"
VOCAB = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,;:!?'\"-—()[]"
    "+*/=&$€£%#@"
)
BATCH_SIZE = 16
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Setup
dataset = IAMDataset(DATA_DIR, XML_PATH)
converter = LabelConverter(VOCAB)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: (zip(*b)))

# Model (you should replace dimensions with those matching your actual implementation)
model = CTCRecognitionModel(
    img_channel=1,
    cnn_out_channels=256,
    num_encoder_layers=16,
    num_heads=8,
    d_model=256,
    dim_feedforward=512,
    vocab_size=len(converter.vocab)  # +1 for CTC blank
)

# Training
train(model, dataloader, converter, DEVICE, epochs=EPOCHS)

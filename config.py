import torch

# Dataset configuration
DATA_DIR = "path/to/iam_dataset"
XML_PATH = "path/to/annotations.xml"
VOCAB = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,;:!?'\"-—()[]"
    "+*/=&$€£%#@"
)

# Training configuration
BATCH_SIZE = 16
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
OPTIMIZER = "sgd"  # Options: "sgd", "adamw"

# Model configuration
MODEL_CONFIG = {
    "img_channel": 1,
    "cnn_out_channels": 256,
    "num_encoder_layers": 16,
    "num_heads": 8,
    "d_model": 256,
    "dim_feedforward": 512,
    "vocab_size": len(VOCAB),
}
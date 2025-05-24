import torch

# Dataset configuration
DATA_DIR = "data/iam_dataset/lines"
XML_PATH = "data/iam_dataset/xml"  # Point to the xml directory, not a specific file
VOCAB = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,;:!?'\"-—()[]"
    "+*/=&$€£%#@"
)
TEST_SIZE = 0.2  # Fraction of the dataset to use for testing

# Training configuration
EPOCHS = 50  # Increase since early stopping will handle when to stop
LEARNING_RATE = 0.001
BATCH_SIZE = 16
EVAL_STEP = 1  # Evaluate every epoch
EVAL_STRATEGY = "accuracy"  # "accuracy" or "loss"
BEST_MODEL_DIR = "resources/best_model"
BEST_MODEL_NAME = "best_model.pth"
CHECKPOINT_PREFIX = "model_epoch"

# Early Stopping Configuration
EARLY_STOPPING = {
    "enabled": True,
    "patience": 7,          # Stop after 7 epochs without improvement
    "min_delta": 0.0001,    # Minimum improvement threshold
    "mode": "max",          # "max" for accuracy, "min" for loss
    "restore_best_weights": True,
    "baseline": None,       # Optional baseline metric
    "verbose": True
}

# Model configuration
MODEL_CONFIG = {
    "vocab_size": len(VOCAB),
    "chunk_width": 320,
    "pad": 32,
}

# Language Model Configuration
USE_LANGUAGE_MODEL = True
LM_MODEL_PATH = "model.binary"
LM_ALPHA = 0.5
LM_BETA = 1.0
BEAM_WIDTH = 100
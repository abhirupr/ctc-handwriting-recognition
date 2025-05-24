import torch

# Dataset configuration
DATA_DIR = "data/iam_dataset/lines"
XML_PATH = "data/iam_dataset/xml"  
VOCAB = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,;:!?'\"-—()[]"
    "+*/=&$€£%#@"
)
TEST_SIZE = 0.2  # Fraction of the dataset to use for testing

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training configuration - SINGLE SET OF VALUES
EPOCHS = 30  
LEARNING_RATE = 0.0005  
BATCH_SIZE = 12  
OPTIMIZER = "adam"  
EVAL_STEP = 1  
EVAL_STRATEGY = "loss"  

# Learning rate scheduling
USE_LR_SCHEDULER = True
LR_SCHEDULER_STEP_SIZE = 5  
LR_SCHEDULER_GAMMA = 0.7  

# Model saving configuration
SAVE_DIR = "resources/checkpoints"  
SAVE_LIMIT = 5  
BEST_MODEL_DIR = "resources/best_model"
BEST_MODEL_NAME = "best_model.pth"
CHECKPOINT_PREFIX = "model_epoch"

# Early Stopping Configuration - CER-based
EARLY_STOPPING = {
    "enabled": True,
    "patience": 8,          
    "min_delta": 1.0,       
    "mode": "min",          
    "restore_best_weights": True,
    "baseline": None,
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

# Debugging configuration
DEBUG_MODE = False  

# Data augmentation
DATA_AUGMENTATION = {
    "rotation": 2.0,
    "shear": 0.1,
    "elastic_transform": True,
    "noise": 0.02
}

# Regularization
DROPOUT_RATE = 0.4  
WEIGHT_DECAY = 1e-4  

# Create necessary directories
import os
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

# Print configuration summary
print(f"🔧 Configuration loaded:")
print(f"   Device: {DEVICE}")
print(f"   Vocabulary size: {len(VOCAB)}")
print(f"   Training samples: IAM dataset")
print(f"   Language Model: {'Enabled' if USE_LANGUAGE_MODEL else 'Disabled'}")
print(f"   Early Stopping: {'Enabled' if EARLY_STOPPING['enabled'] else 'Disabled'}")
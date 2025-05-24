import torch

# Dataset configuration
DATA_DIR = "data/iam_dataset/lines"
XML_PATH = "data/iam_dataset/xml/lines.xml"  # Updated to point to the actual XML file
VOCAB = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    " .,;:!?'\"-—()[]"
    "+*/=&$€£%#@"
)
TEST_SIZE = 0.2  # Fraction of the dataset to use for testing

# Training configuration
BATCH_SIZE = 16
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-3
OPTIMIZER = "sgd"  # Options: "sgd", "adamw"
GRADIENT_ACCUMULATION_STEPS = 1  # Number of steps for gradient accumulation

# Model save configuration
SAVE_DIR = "resources/checkpoints"
SAVE_LIMIT = 5  # Maximum number of checkpoints to keep
EVAL_STEP = 1  # Evaluate the model every `EVAL_STEP` epochs
EVAL_STRATEGY = "loss"  # Options: "loss", "accuracy"
BEST_MODEL_DIR = "resources/best_model"  # Directory for the best model
BEST_MODEL_NAME = "best_model.pth"
CHECKPOINT_PREFIX = "model_epoch"

# Model configuration
MODEL_CONFIG = {
    "vocab_size": len(VOCAB),
    "img_channel": 1,
    "cnn_out_channels": 256,
    "num_encoder_layers": 16,
    "num_heads": 8,
    "d_model": 256,
    "dim_feedforward": 512,
    "chunk_width": 320,
    "pad": 32,
}
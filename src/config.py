"""
Configuration settings for the DARN model implementation.

Contains all hyperparameters, paths, and model configuration settings.
"""

from pathlib import Path

# dataset paths
DATA_ROOT = Path("data/deepfashion")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"

# model parameters
NUM_ATTRIBUTES = 1000  # deepfashion fine-grained attribute count
EMBEDDING_DIM = 512
BACKBONE = "resnet34"  # initial backbone, we'll experiment with others later
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 100

# training settings
DEVICE = "cuda"  # will be updated in runtime based on availability
SAVE_DIR = Path("checkpoints")
LOG_DIR = Path("logs")

# data augmentation parameters
INPUT_SIZE = 224  # standard input size for most pretrained models 
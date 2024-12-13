"""
Configuration System for Model Training and Evaluation.

This Module Defines the Configuration Classes Used to
Control Model Training, Data Loading, and Evaluation.
"""

from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class DataConfig:
    """
    Dataset Configuration Parameters.
    
    Attributes:
        - Data Paths and File Names
        - Batch Processing Settings
        - Dataset Split Information
    """
    data_dir: Path = Path("data_deepfashion")
    batch_size: int = 128  # batch size for training
    num_workers: int = 4  # number of data loading workers
    
    # dataset annotation files
    attribute_list_file: str = "Anno_fine/list_attr_cloth.txt"
    
    # dataset split files
    train_split: str = "Anno_fine/train.txt"
    val_split: str = "Anno_fine/val.txt"
    test_split: str = "Anno_fine/test.txt"
    
    # attribute annotation files
    train_attr: str = "Anno_fine/train_attr.txt"
    val_attr: str = "Anno_fine/val_attr.txt"
    test_attr: str = "Anno_fine/test_attr.txt"

@dataclass
class ModelConfig:
    """
    Model Architecture Configuration.
    
    Attributes:
        - Network Architecture Settings
        - Loss Function Parameters
        - Regularization Settings
    """
    # network architecture
    hidden_dims: list = (512, 256)  # hidden layer dimensions
    dropout: float = 0.2  # dropout probability
    
    # loss function parameters
    pos_weight: float = 2.0  # weight for positive labels
    label_smoothing: float = 0.1  # label smoothing factor
    
    # regularization settings
    weight_decay: float = 0.01  # l2 regularization

@dataclass
class TrainingConfig:
    """
    Training Process Configuration.
    
    Attributes:
        - Training Hyperparameters
        - Optimization Settings
        - Hardware Configuration
    """
    epochs: int = 50  # total training epochs
    learning_rate: float = 1e-3  # initial learning rate
    warmup_epochs: int = 2  # learning rate warmup
    gradient_clip_val: float = 1.0  # gradient clipping
    log_every_n_steps: int = 10  # logging frequency
    early_stopping_patience: int = 5  # early stopping
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # training device

@dataclass
class Config:
    """
    Main Configuration Container.
    
    Attributes:
        - Data Configuration
        - Model Configuration
        - Training Configuration
        - Random Seed
    """
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    seed: int = 42  # random seed for reproducibility
 
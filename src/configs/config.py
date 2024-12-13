"""Configuration Module for Attribute Classification Model Training and Evaluation"""

from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class DataConfig:
    data_dir: Path = Path("data_deepfashion")
    batch_size: int = 128  # increased since we're not loading images
    num_workers: int = 4
    
    # dataset files
    attribute_list_file: str = "Anno_fine/list_attr_cloth.txt"
    
    # split files
    train_split: str = "Anno_fine/train.txt"
    val_split: str = "Anno_fine/val.txt"
    test_split: str = "Anno_fine/test.txt"
    
    # attribute files
    train_attr: str = "Anno_fine/train_attr.txt"
    val_attr: str = "Anno_fine/val_attr.txt"
    test_attr: str = "Anno_fine/test_attr.txt"

@dataclass
class ModelConfig:
    # model architecture
    hidden_dims: list = (512, 256)  # mlp hidden dimensions
    dropout: float = 0.2
    
    # loss function settings
    pos_weight: float = 2.0  # weight for positive labels since they're usually rarer
    label_smoothing: float = 0.1
    
    # regularization
    weight_decay: float = 0.01

@dataclass
class TrainingConfig:
    epochs: int = 50
    learning_rate: float = 1e-3
    warmup_epochs: int = 2
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 10
    early_stopping_patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class Config:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    seed: int = 42
 
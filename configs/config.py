from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class DataConfig:
    """configuration for data loading and processing"""
    data_dir: Path = Path("data_deepfashion")
    image_size: int = 224  # clip default image size
    batch_size: int = 32
    num_workers: int = 4
    
    # dataset files
    category_list_file: str = "list_category_cloth.txt"
    attribute_list_file: str = "list_attr_cloth.txt"
    
    # split files
    train_split: str = "train.txt"
    val_split: str = "val.txt"
    test_split: str = "test.txt"
    
    # category files
    train_category: str = "train_cate.txt"
    val_category: str = "val_cate.txt"
    test_category: str = "test_cate.txt"
    
    # attribute files
    train_attr: str = "train_attr.txt"
    val_attr: str = "val_attr.txt"
    test_attr: str = "test_attr.txt"
    
    # bounding box files
    train_bbox: str = "train_bbox.txt"
    val_bbox: str = "val_bbox.txt"
    test_bbox: str = "test_bbox.txt"

@dataclass
class ModelConfig:
    """configuration for model architecture"""
    # backbone settings
    clip_model: str = "ViT-B/32"
    hidden_dim: int = 512
    
    # regularization
    dropout: float = 0.1  # reduced dropout for stability
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    
    # architecture settings
    num_attention_heads: int = 8
    num_shared_layers: int = 2
    
@dataclass
class TrainingConfig:
    """configuration for training process"""
    epochs: int = 30  # reduced epochs
    learning_rate: float = 1e-4  # reduced learning rate
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01  # reduced weight decay
    warmup_epochs: int = 2
    log_every_n_steps: int = 10
    gradient_clip_val: float = 1.0  # add gradient clipping
    
@dataclass
class Config:
    """main configuration class combining all sub-configs"""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    seed: int = 42
    device: str = "mps"  # for m1 mac
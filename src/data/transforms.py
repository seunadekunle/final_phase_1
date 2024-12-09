"""Data transforms for model training and evaluation"""

import torchvision.transforms as T
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from PIL import Image

def get_train_transform():
    """Get training data transforms with augmentation"""
    return T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        T.ToTensor(),
        T.Lambda(lambda x: torch.clamp(x, 0, 1)),  # ensure values are in [0,1]
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transform():
    """Get validation/test data transforms"""
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Lambda(lambda x: torch.clamp(x, 0, 1)),  # ensure values are in [0,1]
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 
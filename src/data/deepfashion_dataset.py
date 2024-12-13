"""
deepfashion dataset loading and preprocessing.

handles loading and preprocessing of the deepfashion dataset
for fine-grained attribute prediction.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from typing import Dict, Tuple, List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = logging.getLogger(__name__)

class DeepFashionDataset(Dataset):
    """
    deepfashion dataset for fine-grained attribute prediction.
    
    features:
    - loads images and attribute annotations
    - supports train/val/test splits
    - implements data augmentation for training
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        """
        initialize dataset.
        
        args:
            root_dir: path to deepfashion dataset root
            split: dataset split ('train', 'val', or 'test')
            transform: optional albumentations transform pipeline
            target_size: target image size (height, width)
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform or self._get_default_transform(split, target_size)
        
        # load annotations
        self.img_paths, self.attributes = self._load_annotations()
        logger.info(f"Loaded {len(self)} images for {split} split")
        
    def _get_default_transform(
        self,
        split: str,
        target_size: Tuple[int, int]
    ) -> A.Compose:
        """get default transformation pipeline."""
        if split == "train":
            return A.Compose([
                A.RandomResizedCrop(*target_size, scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(target_size[0], target_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def _load_annotations(self) -> Tuple[List[Path], torch.Tensor]:
        """
        load and parse dataset annotations.
        
        returns:
            tuple containing:
            - list of image paths
            - tensor of attribute labels
        """
        # load split-specific files
        split_file = self.root_dir / "Anno_fine" / f"{self.split}.txt"
        attr_file = self.root_dir / "Anno_fine" / f"{self.split}_attr.txt"
        
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        if not attr_file.exists():
            raise FileNotFoundError(f"Attribute file not found: {attr_file}")
            
        # load image paths and attributes
        with open(split_file, 'r') as f:
            img_paths = []
            for line in f:
                img_path = line.strip()
                if img_path: 
                    # construct the correct path relative to root_dir
                    full_path = self.root_dir / img_path
                    img_paths.append(full_path)
            
        with open(attr_file, 'r') as f:
            attributes = []
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1: 
                    attr_values = [int(x) for x in parts[1:]]  # skip image name
                    # convert from -1/1 to 0/1
                    attr_values = [(x + 1) // 2 for x in attr_values]
                    attributes.append(attr_values)
        
        # convert to tensor
        attributes = torch.tensor(attributes, dtype=torch.float32)
        
        # verify matching lengths
        if len(img_paths) != len(attributes):
            raise ValueError(
                f"Number of images ({len(img_paths)}) does not match "
                f"number of attribute labels ({len(attributes)})"
            )
        
        # filter out missing images
        valid_indices = []
        valid_paths = []
        for i, img_path in enumerate(img_paths):
            if img_path.exists():
                valid_indices.append(i)
                valid_paths.append(img_path)
            else:
                logger.warning(f"Image not found: {img_path}")
        
        # keep only valid images and their attributes
        valid_indices = torch.tensor(valid_indices)
        attributes = attributes[valid_indices]
        
        logger.info(f"Loaded {len(valid_paths)} valid images out of {len(img_paths)} total")
        
        return valid_paths, attributes
    
    def __len__(self) -> int:
        """get dataset length."""
        return len(self.img_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        get a single sample.
        
        args:
            idx: sample index
            
        returns:
            dict containing:
            - image: preprocessed image tensor
            - attributes: attribute labels
        """
        # load and preprocess image
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # apply transforms
        if self.transform:
            image = self.transform(image=np.array(image))["image"]
        
        return {
            "image": image,
            "attributes": self.attributes[idx]
        }

def create_dataloaders(
    root_dir: str,
    batch_size: int,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (224, 224)
) -> Dict[str, DataLoader]:
    """
    create dataloader instances for all splits.
    
    args:
        root_dir: path to dataset root
        batch_size: batch size
        num_workers: number of data loading workers
        target_size: target image size
        
    returns:
        dict containing dataloaders for train, val, and test splits
    """
    dataloaders = {}
    
    for split in ["train", "val", "test"]:
        dataset = DeepFashionDataset(
            root_dir=root_dir,
            split=split,
            target_size=target_size
        )
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True
        )
        
    return dataloaders

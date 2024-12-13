"""
DeepFashion Dataset Management Module.

This Module Provides a Unified Interface for Managing Dataset Splits,
DataLoaders, and Data Processing for the DeepFashion Dataset.
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader

from .dataset import DeepFashionDataset

class DeepFashionDataModule:
    """
    Data Management System for DeepFashion Dataset.
    
    Features:
        - Dataset Loading and Preprocessing
        - DataLoader Configuration
        - Split Management (Train/Val/Test)
        - Dynamic Image Size Updates
    """
    
    def __init__(self, config, image_size=None):
        """
        Initialize Data Module Components.
        
        Args:
            config: Configuration object containing:
                - data_dir: Dataset directory path
                - batch_size: Training batch size
                - num_workers: DataLoader workers
                - image_size: Image dimensions
                - category_list_file: Category list path
                - attribute_list_file: Attribute list path
                - train_split: Training split file
                - val_split: Validation split file
                - test_split: Test split file
                - train_category: Training category labels
                - val_category: Validation category labels
                - test_category: Test category labels
                - train_attr: Training attribute labels
                - val_attr: Validation attribute labels
                - test_attr: Test attribute labels
                - train_bbox: Training bounding boxes
                - val_bbox: Validation bounding boxes
                - test_bbox: Test bounding boxes
            image_size: Optional override for config image size
        """
        self.config = config
        self.image_size = image_size or config.image_size
        self.train_dataset: Optional[DeepFashionDataset] = None
        self.val_dataset: Optional[DeepFashionDataset] = None
        self.test_dataset: Optional[DeepFashionDataset] = None
        
    def setup(self):
        """Set Up All Dataset Splits and Initialize Properties."""
        # create training dataset
        self.train_dataset = DeepFashionDataset(
            data_dir=self.config.data_dir,
            split_file=self.config.train_split,
            category_file=self.config.train_category,
            attribute_file=self.config.train_attr,
            bbox_file=self.config.train_bbox,
            category_list_file=self.config.category_list_file,
            attribute_list_file=self.config.attribute_list_file,
            image_size=self.image_size
        )
        
        # create validation dataset
        self.val_dataset = DeepFashionDataset(
            data_dir=self.config.data_dir,
            split_file=self.config.val_split,
            category_file=self.config.val_category,
            attribute_file=self.config.val_attr,
            bbox_file=self.config.val_bbox,
            category_list_file=self.config.category_list_file,
            attribute_list_file=self.config.attribute_list_file,
            image_size=self.image_size
        )
        
        # create test dataset
        self.test_dataset = DeepFashionDataset(
            data_dir=self.config.data_dir,
            split_file=self.config.test_split,
            category_file=self.config.test_category,
            attribute_file=self.config.test_attr,
            bbox_file=self.config.test_bbox,
            category_list_file=self.config.category_list_file,
            attribute_list_file=self.config.attribute_list_file,
            image_size=self.image_size
        )
        
        # store dataset properties for model
        self.num_categories = self.train_dataset.num_categories
        self.num_attributes = self.train_dataset.num_attributes
        
    def update_image_size(self, new_size):
        """
        Update Image Size for All Dataset Splits.
        
        Args:
            new_size: New image dimensions to use
        """
        self.image_size = new_size
        if self.train_dataset:
            self.train_dataset.image_size = new_size
        if self.val_dataset:
            self.val_dataset.image_size = new_size
        if self.test_dataset:
            self.test_dataset.image_size = new_size
        
    def train_dataloader(self):
        """Create Training DataLoader with Shuffling."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
    def val_dataloader(self):
        """Create Validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
    def test_dataloader(self):
        """Create Test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
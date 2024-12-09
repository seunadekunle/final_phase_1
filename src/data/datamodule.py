"""data module for managing deepfashion dataset splits and dataloaders"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from .dataset import DeepFashionDataset

class DeepFashionDataModule:
    """data module for managing deepfashion dataset loading and preprocessing
    
    args:
        config: configuration object containing data parameters
        image_size: optional override for image size (default: None)
    """
    
    def __init__(self, config, image_size=None):
        """initialize data module
        
        args:
            config: configuration object with data parameters
                - data_dir: path to dataset directory
                - batch_size: batch size for dataloaders
                - num_workers: number of workers for dataloaders
                - image_size: size to resize images to
                - category_list_file: path to category list file
                - attribute_list_file: path to attribute list file
                - train_split: name of training split file
                - val_split: name of validation split file
                - test_split: name of test split file
                - train_category: name of training category labels file
                - val_category: name of validation category labels file
                - test_category: name of test category labels file
                - train_attr: name of training attribute labels file
                - val_attr: name of validation attribute labels file
                - test_attr: name of test attribute labels file
                - train_bbox: name of training bbox file
                - val_bbox: name of validation bbox file
                - test_bbox: name of test bbox file
            image_size: optional override for config image size
        """
        self.config = config
        self.image_size = image_size or config.image_size
        self.train_dataset: Optional[DeepFashionDataset] = None
        self.val_dataset: Optional[DeepFashionDataset] = None
        self.test_dataset: Optional[DeepFashionDataset] = None
        
    def setup(self):
        """set up all dataset splits"""
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
        
        # store dataset properties for model initialization
        self.num_categories = self.train_dataset.num_categories
        self.num_attributes = self.train_dataset.num_attributes
        
    def update_image_size(self, new_size):
        """update image size for all datasets
        
        args:
            new_size: new image size to use
        """
        self.image_size = new_size
        if self.train_dataset:
            self.train_dataset.image_size = new_size
        if self.val_dataset:
            self.val_dataset.image_size = new_size
        if self.test_dataset:
            self.test_dataset.image_size = new_size
        
    def train_dataloader(self):
        """create training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
    def val_dataloader(self):
        """create validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
    def test_dataloader(self):
        """create test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
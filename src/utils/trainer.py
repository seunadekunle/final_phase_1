"""trainer class for model training and evaluation"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import wandb
import matplotlib.pyplot as plt
from pathlib import Path
import json

from .wandb_logger import WandbLogger

logger = logging.getLogger(__name__)

class Trainer:
    """trainer class for style classifier"""
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        num_epochs,
        early_stopping_patience=5,
        wandb_run=None
    ):
        """initialize trainer
        
        args:
            model: model to train
            train_loader: training data loader
            val_loader: validation data loader
            criterion: loss function for training
            optimizer: optimizer for training
            scheduler: learning rate scheduler
            device: device to use for training
            num_epochs: number of epochs to train
            early_stopping_patience: patience for early stopping
            wandb_run: optional wandb run
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.wandb_run = wandb_run
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
    def train_epoch(self):
        """train for one epoch
        
        returns:
            tuple of training metrics
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            targets = batch['attributes'].to(self.device)
            
            # Generate positive/negative pairs for contrastive learning
            batch_size = images.size(0)
            if batch_size > 1:
                # Get style attributes
                style_attrs = targets[:, 23:26]
                
                # Find samples with similar and different styles
                style_sims = torch.matmul(style_attrs, style_attrs.t())
                similar_pairs = torch.where(style_sims > 0.5)
                different_pairs = torch.where(style_sims < 0.2)
                
                # Randomly select pairs if available
                if len(similar_pairs[0]) > 0 and len(different_pairs[0]) > 0:
                    pos_idx = torch.randint(0, len(similar_pairs[0]), (batch_size,))
                    neg_idx = torch.randint(0, len(different_pairs[0]), (batch_size,))
                    
                    positive_pairs = (
                        images[similar_pairs[0][pos_idx]],
                        images[similar_pairs[1][pos_idx]]
                    )
                    negative_pairs = (
                        images[different_pairs[0][neg_idx]],
                        images[different_pairs[1][neg_idx]]
                    )
                    
                    # Get embeddings for pairs
                    with torch.no_grad():
                        pos_out = self.model(positive_pairs[0])
                        neg_out = self.model(negative_pairs[0])
                        style_pairs = (
                            pos_out['style_embedding'],
                            neg_out['style_embedding']
                        )
                else:
                    style_pairs = None
            else:
                style_pairs = None
            
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            
            # Compute loss with contrastive learning
            loss, loss_dict = self.model.compute_loss(
                outputs,
                targets,
                style_pairs=style_pairs
            )
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            predictions = torch.sigmoid(outputs['predictions'])
            correct += ((predictions > 0.5) == targets).float().sum().item()
            total += targets.numel()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'avg_loss': total_loss / (batch_idx + 1)
            })
            
            # Log to wandb
            if self.wandb_run is not None:
                self.wandb_run.log({
                    'batch_loss': loss.item(),
                    'batch_accuracy': ((predictions > 0.5) == targets).float().mean().item()
                })
                
                # Log individual group losses
                for name, value in loss_dict.items():
                    self.wandb_run.log({f'batch_{name}': value})
        
        return total_loss / len(self.train_loader), correct / total
    
    def validate(self):
        """validate model
        
        returns:
            tuple of validation metrics
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                images = batch['image'].to(self.device)
                targets = batch['attributes'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss
                loss, _ = self.model.compute_loss(outputs, targets)
                
                # Update metrics
                total_loss += loss.item()
                predictions = torch.sigmoid(outputs['predictions'])
                correct += ((predictions > 0.5) == targets).float().sum().item()
                total += targets.numel()
        
        return total_loss / len(self.val_loader), correct / total
    
    def train(self):
        """train model"""
        for epoch in range(self.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Step scheduler with validation loss
            self.scheduler.step(val_loss)
            
            # Log metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }
            
            if self.wandb_run is not None:
                self.wandb_run.log(metrics)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"early stopping triggered after {epoch + 1} epochs")
                break
        
        return metrics 
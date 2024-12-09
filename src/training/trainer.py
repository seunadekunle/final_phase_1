"""trainer module for style classifier with improved stability"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import logging
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

class StyleTrainer:
    """trainer class for style classifier with enhanced stability"""
    
    def __init__(
        self,
        model: nn.Module,
        config,
        train_loader,
        val_loader,
        test_loader,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        logger = None,
        category_loss_weight: float = 1.5,
        attribute_loss_weight: float = 1.0
    ):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        
        # move model to device
        self.device = torch.device(config.training.device)
        self.model = self.model.to(self.device)
        
        # initialize loss functions with label smoothing
        self.category_criterion = nn.CrossEntropyLoss(
            label_smoothing=config.model.label_smoothing
        )
        self.attribute_criterion = nn.BCEWithLogitsLoss()
        
        # tracking best model performance
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # early stopping with increased patience
        self.patience = 7  # increased from 5
        self.patience_counter = 0
        
        # gradient clipping
        self.max_grad_norm = 1.0
        
        # loss weighting
        self.category_weight = category_loss_weight
        self.attribute_weight = attribute_loss_weight
        
    def _compute_loss(self, category_logits, attribute_logits, category_labels, attribute_labels):
        """compute combined loss with stability measures"""
        # compute individual losses
        category_loss = self.category_criterion(category_logits, category_labels)
        attribute_loss = self.attribute_criterion(attribute_logits, attribute_labels)
        
        # combine losses with dynamic weighting
        total_loss = self.category_weight * category_loss + self.attribute_weight * attribute_loss
        
        return total_loss, category_loss, attribute_loss
        
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """train for one epoch with gradient clipping"""
        self.model.train()
        total_loss = 0
        total_category_loss = 0
        total_attribute_loss = 0
        num_batches = len(self.train_loader)
        
        for batch_idx, (images, category_labels, attribute_labels) in enumerate(self.train_loader):
            # move data to device
            images = images.to(self.device)
            category_labels = category_labels.to(self.device)
            attribute_labels = attribute_labels.to(self.device)
            
            # forward pass
            category_logits, attribute_logits = self.model(
                images, category_labels, attribute_labels
            )
            
            # compute losses
            loss, category_loss, attribute_loss = self._compute_loss(
                category_logits, attribute_logits,
                category_labels, attribute_labels
            )
            
            # backward pass with gradient clipping
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            # update metrics
            total_loss += loss.item()
            total_category_loss += category_loss.item()
            total_attribute_loss += attribute_loss.item()
            
            # log progress
            if batch_idx % 10 == 0:
                logger.info(
                    f"Train Epoch: {epoch} [{batch_idx}/{num_batches} "
                    f"({100. * batch_idx / num_batches:.0f}%)]\t"
                    f"Loss: {loss.item():.6f}"
                )
            
            # step scheduler if using OneCycleLR
            if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()
        
        # compute average losses
        avg_loss = total_loss / num_batches
        avg_category_loss = total_category_loss / num_batches
        avg_attribute_loss = total_attribute_loss / num_batches
        
        return {
            "train_loss": avg_loss,
            "train_category_loss": avg_category_loss,
            "train_attribute_loss": avg_attribute_loss,
            "learning_rate": self.optimizer.param_groups[0]["lr"]
        }
        
    @torch.no_grad()
    def evaluate(self, data_loader, prefix: str = "val") -> Dict[str, float]:
        """evaluate model with stability measures"""
        self.model.eval()
        total_loss = 0.0
        total_cat_loss = 0.0
        total_attr_loss = 0.0
        num_batches = len(data_loader)
        
        pbar = tqdm(data_loader, desc=f"Evaluating ({prefix})")
        
        for images, category_labels, attribute_labels in pbar:
            # move batch to device
            images = images.to(self.device)
            category_labels = category_labels.to(self.device)
            attribute_labels = attribute_labels.float().to(self.device)
            
            # forward pass without mixup
            category_logits, attribute_logits = self.model(images)
            
            # compute losses
            loss, cat_loss, attr_loss = self._compute_loss(
                category_logits, attribute_logits,
                category_labels, attribute_labels
            )
            
            # update running metrics
            total_loss += loss.item()
            total_cat_loss += cat_loss.item()
            total_attr_loss += attr_loss.item()
            
            # update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.3f}",
                'cat_loss': f"{cat_loss.item():.3f}",
                'attr_loss': f"{attr_loss.item():.3f}"
            })
            
        # compute average metrics
        avg_loss = total_loss / num_batches
        avg_cat_loss = total_cat_loss / num_batches
        avg_attr_loss = total_attr_loss / num_batches
        
        return {
            f'{prefix}_loss': avg_loss,
            f'{prefix}_category_loss': avg_cat_loss,
            f'{prefix}_attribute_loss': avg_attr_loss
        }
        
    def train(self, start_epoch: int = 0, num_epochs: Optional[int] = None):
        """full training loop with stability measures and epoch range support
        
        args:
            start_epoch: epoch to start training from
            num_epochs: number of epochs to train for (if None, use config)
        """
        logger.info(f"starting training from epoch {start_epoch}...")
        
        # determine number of epochs
        if num_epochs is None:
            num_epochs = self.config.training.epochs
        end_epoch = start_epoch + num_epochs
        
        for epoch in range(start_epoch, end_epoch):
            # train for one epoch
            train_metrics = self._train_epoch(epoch)
            
            # evaluate on validation set
            val_metrics = self.evaluate(self.val_loader)
            
            # log metrics if logger is available
            if self.logger is not None:
                self.logger.log_epoch({
                    'epoch': epoch,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    **train_metrics,
                    **val_metrics
                })
            
            # save best model checkpoint
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.best_epoch = epoch
                self.patience_counter = 0
                if self.logger is not None:
                    self.logger.save_model(
                        self.model,
                        self.optimizer,
                        epoch,
                        self.best_val_loss
                    )
            else:
                self.patience_counter += 1
                
            # early stopping check
            if self.patience_counter >= self.patience:
                logger.info(f"early stopping triggered after {epoch + 1} epochs")
                break
                    
        # save final metrics
        if self.logger is not None:
            self.logger.save_metrics()
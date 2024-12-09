"""wandb logger utilities for style classifier"""

import wandb
import torch
import torch.nn as nn
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class WandbLogger:
    """wandb logger class for style classifier"""
    
    def __init__(self):
        """initialize wandb logger"""
        self.metrics = {}
        
    def log_metrics(self, metrics: dict, step: int = None):
        """log metrics to wandb
        
        args:
            metrics: dict of metrics to log
            step: optional step number
        """
        wandb.log(metrics, step=step)
        self.metrics.update(metrics)
        
    def save_model(self, model: nn.Module, optimizer: torch.optim.Optimizer, 
                  epoch: int, val_loss: float):
        """save model checkpoint to wandb
        
        args:
            model: model to save
            optimizer: optimizer state
            epoch: current epoch
            val_loss: validation loss
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss
        }
        
        # save locally first
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_path = checkpoint_dir / f'model_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # log to wandb
        artifact = wandb.Artifact(
            name=f'model-checkpoint-{epoch}',
            type='model',
            description=f'Model checkpoint from epoch {epoch}'
        )
        artifact.add_file(str(checkpoint_path))
        wandb.log_artifact(artifact)
        
    def save_metrics(self):
        """save final metrics to wandb"""
        # create summary metrics
        for key, value in self.metrics.items():
            if isinstance(value, (int, float)):
                wandb.run.summary[f'final_{key}'] = value 
"""evaluator utilities for style classifier"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class Evaluator:
    """evaluator class for style classifier"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda'):
        """initialize evaluator
        
        args:
            model: model to evaluate
            device: device to use for evaluation
        """
        self.model = model
        self.device = device
        self.criterion = nn.BCEWithLogitsLoss()
        
    def evaluate(self, dataloader: DataLoader) -> dict:
        """evaluate model on dataloader
        
        args:
            dataloader: dataloader to evaluate on
            
        returns:
            dict of metrics
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Evaluating'):
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions.append(outputs)
                targets.append(labels)
        
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
        
        metrics = {
            'loss': total_loss / len(dataloader),
            'accuracy': ((predictions > 0.5) == targets).float().mean().item()
        }
        
        # compute per-attribute metrics
        for i in range(predictions.shape[1]):
            attr_preds = predictions[:, i]
            attr_targets = targets[:, i]
            metrics[f'attr_{i}_accuracy'] = ((attr_preds > 0.5) == attr_targets).float().mean().item()
        
        return metrics
    
    def get_predictions(self, dataloader: DataLoader) -> tuple:
        """get model predictions on dataloader
        
        args:
            dataloader: dataloader to get predictions for
            
        returns:
            tuple of (predictions, targets)
        """
        self.model.eval()
        predictions = []
        targets = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Getting predictions'):
                images = batch['image'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(images)
                predictions.append(outputs)
                targets.append(labels)
        
        return torch.cat(predictions), torch.cat(targets) 
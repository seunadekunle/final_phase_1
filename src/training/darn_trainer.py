"""trainer implementation for darn model"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import numpy as np
from pathlib import Path

from src.models.darn import DARN
from src.models.darn_loss import DARNLoss

logger = logging.getLogger(__name__)

class DARNTrainer:
    """trainer class for darn model"""
    
    def __init__(
        self,
        model: DARN,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        num_epochs: int = 50,
        device: str = 'cuda',
        output_dir: str = 'outputs',
        attribute_weight: float = 1.0,
        category_weight: float = 1.0,
        ranking_weight: float = 0.5,
        triplet_margin: float = 0.3,
        label_smoothing: float = 0.1,
        num_triplets: int = 5,  # number of triplets to mine per anchor
        mining_strategy: str = 'hardest'  # 'hardest', 'semi-hard', or 'random'
    ):
        """initialize trainer
        
        args:
            model: darn model instance
            train_loader: training data loader
            val_loader: validation data loader
            test_loader: test data loader
            lr: learning rate
            weight_decay: weight decay for optimizer
            num_epochs: number of training epochs
            device: device to use for training
            output_dir: directory to save checkpoints and logs
            attribute_weight: weight for attribute classification loss
            category_weight: weight for category classification loss
            ranking_weight: weight for ranking loss
            triplet_margin: margin for triplet loss
            label_smoothing: label smoothing factor for classification
            num_triplets: number of triplets to mine per anchor
            mining_strategy: strategy for mining triplets
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.num_epochs = num_epochs
        self.num_triplets = num_triplets
        self.mining_strategy = mining_strategy
        
        # create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # initialize loss function
        self.criterion = DARNLoss(
            attribute_weight=attribute_weight,
            category_weight=category_weight,
            ranking_weight=ranking_weight,
            triplet_margin=triplet_margin,
            label_smoothing=label_smoothing
        )
        
        # initialize best metrics
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience = 7
        self.patience_counter = 0
        
    def mine_triplets(
        self,
        embeddings: torch.Tensor,
        attributes: torch.Tensor,
        categories: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """mine triplets for training
        
        args:
            embeddings: feature embeddings (n, d)
            attributes: attribute labels (n, num_attributes)
            categories: category labels (n,)
            
        returns:
            tuple containing:
                - anchor indices
                - positive indices
                - negative indices
        """
        n = len(embeddings)
        device = embeddings.device
        
        # compute pairwise distances
        dist_matrix = torch.cdist(embeddings, embeddings)
        
        # compute attribute similarities
        attr_sim = torch.zeros((n, n), device=device)
        for i in range(n):
            attr_sim[i] = self.model.compute_attribute_similarity(
                attributes[i:i+1].expand(n, -1),
                attributes
            )
            
        # initialize triplet indices
        anchor_idx = []
        pos_idx = []
        neg_idx = []
        
        # mine triplets for each anchor
        for anchor in range(n):
            # find candidates with same category
            pos_candidates = torch.where(categories == categories[anchor])[0]
            pos_candidates = pos_candidates[pos_candidates != anchor]
            
            if len(pos_candidates) == 0:
                continue
                
            # find candidates with different category
            neg_candidates = torch.where(categories != categories[anchor])[0]
            
            if len(neg_candidates) == 0:
                continue
                
            # compute positive distances and similarities
            pos_dists = dist_matrix[anchor, pos_candidates]
            pos_sims = attr_sim[anchor, pos_candidates]
            
            # compute negative distances and similarities
            neg_dists = dist_matrix[anchor, neg_candidates]
            neg_sims = attr_sim[anchor, neg_candidates]
            
            # select positives and negatives based on strategy
            if self.mining_strategy == 'hardest':
                # select hardest positives (furthest) and negatives (closest)
                pos_indices = torch.argsort(pos_dists, descending=True)[:self.num_triplets]
                neg_indices = torch.argsort(neg_dists)[:self.num_triplets]
            
            elif self.mining_strategy == 'semi-hard':
                # select semi-hard negatives (closer than positives but not too close)
                pos_indices = torch.argsort(pos_sims, descending=True)[:self.num_triplets]
                pos_dist = pos_dists[pos_indices].view(-1, 1)
                valid_negs = (neg_dists > pos_dist) & (neg_dists < pos_dist + self.criterion.triplet_margin)
                neg_indices = torch.where(valid_negs)[0][:self.num_triplets]
            
            else:  # random
                pos_indices = torch.randperm(len(pos_candidates))[:self.num_triplets]
                neg_indices = torch.randperm(len(neg_candidates))[:self.num_triplets]
            
            # add triplets
            anchor_idx.extend([anchor] * len(pos_indices))
            pos_idx.extend(pos_candidates[pos_indices])
            neg_idx.extend(neg_candidates[neg_indices])
        
        return (
            torch.tensor(anchor_idx, device=device),
            torch.tensor(pos_idx, device=device),
            torch.tensor(neg_idx, device=device)
        )
        
    def train_epoch(self) -> Dict[str, float]:
        """train one epoch
        
        returns:
            dict of training metrics
        """
        self.model.train()
        total_loss = 0
        metrics = {
            'attribute_loss': 0.,
            'category_loss': 0.,
            'ranking_loss': 0.
        }
        
        for batch in tqdm(self.train_loader, desc='Training'):
            # move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # forward pass
            outputs = self.model(batch['image'])
            
            # mine triplets
            anchor_idx, pos_idx, neg_idx = self.mine_triplets(
                outputs['attribute_embeddings'],
                batch['attributes'],
                batch['categories']
            )
            
            # create triplet batch
            triplet_batch = {
                'attributes': batch['attributes'],
                'categories': batch['categories'],
                'positive_attributes': batch['attributes'][pos_idx],
                'negative_attributes': batch['attributes'][neg_idx],
                'positive_embeddings': {
                    'attribute_embeddings': outputs['attribute_embeddings'][pos_idx],
                    'category_embeddings': outputs['category_embeddings'][pos_idx]
                },
                'negative_embeddings': {
                    'attribute_embeddings': outputs['attribute_embeddings'][neg_idx],
                    'category_embeddings': outputs['category_embeddings'][neg_idx]
                }
            }
            
            # compute loss
            loss, batch_metrics = self.criterion(outputs, triplet_batch)
            
            # backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # update metrics
            total_loss += loss.item()
            for k, v in batch_metrics.items():
                metrics[k] = metrics.get(k, 0.) + v
            
        # average metrics
        num_batches = len(self.train_loader)
        metrics = {k: v / num_batches for k, v in metrics.items()}
        metrics['loss'] = total_loss / num_batches
        
        return metrics
        
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """validate model
        
        returns:
            dict of validation metrics
        """
        self.model.eval()
        total_loss = 0
        metrics = {
            'attribute_loss': 0.,
            'category_loss': 0.,
            'ranking_loss': 0.
        }
        
        for batch in tqdm(self.val_loader, desc='Validating'):
            # move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # forward pass
            outputs = self.model(batch['image'])
            
            # mine triplets
            anchor_idx, pos_idx, neg_idx = self.mine_triplets(
                outputs['attribute_embeddings'],
                batch['attributes'],
                batch['categories']
            )
            
            # create triplet batch
            triplet_batch = {
                'attributes': batch['attributes'],
                'categories': batch['categories'],
                'positive_attributes': batch['attributes'][pos_idx],
                'negative_attributes': batch['attributes'][neg_idx],
                'positive_embeddings': {
                    'attribute_embeddings': outputs['attribute_embeddings'][pos_idx],
                    'category_embeddings': outputs['category_embeddings'][pos_idx]
                },
                'negative_embeddings': {
                    'attribute_embeddings': outputs['attribute_embeddings'][neg_idx],
                    'category_embeddings': outputs['category_embeddings'][neg_idx]
                }
            }
            
            # compute loss
            loss, batch_metrics = self.criterion(outputs, triplet_batch)
            
            # update metrics
            total_loss += loss.item()
            for k, v in batch_metrics.items():
                metrics[k] = metrics.get(k, 0.) + v
            
        # average metrics
        num_batches = len(self.val_loader)
        metrics = {k: v / num_batches for k, v in metrics.items()}
        metrics['loss'] = total_loss / num_batches
        
        return metrics
        
    def train(self) -> Dict[str, List[float]]:
        """train model
        
        returns:
            dict of training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
        
        for epoch in range(self.num_epochs):
            logger.info(f'Epoch {epoch + 1}/{self.num_epochs}')
            
            # train epoch
            train_metrics = self.train_epoch()
            history['train_loss'].append(train_metrics['loss'])
            history['train_metrics'].append(train_metrics)
            
            # validate
            val_metrics = self.validate()
            history['val_loss'].append(val_metrics['loss'])
            history['val_metrics'].append(val_metrics)
            
            # log metrics
            logger.info(
                f'Train Loss: {train_metrics["loss"]:.4f}, '
                f'Val Loss: {val_metrics["loss"]:.4f}'
            )
            
            # check for improvement
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # save best model
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_metrics['loss'],
                        'metrics': val_metrics
                    },
                    self.output_dir / 'best_model.pt'
                )
            else:
                self.patience_counter += 1
                
            # early stopping
            if self.patience_counter >= self.patience:
                logger.info(
                    f'Early stopping triggered after {epoch + 1} epochs. '
                    f'Best validation loss: {self.best_val_loss:.4f} '
                    f'at epoch {self.best_epoch + 1}'
                )
                break
                
        return history
        
    @torch.no_grad()
    def test(self) -> Dict[str, float]:
        """test model on test set
        
        returns:
            dict of test metrics
        """
        if self.test_loader is None:
            logger.warning('No test loader provided')
            return {}
            
        # load best model
        checkpoint = torch.load(self.output_dir / 'best_model.pt')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        metrics = {
            'attribute_loss': 0.,
            'category_loss': 0.,
            'ranking_loss': 0.
        }
        
        for batch in tqdm(self.test_loader, desc='Testing'):
            # move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # forward pass
            outputs = self.model(batch['image'])
            
            # compute metrics (no triplet mining for test set)
            loss, batch_metrics = self.criterion(
                outputs,
                {
                    'attributes': batch['attributes'],
                    'categories': batch['categories']
                }
            )
            
            # update metrics
            for k, v in batch_metrics.items():
                metrics[k] = metrics.get(k, 0.) + v
            
        # average metrics
        num_batches = len(self.test_loader)
        metrics = {k: v / num_batches for k, v in metrics.items()}
        
        return metrics 
"""
Loss functions for the DARN model.

Implements both attribute classification loss and ranking loss
for attribute-aware fashion image retrieval.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class DARNLoss(nn.Module):
    """
    Combined loss function for DARN model.
    
    Combines:
    1. Binary cross-entropy loss for attribute classification
    2. Triplet ranking loss for attribute-aware retrieval
    """
    
    def __init__(
        self,
        margin: float = 0.3,
        lambda_rank: float = 1.0,
        lambda_attr: float = 1.0
    ):
        """
        Initialize DARN loss.
        
        Args:
            margin: Margin for triplet ranking loss
            lambda_rank: Weight for ranking loss
            lambda_attr: Weight for attribute classification loss
        """
        super().__init__()
        self.margin = margin
        self.lambda_rank = lambda_rank
        self.lambda_attr = lambda_attr
        
        # Binary cross entropy for attribute classification
        self.bce_loss = nn.BCELoss()
        
    def attribute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attribute classification loss.
        
        Args:
            predictions: Predicted attribute probabilities (batch_size, num_attributes)
            targets: Ground truth attribute labels (batch_size, num_attributes)
            
        Returns:
            Binary cross entropy loss
        """
        return self.bce_loss(predictions, targets)
    
    def ranking_loss(
        self,
        anchor_embed: torch.Tensor,
        positive_embed: torch.Tensor,
        negative_embed: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet ranking loss.
        
        Args:
            anchor_embed: Anchor image embeddings
            positive_embed: Positive image embeddings
            negative_embed: Negative image embeddings
            
        Returns:
            Triplet ranking loss
        """
        # Compute distances
        pos_dist = F.pairwise_distance(anchor_embed, positive_embed)
        neg_dist = F.pairwise_distance(anchor_embed, negative_embed)
        
        # Compute triplet loss with margin
        loss = F.relu(pos_dist - neg_dist + self.margin)
        
        return loss.mean()
    
    def forward(
        self,
        model_output: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            model_output: Dict containing model outputs:
                - predictions: Attribute predictions
                - global_features: Global branch features
                - local_features: Local branch features
            targets: Dict containing ground truth:
                - attributes: Target attribute labels
                - pos_attributes: Positive sample attributes
                - neg_attributes: Negative sample attributes
                
        Returns:
            Tuple of:
            - Total loss
            - Dict of individual losses
        """
        # Compute attribute classification loss
        attr_loss = self.attribute_loss(
            model_output["predictions"],
            targets["attributes"]
        )
        
        # Compute ranking loss if triplet samples are provided
        rank_loss = torch.tensor(0.0, device=attr_loss.device)
        if all(k in targets for k in ["pos_attributes", "neg_attributes"]):
            # Get embeddings
            anchor_embed = model_output["global_features"]
            pos_embed = targets["pos_attributes"]
            neg_embed = targets["neg_attributes"]
            
            rank_loss = self.ranking_loss(
                anchor_embed,
                pos_embed,
                neg_embed
            )
        
        # Combine losses
        total_loss = (
            self.lambda_attr * attr_loss +
            self.lambda_rank * rank_loss
        )
        
        return total_loss, {
            "total_loss": total_loss.item(),
            "attr_loss": attr_loss.item(),
            "rank_loss": rank_loss.item()
        } 
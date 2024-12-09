"""style classifier model implementation"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from pathlib import Path

logger = logging.getLogger(__name__)

class AttentionModule(nn.Module):
    """spatial and channel attention module"""
    def __init__(self, in_channels):
        super().__init__()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//8, 1, 1),
            nn.Sigmoid()
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//8, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        spatial_weights = self.spatial_attention(x)
        channel_weights = self.channel_attention(x)
        return x * spatial_weights * channel_weights

class CrossAttention(nn.Module):
    """cross attention between attribute groups"""
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.scale = dim ** -0.5
        
    def forward(self, x1, x2):
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

class StyleClassifier(nn.Module):
    """hierarchical classifier for fashion attributes"""
    
    # Fixed attribute groups matching dataset structure
    ATTRIBUTE_GROUPS = {
        'texture': list(range(0, 6)),    # 6 texture attributes
        'fabric': list(range(6, 11)),    # 5 fabric attributes
        'shape': list(range(11, 17)),    # 6 shape attributes
        'part': list(range(17, 23)),     # 6 part attributes
        'style': list(range(23, 26))     # 3 style attributes
    }
    
    def __init__(
        self,
        num_attributes: int = 26,
        hidden_size: int = 256,  # increased hidden size
        num_layers: int = 3,     # increased layers
        dropout: float = 0.5,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        
        # Verify attribute count matches groups
        total_attrs = sum(len(indices) for indices in self.ATTRIBUTE_GROUPS.values())
        if total_attrs != num_attributes:
            raise ValueError(
                f"Total attributes in groups ({total_attrs}) does not match "
                f"expected number of attributes ({num_attributes})"
            )
            
        # ResNet50 backbone with pretrained weights
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # remove avg pool and fc
        
        # Freeze early layers
        for param in self.backbone[:6].parameters():
            param.requires_grad = False
            
        # Add attention modules
        self.attention1 = AttentionModule(2048)  # ResNet final channels
        
        # Feature projection with larger capacity
        self.projection = nn.Sequential(
            nn.Conv2d(2048, 512, 1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_attributes),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Create encoders for each attribute group
        self.group_encoders = nn.ModuleDict()
        for group_name, indices in self.ATTRIBUTE_GROUPS.items():
            input_size = len(indices)
            layers = []
            current_size = input_size
            
            for i in range(num_layers):
                out_size = hidden_size if i < num_layers-1 else hidden_size*2
                layers.extend([
                    nn.Linear(current_size, out_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.BatchNorm1d(out_size)
                ])
                current_size = out_size
                
            self.group_encoders[group_name] = nn.Sequential(*layers)
        
        # Enhanced fusion layer
        total_hidden = hidden_size * 2 * len(self.ATTRIBUTE_GROUPS)
        self.fusion = nn.Sequential(
            nn.Linear(total_hidden, total_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(total_hidden // 2),
            nn.Linear(total_hidden // 2, total_hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(total_hidden // 4)
        )
        
        # Prediction heads with increased capacity
        self.group_predictors = nn.ModuleDict()
        for group_name, indices in self.ATTRIBUTE_GROUPS.items():
            self.group_predictors[group_name] = nn.Sequential(
                nn.Linear(total_hidden // 4, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, len(indices))
            )
            
        # Cross-attention between groups
        self.cross_attention = nn.ModuleDict()
        for g1 in self.ATTRIBUTE_GROUPS:
            for g2 in self.ATTRIBUTE_GROUPS:
                if g1 != g2:
                    self.cross_attention[f"{g1}_{g2}"] = CrossAttention(hidden_size*2)
        
        # Style-specific contrastive head
        self.style_projector = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 128)  # contrastive embedding dimension
        )
    
    def _split_attributes(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """split input tensor into attribute groups"""
        return {
            name: x[:, indices]
            for name, indices in self.ATTRIBUTE_GROUPS.items()
        }
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """forward pass with attention and enhanced feature extraction"""
        # Extract features with attention
        features = self.backbone(x)
        attended = self.attention1(features)
        
        # Project to attribute space
        attributes = self.projection(attended)
        
        # Split and process attribute groups
        group_inputs = self._split_attributes(attributes)
        group_embeddings = {
            name: self.group_encoders[name](group_x)
            for name, group_x in group_inputs.items()
        }
        
        # Apply cross-attention between groups
        enhanced_embeddings = {}
        for g1, emb1 in group_embeddings.items():
            cross_attended = []
            for g2, emb2 in group_embeddings.items():
                if g1 != g2:
                    attended = self.cross_attention[f"{g1}_{g2}"](emb1, emb2)
                    cross_attended.append(attended)
            # Combine cross-attended features
            if cross_attended:
                enhanced_embeddings[g1] = emb1 + torch.stack(cross_attended).mean(0)
            else:
                enhanced_embeddings[g1] = emb1
        
        # Enhanced fusion with cross-attended features
        combined = torch.cat(list(enhanced_embeddings.values()), dim=1)
        fused = self.fusion(combined)
        
        # Group-specific predictions
        group_predictions = {
            name: self.group_predictors[name](fused)
            for name in self.ATTRIBUTE_GROUPS.keys()
        }
        
        # Generate contrastive embeddings for style attributes
        style_embedding = self.style_projector(enhanced_embeddings['style'])
        
        # Combine predictions
        all_predictions = []
        for name, indices in self.ATTRIBUTE_GROUPS.items():
            all_predictions.append(group_predictions[name])
        predictions = torch.cat(all_predictions, dim=1)
        
        return {
            'predictions': predictions,
            'style_embedding': style_embedding,
            'group_predictions': group_predictions
        }
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        group_weights: Dict[str, float] = None,
        style_pairs: Tuple[torch.Tensor, torch.Tensor] = None,
        temperature: float = 0.07
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """compute loss with contrastive learning for style attributes"""
        # Default equal weights if not provided
        if group_weights is None:
            group_weights = {name: 1.0 for name in self.ATTRIBUTE_GROUPS}
            # Increase weight for style attributes
            group_weights['style'] = 2.0
        
        # Split targets by group
        target_groups = self._split_attributes(targets)
        
        # Compute loss for each group
        group_losses = {}
        total_loss = 0
        
        for group_name in self.ATTRIBUTE_GROUPS:
            group_pred = predictions['group_predictions'][group_name]
            group_target = target_groups[group_name]
            
            # Binary cross entropy loss
            group_loss = F.binary_cross_entropy_with_logits(
                group_pred,
                group_target,
                reduction='mean'
            )
            
            # Store raw and weighted losses
            group_losses[f"{group_name}_loss"] = group_loss.item()
            weighted_loss = group_loss * group_weights[group_name]
            group_losses[f"{group_name}_weighted"] = weighted_loss.item()
            
            # Add to total loss
            total_loss += weighted_loss
        
        # Add contrastive loss for style attributes if pairs provided
        if style_pairs is not None:
            anchor_emb = predictions['style_embedding']
            positive_emb = style_pairs[0]
            negative_emb = style_pairs[1]
            
            # Normalize embeddings
            anchor_emb = F.normalize(anchor_emb, dim=1)
            positive_emb = F.normalize(positive_emb, dim=1)
            negative_emb = F.normalize(negative_emb, dim=1)
            
            # Compute similarities
            pos_sim = torch.sum(anchor_emb * positive_emb, dim=1)
            neg_sim = torch.sum(anchor_emb * negative_emb, dim=1)
            
            # Contrastive loss (InfoNCE)
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim.unsqueeze(1)], dim=1) / temperature
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
            contrastive_loss = F.cross_entropy(logits, labels)
            
            # Add to total loss
            total_loss += contrastive_loss * group_weights['style']
            group_losses['contrastive_loss'] = contrastive_loss.item()
        
        # Store total loss
        group_losses['total_loss'] = total_loss.item()
        
        return total_loss, group_losses
"""consolidated style classifier model implementation combining best features from v1 and v2

this version includes:
- hierarchical attribute classification
- multi-head attention
- feature pyramid network
- residual connections
- contrastive learning for style attributes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import logging
from torchvision.models import resnet50, ResNet50_Weights
from pathlib import Path

logger = logging.getLogger(__name__)

class AttentionModule(nn.Module):
    """multi-head attention module with spatial and channel attention"""
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mha = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        
        # spatial attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(dim, dim//8, 1),
            nn.ReLU(),
            nn.Conv2d(dim//8, 1, 1),
            nn.Sigmoid()
        )
        
        # channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim//8, 1),
            nn.ReLU(),
            nn.Conv2d(dim//8, dim, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # apply spatial and channel attention
        spatial_weights = self.spatial_attention(x)
        channel_weights = self.channel_attention(x)
        attended = x * spatial_weights * channel_weights
        
        # reshape for multi-head attention
        b, c, h, w = attended.shape
        attended = attended.flatten(2).permute(2, 0, 1)  # (h*w, b, c)
        
        # apply multi-head attention
        attended = self.norm(attended)
        mha_out, _ = self.mha(attended, attended, attended)
        attended = self.dropout(mha_out)
        
        # reshape back
        attended = attended.permute(1, 2, 0).view(b, c, h, w)
        return attended

class ResidualBlock(nn.Module):
    """residual block with pre-norm"""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.ff(x)
        x = residual + x
        return self.norm2(x)

class StyleClassifier(nn.Module):
    """hierarchical style classifier with attention and contrastive learning"""
    # fixed attribute groups matching dataset structure
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
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_heads: int = 8,
        device: str = 'cuda'
    ):
        """initialize model
        
        args:
            num_attributes: total number of attributes to predict
            hidden_dim: hidden dimension size
            num_layers: number of transformer layers
            dropout: dropout rate
            num_heads: number of attention heads
            device: device to use
        """
        super().__init__()
        self.device = device
        
        # verify attribute count matches groups
        total_attrs = sum(len(indices) for indices in self.ATTRIBUTE_GROUPS.values())
        if total_attrs != num_attributes:
            raise ValueError(
                f"Total attributes in groups ({total_attrs}) does not match "
                f"expected number of attributes ({num_attributes})"
            )
        
        # ResNet50 backbone
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        backbone_dim = 2048
        
        # freeze early layers
        for param in self.backbone[:6].parameters():
            param.requires_grad = False
        
        # Multi-scale attention
        self.attention = AttentionModule(
            backbone_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # feature pyramid
        self.fpn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(backbone_dim, hidden_dim, 1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ) for _ in range(4)
        ])
        
        # feature projection
        self.projection = nn.Sequential(
            nn.Conv2d(backbone_dim, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # group encoders with residual connections
        self.group_encoders = nn.ModuleDict()
        for group_name, indices in self.ATTRIBUTE_GROUPS.items():
            layers = []
            input_size = len(indices)
            
            # initial projection to hidden_dim
            layers.append(nn.Linear(input_size, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            # residual blocks
            for _ in range(num_layers):
                layers.append(ResidualBlock(hidden_dim, hidden_dim * 4, dropout))
            
            self.group_encoders[group_name] = nn.Sequential(*layers)
        
        # Ggoup attention for cross-group feature fusion
        self.group_attention = nn.ModuleDict()
        for g1 in self.ATTRIBUTE_GROUPS:
            self.group_attention[g1] = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout
            )
        
        # final prediction heads
        self.group_predictors = nn.ModuleDict()
        for group_name, indices in self.ATTRIBUTE_GROUPS.items():
            self.group_predictors[group_name] = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, len(indices))
            )
        
        # style-specific contrastive head
        self.style_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128)  # contrastive embedding dimension
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """initialize network weights"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _split_attributes(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """split attributes into groups"""
        return {
            name: x[:, indices]
            for name, indices in self.ATTRIBUTE_GROUPS.items()
        }
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """forward pass
        
        args:
            x: input images (batch_size, channels, height, width)
            return_features: whether to return intermediate features
            
        returns:
            dict containing:
                - predictions: final attribute predictions
                - style_embedding: style embedding for contrastive learning
                - group_predictions: per-group predictions
                - features: intermediate features (if return_features=True)
        """
        # Extract backbone features
        features = self.backbone(x)
        
        # Multi-scale attention
        attended = self.attention(features)
        
        # Feature pyramid
        pyramid_features = [layer(attended) for layer in self.fpn]
        pyramid_features = [f.flatten(1) for f in pyramid_features]
        
        # Project to attribute space
        projected = self.projection(attended)
        
        # Process each attribute group
        group_features = {}
        for group_name, indices in self.ATTRIBUTE_GROUPS.items():
            # Eetract group features
            group_input = projected[:, :len(indices)]
            group_features[group_name] = self.group_encoders[group_name](group_input)
        
        # cross-group attention
        enhanced_features = {}
        for g1, feat1 in group_features.items():
            # gather features from other groups
            other_feats = [f for g2, f in group_features.items() if g2 != g1]
            if other_feats:
                # Apply attention
                other_feats = torch.stack(other_feats, dim=0)
                attended, _ = self.group_attention[g1](
                    feat1.unsqueeze(0),
                    other_feats,
                    other_feats
                )
                enhanced_features[g1] = attended.squeeze(0) + feat1  # residual
            else:
                enhanced_features[g1] = feat1
        
        # Ggnerate predictions for each group
        group_predictions = {
            name: self.group_predictors[name](feat)
            for name, feat in enhanced_features.items()
        }
        
        # generate style embedding
        style_embedding = self.style_projector(enhanced_features['style'])
        
        # Combine all predictions
        all_predictions = []
        for name, indices in self.ATTRIBUTE_GROUPS.items():
            all_predictions.append(group_predictions[name])
        predictions = torch.cat(all_predictions, dim=1)
        
        outputs = {
            'predictions': predictions,
            'style_embedding': style_embedding,
            'group_predictions': group_predictions
        }
        
        if return_features:
            outputs['features'] = {
                'backbone': features,
                'attended': attended,
                'pyramid': pyramid_features,
                'projected': projected,
                'group': enhanced_features
            }
        
        return outputs
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        group_weights: Optional[Dict[str, float]] = None,
        style_pairs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        temperature: float = 0.07
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """compute hierarchical loss with contrastive learning
        
        args:
            predictions: model predictions
            targets: target labels
            group_weights: optional weights for each attribute group
            style_pairs: optional (positive, negative) pairs for contrastive loss
            temperature: temperature for contrastive loss
            
        returns:
            tuple containing:
                - total loss
                - dictionary of individual loss components
        """
        # default group weights
        if group_weights is None:
            group_weights = {name: 1.0 for name in self.ATTRIBUTE_GROUPS}
            group_weights['style'] = 2.0  # emphasize style attributes
        
        # split targets by group
        target_groups = self._split_attributes(targets)
        
        # Ccmpute loss for each group
        losses = {}
        total_loss = 0
        
        for group_name in self.ATTRIBUTE_GROUPS:
            group_pred = predictions['group_predictions'][group_name]
            group_target = target_groups[group_name]
            
            # binary cross entropy loss
            group_loss = F.binary_cross_entropy_with_logits(
                group_pred,
                group_target,
                reduction='mean'
            )
            
            # Store losses
            losses[f"{group_name}_loss"] = group_loss.item()
            weighted_loss = group_loss * group_weights[group_name]
            losses[f"{group_name}_weighted"] = weighted_loss.item()
            
            total_loss += weighted_loss
        
        # contrastive loss for style attributes
        if style_pairs is not None:
            anchor_emb = predictions['style_embedding']
            positive_emb, negative_emb = style_pairs
            
            # nrmalize embeddings
            anchor_emb = F.normalize(anchor_emb, dim=1)
            positive_emb = F.normalize(positive_emb, dim=1)
            negative_emb = F.normalize(negative_emb, dim=1)
            
            # compute similarities
            pos_sim = torch.sum(anchor_emb * positive_emb, dim=1)
            neg_sim = torch.sum(anchor_emb * negative_emb, dim=1)
            
            # InfoNCE loss
            logits = torch.cat([
                pos_sim.unsqueeze(1),
                neg_sim.unsqueeze(1)
            ], dim=1) / temperature
            
            labels = torch.zeros(
                logits.size(0),
                dtype=torch.long,
                device=logits.device
            )
            
            contrastive_loss = F.cross_entropy(logits, labels)
            
            # add to total loss
            total_loss += contrastive_loss * group_weights['style']
            losses['contrastive_loss'] = contrastive_loss.item()
        
        losses['total_loss'] = total_loss.item()
        return total_loss, losses 
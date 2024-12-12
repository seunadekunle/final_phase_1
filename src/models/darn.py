"""
Implementation of the Dual Attribute-aware Ranking Network (DARN).

This module contains the PyTorch implementation of the DARN architecture
for fine-grained attribute prediction on fashion images.

Reference:
    "DARN: A Deep Attentive Recurrent Network for Learning Attribute-Specific
    Representations of Product Images"
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from typing import Tuple, Dict

class AttributeAttention(nn.Module):
    """Attribute-aware attention module."""
    
    def __init__(self, in_features: int, num_attributes: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(),
            nn.Linear(in_features, num_attributes),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply attribute-specific attention.
        
        Args:
            x: Input features (batch_size, in_features)
            
        Returns:
            Tuple of:
            - Attended features (batch_size, in_features)
            - Attention weights (batch_size, num_attributes)
        """
        attention_weights = self.attention(x)
        attended_features = x.unsqueeze(1) * attention_weights.unsqueeze(-1)
        attended_features = attended_features.mean(dim=1)
        return attended_features, attention_weights

class DARN(nn.Module):
    """
    Dual Attribute-aware Ranking Network for fine-grained attribute prediction.
    
    Features:
    - VGG16 backbone with feature pyramid
    - Dual attribute-aware attention branches
    - Attribute prediction with ranking capability
    """
    
    def __init__(
        self,
        num_attributes: int,
        embedding_dim: int = 512,
        backbone: str = "vgg16",
        pretrained: bool = True
    ):
        """
        Initialize DARN model.
        
        Args:
            num_attributes: Number of attributes to predict
            embedding_dim: Dimension of the attribute embeddings
            backbone: Name of the backbone CNN architecture
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # Load backbone and get intermediate features
        self.backbone = self._get_backbone(backbone, pretrained)
        backbone_dim = self._get_backbone_dim(backbone)
        
        # Add adaptive pooling for ResNet
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Global branch
        self.global_branch = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.global_attention = AttributeAttention(embedding_dim, num_attributes)
        
        # Local branch (attribute-specific)
        self.local_branch = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.local_attention = AttributeAttention(embedding_dim, num_attributes)
        
        # Attribute prediction heads
        self.attribute_classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim, num_attributes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _get_backbone(self, backbone_name: str, pretrained: bool) -> nn.Module:
        """Get pretrained backbone CNN."""
        if backbone_name == "vgg16":
            model = models.vgg16(pretrained=pretrained)
            return nn.Sequential(*list(model.features.children()))
        elif backbone_name == "resnet34":
            model = models.resnet34(pretrained=pretrained)
            return nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu,
                model.maxpool,
                model.layer1,
                model.layer2,
                model.layer3,
                model.layer4
            )
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    def _get_backbone_dim(self, backbone_name: str) -> int:
        """Get backbone output dimension."""
        if backbone_name == "vgg16":
            return 512 * 7 * 7  # VGG16 output size with 224x224 input
        elif backbone_name == "resnet34":
            return 512  # ResNet34 output channels
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Dict containing:
            - predictions: Attribute predictions
            - global_features: Global branch features
            - local_features: Local branch features
            - global_attention: Global attention weights
            - local_attention: Local attention weights
        """
        # Extract backbone features
        features = self.backbone(x)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        
        # Global branch
        global_features = self.global_branch(features)
        global_attended, global_attention = self.global_attention(global_features)
        
        # Local branch
        local_features = self.local_branch(features)
        local_attended, local_attention = self.local_attention(local_features)
        
        # Combine features
        combined_features = torch.cat([global_attended, local_attended], dim=1)
        
        # Predict attributes
        predictions = torch.sigmoid(self.attribute_classifier(combined_features))
        
        return {
            "predictions": predictions,
            "global_features": global_features,
            "local_features": local_features,
            "global_attention": global_attention,
            "local_attention": local_attention
        }
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attribute-aware embeddings for retrieval.
        
        Args:
            x: Input tensor
            
        Returns:
            Combined embedding from both branches
        """
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
            
        global_features = self.global_branch(features)
        local_features = self.local_branch(features)
        
        return torch.cat([global_features, local_features], dim=1)
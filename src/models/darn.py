"""
Implementation of the Dual Attribute-aware Ranking Network (DARN).

This Module Contains the PyTorch Implementation of the DARN Architecture
For Fine-grained Attribute Prediction on Fashion Images.

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
    """Attribute-aware Attention Module for Feature Weighting."""
    
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
        Apply Attribute-specific Attention to Input Features.
        
        Args:
            x: Input features (batch_size, in_features)
            
        Returns:
            Tuple of:
            - Attended features (batch_size, in_features)
            - Attention weights (batch_size, num_attributes)
        """
        # compute attention weights
        attention_weights = self.attention(x)
        # apply attention to features
        attended_features = x.unsqueeze(1) * attention_weights.unsqueeze(-1)
        attended_features = attended_features.mean(dim=1)
        return attended_features, attention_weights

class DARN(nn.Module):
    """
    Dual Attribute-aware Ranking Network for Fine-grained Attribute Prediction.
    
    Features:
        - VGG16 Backbone with Feature Pyramid
        - Dual Attribute-aware Attention Branches
        - Attribute Prediction with Ranking Capability
    """
    
    def __init__(
        self,
        num_attributes: int,
        embedding_dim: int = 512,
        backbone: str = "vgg16",
        pretrained: bool = True
    ):
        """
        Initialize DARN Model Components.
        
        Args:
            num_attributes: Number of attributes to predict
            embedding_dim: Dimension of the attribute embeddings
            backbone: Name of the backbone CNN architecture
            pretrained: Whether to use pretrained weights
        """
        super().__init__()
        
        # load backbone and get intermediate features
        self.backbone = self._get_backbone(backbone, pretrained)
        backbone_dim = self._get_backbone_dim(backbone)
        
        # add adaptive pooling for resnet
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # global branch
        self.global_branch = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.global_attention = AttributeAttention(embedding_dim, num_attributes)
        
        # local branch (attribute-specific)
        self.local_branch = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.local_attention = AttributeAttention(embedding_dim, num_attributes)
        
        # attribute prediction heads
        self.attribute_classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(embedding_dim, num_attributes)
        )
        
        # initialize weights
        self._init_weights()
        
    def _get_backbone(self, backbone_name: str, pretrained: bool) -> nn.Module:
        """
        Get Pretrained Backbone CNN Architecture.
        
        Args:
            backbone_name: Name of the backbone to use
            pretrained: Whether to use pretrained weights
            
        Returns:
            Backbone CNN module
        """
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
        """
        Get Output Dimension of Backbone CNN.
        
        Args:
            backbone_name: Name of the backbone
            
        Returns:
            Output dimension of the backbone
        """
        if backbone_name == "vgg16":
            return 512 * 7 * 7  # vgg16 output size with 224x224 input
        elif backbone_name == "resnet34":
            return 512  # resnet34 output channels
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    def _init_weights(self):
        """Initialize Model Weights Using Kaiming Initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward Pass Through the Network.
        
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
        # extract backbone features
        features = self.backbone(x)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        
        # global branch processing
        global_features = self.global_branch(features)
        global_attended, global_attention = self.global_attention(global_features)
        
        # local branch processing
        local_features = self.local_branch(features)
        local_attended, local_attention = self.local_attention(local_features)
        
        # combine features
        combined_features = torch.cat([global_attended, local_attended], dim=1)
        
        # predict attributes
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
        Get Attribute-aware Embeddings for Retrieval.
        
        Args:
            x: Input tensor
            
        Returns:
            Combined embedding from both branches
        """
        # extract features
        features = self.backbone(x)
        if len(features.shape) > 2:
            features = features.view(features.size(0), -1)
            
        # get branch features
        global_features = self.global_branch(features)
        local_features = self.local_branch(features)
        
        return torch.cat([global_features, local_features], dim=1)
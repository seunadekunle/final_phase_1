"""
CLIP Model Integration for Visual Feature Extraction.

This Module Provides a Wrapper Around OpenAI's CLIP Model
For Extracting Visual Features from Fashion Images.
"""

from typing import Tuple
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class CLIPEncoder(nn.Module):
    """
    CLIP Model Wrapper for Visual Feature Extraction.
    
    Features:
        - Frozen CLIP Parameters
        - Efficient Feature Extraction
        - Consistent Feature Representations
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP Encoder Components.
        
        Args:
            model_name: Name of the CLIP model to use
        """
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # freeze clip parameters for feature extraction
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract Visual Features Using CLIP Model.
        
        Args:
            images: Batch of images (B, C, H, W)
            
        Returns:
            Visual features (B, hidden_dim)
        """
        # extract features without gradient computation
        with torch.no_grad():
            features = self.model.get_image_features(images)
            
        return features
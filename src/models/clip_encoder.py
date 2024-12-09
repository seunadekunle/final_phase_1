from typing import Tuple
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class CLIPEncoder(nn.Module):
    """Wrapper for CLIP model to extract visual features."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP encoder.
        
        Args:
            model_name: Name of the CLIP model to use
        """
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Freeze CLIP parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from images using CLIP.
        
        Args:
            images: Batch of images (B, C, H, W)
            
        Returns:
            Visual features (B, hidden_dim)
        """
        # CLIP expects images in (B, C, H, W) format
        with torch.no_grad():
            features = self.model.get_image_features(images)
            
        return features 
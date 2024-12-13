from typing import Tuple
import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class CLIPEncoder(nn.Module):
    """Wrapper for CLIP Model to Extract Visual Features."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP Encoder.
        
        Args:
            model_name: Name of the CLIP model to use
        """
        super().__init__()
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # freeze clip parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract Visual Features from Images Using CLIP.
        
        Args:
            images: Batch of images (B, C, H, W)
            
        Returns:
            Visual features (B, hidden_dim)
        """
        # clip expects images in (b, c, h, w) format
        with torch.no_grad():
            features = self.model.get_image_features(images)
            
        return features
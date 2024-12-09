"""enhanced style classifier with attention and pyramid features for fashion classification"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

class AttentionBlock(nn.Module):
    """self attention block for feature refinement"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """apply self attention to input features"""
        x = self.norm(x)
        # reshape for attention
        x = x.unsqueeze(0)  # add sequence length dimension
        attn_out, _ = self.attn(x, x, x)
        x = self.dropout(attn_out)
        return x.squeeze(0)  # remove sequence dimension

class ResidualBlock(nn.Module):
    """residual block with pre-norm and skip connection"""
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # more stable than ReLU
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """forward pass with residual connection"""
        # pre-norm residual path
        residual = x
        x = self.norm1(x)
        x = self.ff(x)
        x = residual + x
        return self.norm2(x)

class PyramidAttention(nn.Module):
    """attention module for feature pyramid network"""
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, num_heads)
        
    def forward(self, features_list):
        # normalize and combine features
        features = torch.stack([self.norm(f) for f in features_list], dim=0)
        attended_features, _ = self.attention(features, features, features)
        return torch.mean(attended_features, dim=0)

class CategoryClassifier(nn.Module):
    """enhanced category classifier with residual connections"""
    def __init__(self, in_dim, hidden_dim, num_categories, dropout_rate=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim * 2)
        self.bn1 = nn.BatchNorm1d(hidden_dim * 2)
        
        self.norm2 = nn.LayerNorm(hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_categories)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # first block with residual
        identity = x
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        if x.size(-1) == identity.size(-1):
            x = x + identity
            
        # second block with residual
        identity = x
        x = self.norm2(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.gelu(x)
        x = self.dropout(x)
        
        # final classification
        x = self.norm3(x)
        x = self.fc3(x)
        return x

class StyleClassifierV2(nn.Module):
    """improved style classifier with advanced architecture"""
    def __init__(
        self,
        num_categories: int,
        num_attributes: int,
        hidden_dim: int = 512,
        dropout_rate: float = 0.1,
        num_attention_heads: int = 8
    ):
        super().__init__()
        
        # load and freeze clip backbone
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        for param in self.clip.parameters():
            param.requires_grad = False
            
        self.vision_dim = self.clip.vision_model.config.hidden_size
        
        # feature refinement with attention
        self.attention = AttentionBlock(
            self.vision_dim,
            num_heads=num_attention_heads,
            dropout=dropout_rate
        )
        
        # enhanced feature pyramid with more levels
        self.fpn = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.vision_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            ) for _ in range(4)  # increased to 4 levels
        ])
        
        # pyramid attention for feature fusion
        self.pyramid_attention = PyramidAttention(hidden_dim, num_heads=4)
        
        # shared feature processing
        self.shared_layers = nn.ModuleList([
            ResidualBlock(hidden_dim, hidden_dim * 2, dropout_rate)
            for _ in range(3)
        ])
        
        # feature dimension reduction
        self.feature_reduction = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # enhanced category classifier
        self.category_classifier = CategoryClassifier(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_categories=num_categories,
            dropout_rate=dropout_rate
        )
        
        # attribute classifier remains the same
        self.attribute_classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_attributes)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """initialize network weights using kaiming normal"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, 
                    mode='fan_out', 
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def mixup(self, x, y, alpha=0.2):
        """apply mixup augmentation to input and labels"""
        if not self.training:
            return x, y
            
        batch_size = x.size(0)
        perm = torch.randperm(batch_size).to(x.device)
        
        # generate mixup weights from beta distribution
        lambda_param = torch.distributions.Beta(alpha, alpha).sample().to(x.device)
        
        # mix the samples and labels
        mixed_x = lambda_param * x + (1 - lambda_param) * x[perm]
        mixed_y = lambda_param * y + (1 - lambda_param) * y[perm]
        
        return mixed_x, mixed_y
        
    def forward(self, images, category_labels=None, attribute_labels=None):
        """forward pass with enhanced feature processing"""
        # preprocess images to match CLIP's expected input size
        if images.shape[-2:] != (224, 224):
            images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            
        # extract clip vision features
        vision_outputs = self.clip.vision_model(
            images,
            output_hidden_states=True,
            return_dict=True
        )
        
        # get features from different scales (last 4 layers)
        features = vision_outputs.pooler_output
        hidden_states = vision_outputs.hidden_states[-4:]
        
        # apply attention to base features
        features = self.attention(features)
        
        # process features at different scales
        fpn_features = []
        for fpn_layer, hidden_state in zip(self.fpn, hidden_states):
            scale_features = hidden_state.mean(dim=1)
            fpn_features.append(fpn_layer(scale_features))
        
        # apply pyramid attention
        fused_pyramid_features = self.pyramid_attention(fpn_features)
        
        # combine all features
        combined_features = torch.cat([features, fused_pyramid_features] + fpn_features, dim=1)
        
        # reduce feature dimension
        processed_features = self.feature_reduction(combined_features)
        
        # shared feature processing
        for layer in self.shared_layers:
            processed_features = layer(processed_features)
            
        # classification
        category_logits = self.category_classifier(processed_features)
        attribute_logits = self.attribute_classifier(processed_features)
        
        # apply mixup if training
        if self.training and category_labels is not None:
            category_logits, category_labels = self.mixup(category_logits, category_labels)
            attribute_logits, attribute_labels = self.mixup(attribute_logits, attribute_labels)
        
        return category_logits, attribute_logits
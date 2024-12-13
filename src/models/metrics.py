"""
Evaluation Metrics for Fashion Attribute Prediction.

This Module Implements Various Evaluation Metrics for
Assessing Model Performance on Fashion Attribute Tasks.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple

def compute_top_k_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5
) -> float:
    """
    Compute Top-K Accuracy for Multi-class Predictions.
    
    Args:
        predictions: Predicted logits (batch_size, num_classes)
        targets: Target labels (batch_size,)
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy score
    """
    # get top k predictions
    batch_size = targets.size(0)
    _, pred = predictions.topk(k, 1, True, True)
    pred = pred.t()
    
    # compute correct predictions
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    
    # calculate accuracy percentage
    return correct_k.mul_(100.0 / batch_size).item()

def compute_per_attribute_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> Tuple[float, List[float]]:
    """
    Compute Accuracy for Individual Attributes.
    
    Args:
        predictions: Predicted logits (batch_size, num_attributes)
        targets: Target labels (batch_size, num_attributes)
        
    Returns:
        Tuple containing:
        - Mean accuracy across attributes
        - List of per-attribute accuracies
    """
    # convert predictions to binary values
    binary_preds = (torch.sigmoid(predictions) > 0.5).float() * 2 - 1
    
    # compute accuracy for each attribute
    correct_per_attr = (binary_preds == targets).float().mean(dim=0)
    accuracies = correct_per_attr.tolist()
    mean_accuracy = correct_per_attr.mean().item()
    
    return mean_accuracy, accuracies

def compute_recall_at_k(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    k: int = 5
) -> float:
    """
    Compute Recall@K for Image Retrieval.
    
    Args:
        query_features: Query feature vectors (num_queries, feature_dim)
        gallery_features: Gallery feature vectors (num_gallery, feature_dim)
        query_labels: Query labels (num_queries, num_attributes)
        gallery_labels: Gallery labels (num_gallery, num_attributes)
        k: Number of top retrievals to consider
        
    Returns:
        Recall@K score
    """
    # compute similarity matrix
    similarity = F.normalize(query_features, dim=1) @ F.normalize(gallery_features, dim=1).t()
    
    # get top-k matches
    _, indices = similarity.topk(k, dim=1)
    
    # compare attribute matches
    query_labels = query_labels.unsqueeze(1).expand(-1, k, -1)
    gallery_labels = gallery_labels[indices]
    
    # compute recall score
    matches = (query_labels == gallery_labels).all(dim=2).any(dim=1)
    recall = matches.float().mean().item()
    
    return recall * 100.0

def compute_baseline_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    features: torch.Tensor = None,
    split: str = "val"
) -> Dict[str, float]:
    """
    Compute Standard Evaluation Metrics.
    
    Args:
        predictions: Dict containing model outputs
        targets: Target attributes
        features: Optional feature vectors for retrieval
        split: Dataset split (train/val/test)
        
    Returns:
        Dictionary of computed metrics
    """
    metrics = {}
    
    # compute attribute prediction metrics
    mean_acc, per_attr_acc = compute_per_attribute_accuracy(
        predictions["attribute_predictions"],
        targets
    )
    metrics["mean_attribute_accuracy"] = mean_acc
    metrics["per_attribute_accuracy"] = per_attr_acc
    
    # compute retrieval metrics if features provided
    if features is not None and split != "train":
        recall_5 = compute_recall_at_k(
            features, features, targets, targets, k=5
        )
        metrics["recall@5"] = recall_5
        
    return metrics

def compute_improved_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: torch.Tensor
) -> Dict[str, float]:
    """
    Compute Enhanced Evaluation Metrics.
    
    Args:
        predictions: Dict containing model outputs
        targets: Target attributes
        
    Returns:
        Dictionary of computed metrics
    """
    # get baseline metrics
    metrics = compute_baseline_metrics(predictions, targets)
    
    # add self-supervised metrics if available
    if "correlation_loss" in predictions:
        metrics["correlation_loss"] = predictions["correlation_loss"].item()
        metrics["missing_attr_loss"] = predictions["missing_attr_loss"].item()
        
    return metrics
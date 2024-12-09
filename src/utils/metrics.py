"""
Evaluation metrics for attribute prediction.

Implements standard metrics for evaluating attribute prediction performance:
- Accuracy
- Precision
- Recall
- F1 Score
"""

import torch
import numpy as np
from typing import Dict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_attribute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute attribute prediction metrics.
    
    Args:
        predictions: Predicted probabilities (N, num_attributes)
        targets: Ground truth labels (N, num_attributes)
        threshold: Classification threshold
        
    Returns:
        Dict containing metrics:
        - accuracy: Overall classification accuracy
        - precision: Mean precision across attributes
        - recall: Mean recall across attributes
        - f1: Mean F1 score across attributes
    """
    # Convert predictions to binary
    pred_labels = (predictions >= threshold).float()
    
    # Convert to numpy for sklearn metrics
    pred_np = pred_labels.cpu().numpy()
    target_np = targets.cpu().numpy()
    
    # Compute per-attribute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        target_np,
        pred_np,
        average='macro',
        zero_division=0
    )
    
    # Compute overall accuracy
    accuracy = accuracy_score(
        target_np.flatten(),
        pred_np.flatten()
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def compute_retrieval_metrics(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    query_attributes: torch.Tensor,
    gallery_attributes: torch.Tensor,
    k_values: list = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute retrieval metrics.
    
    Args:
        query_embeddings: Query image embeddings (N_q, embedding_dim)
        gallery_embeddings: Gallery image embeddings (N_g, embedding_dim)
        query_attributes: Query image attributes (N_q, num_attributes)
        gallery_attributes: Gallery image attributes (N_g, num_attributes)
        k_values: List of k values for recall@k computation
        
    Returns:
        Dict containing:
        - recall@k: Recall at different k values
        - mean_attribute_precision: Mean attribute precision of retrieved items
    """
    # Compute pairwise distances
    distances = torch.cdist(query_embeddings, gallery_embeddings)
    
    # Get top-k indices for each query
    _, topk_indices = distances.topk(k=max(k_values), dim=1, largest=False)
    
    metrics = {}
    
    # Compute recall@k
    for k in k_values:
        # Get top-k predictions for each query
        topk_attrs = gallery_attributes[topk_indices[:, :k]]
        
        # Compute attribute matches
        matches = (query_attributes.unsqueeze(1) == topk_attrs).float()
        
        # Average over attributes and queries
        recall_k = matches.mean().item()
        metrics[f"recall@{k}"] = recall_k
    
    # Compute mean attribute precision
    attribute_matches = (query_attributes.unsqueeze(1) == 
                        gallery_attributes[topk_indices[:, 0]]).float()
    mean_attr_precision = attribute_matches.mean().item()
    metrics["mean_attribute_precision"] = mean_attr_precision
    
    return metrics 
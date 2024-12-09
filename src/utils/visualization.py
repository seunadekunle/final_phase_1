"""visualization utilities for style classifier"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, save_path: Path):
    """plot confusion matrix for each attribute
    
    args:
        predictions: model predictions (N, num_attributes)
        targets: ground truth labels (N, num_attributes)
        save_path: path to save plots
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # create confusion matrix for each attribute
    for attr_idx in range(predictions.shape[1]):
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(targets[:, attr_idx], predictions[:, attr_idx] > 0.5)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Attribute {attr_idx}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path / f'confusion_matrix_attr_{attr_idx}.png')
        plt.close()

def plot_embedding_visualization(embeddings: torch.Tensor, targets: torch.Tensor, save_path: Path):
    """plot t-SNE visualization of embeddings
    
    args:
        embeddings: model embeddings (N, embedding_dim)
        targets: ground truth labels (N, num_attributes)
        save_path: path to save plot
    """
    # reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings.cpu().numpy())
    
    # plot for first attribute
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=targets[:, 0].cpu().numpy(), cmap='viridis')
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Style Embeddings')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(save_path / 'embedding_visualization.png')
    plt.close()

def plot_attribute_correlations(predictions: torch.Tensor, save_path: Path):
    """plot correlation matrix between attributes
    
    args:
        predictions: model predictions (N, num_attributes)
        save_path: path to save plot
    """
    # compute correlation matrix
    corr = np.corrcoef(predictions.cpu().numpy().T)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
    plt.title('Attribute Correlations')
    plt.savefig(save_path / 'attribute_correlations.png')
    plt.close() 
"""
Evaluation script for DARN model.

Handles model evaluation, metrics computation, and visualization.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Dict, Any

from src.models.darn import DARN
from src.data.deepfashion_dataset import DeepFashionDataset, create_dataloaders
from src.utils.metrics import compute_attribute_metrics, compute_retrieval_metrics

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate DARN model")
    
    parser.add_argument("--data_root", type=str, required=True,
                      help="Path to DeepFashion dataset root")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                      help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="Number of data loading workers")
    parser.add_argument("--output_dir", type=str, default="experiments/evaluation",
                      help="Directory to save evaluation results")
    
    return parser.parse_args()

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluate model on test set.
    
    Args:
        model: DARN model
        test_loader: Test data loader
        device: Device to evaluate on
        
    Returns:
        Dict containing evaluation metrics and predictions
    """
    model.eval()
    
    # Collect predictions and embeddings
    all_preds = []
    all_targets = []
    all_embeddings = []
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        # Move data to device
        images = batch["image"].to(device)
        attributes = batch["attributes"].to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Store predictions and embeddings
        all_preds.append(outputs["predictions"].cpu())
        all_targets.append(attributes.cpu())
        all_embeddings.append(outputs["embeddings"].cpu())
    
    # Concatenate all predictions and targets
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_embeddings = torch.cat(all_embeddings)
    
    # Compute attribute prediction metrics
    attr_metrics = compute_attribute_metrics(all_preds, all_targets)
    
    # Compute retrieval metrics
    retrieval_metrics = compute_retrieval_metrics(
        all_embeddings[:1000],  # Use subset for efficiency
        all_embeddings,
        all_targets[:1000],
        all_targets
    )
    
    # Combine metrics
    metrics = {
        **attr_metrics,
        **retrieval_metrics
    }
    
    return {
        'metrics': metrics,
        'predictions': all_preds,
        'targets': all_targets,
        'embeddings': all_embeddings
    }

def plot_confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, save_path: Path):
    """Plot confusion matrix for each attribute."""
    # Convert logits to binary predictions
    binary_preds = (torch.sigmoid(predictions) > 0.5).float()
    
    for attr_idx in tqdm(range(predictions.size(1)), desc='Plotting confusion matrices'):
        # Compute confusion matrix
        cm = confusion_matrix(
            targets[:, attr_idx].numpy(),
            binary_preds[:, attr_idx].numpy()
        )
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Attribute {attr_idx}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path / f'confusion_matrix_attr_{attr_idx}.png')
        plt.close()

def plot_embedding_visualization(embeddings: torch.Tensor, targets: torch.Tensor, save_path: Path):
    """Plot t-SNE visualization of embeddings."""
    # Reduce dimensionality
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings.numpy())
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=targets.sum(dim=1).numpy(),  # Color by number of positive attributes
        cmap='viridis'
    )
    plt.colorbar(scatter, label='Number of Positive Attributes')
    plt.title('t-SNE Visualization of Feature Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.savefig(save_path / 'embedding_visualization.png')
    plt.close()

def plot_attribute_correlations(predictions: torch.Tensor, save_path: Path):
    """Plot attribute correlation matrix."""
    # Compute correlations
    probs = torch.sigmoid(predictions)
    correlations = np.corrcoef(probs.t().numpy())
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlations, cmap='RdBu', center=0)
    plt.title('Attribute Correlations')
    plt.savefig(save_path / 'attribute_correlations.png')
    plt.close()

def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    checkpoint_args = checkpoint["args"]
    
    # Create model and load weights
    model = DARN(
        num_attributes=1000,  # DeepFashion attribute count
        embedding_dim=checkpoint_args["embedding_dim"],
        backbone=checkpoint_args["backbone"]
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Create test dataloader
    dataloaders = create_dataloaders(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    test_loader = dataloaders["test"]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Evaluate model
    results = evaluate_model(model, test_loader, device)
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results["metrics"], f, indent=2)
    
    # Generate visualizations
    plot_confusion_matrix(
        results["predictions"],
        results["targets"],
        output_dir
    )
    
    plot_embedding_visualization(
        results["embeddings"],
        results["targets"],
        output_dir
    )
    
    plot_attribute_correlations(
        results["predictions"],
        output_dir
    )
    
    # Log results
    logger.info("Evaluation Results:")
    for name, value in results["metrics"].items():
        logger.info(f"{name}: {value:.4f}")
    
    logger.info(f"Saved results to {output_dir}")

if __name__ == "__main__":
    main() 
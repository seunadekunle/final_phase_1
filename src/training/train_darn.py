"""
Training script for DARN model with visualization and advanced training features.

Handles model training, validation, logging, and visualization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
import argparse
from pathlib import Path
from tqdm import tqdm
import wandb
import sys
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.models.darn import DARN
from src.models.darn_loss import DARNLoss
from src.data.deepfashion_dataset import DeepFashionDataset, create_dataloaders
from src.utils.metrics import compute_attribute_metrics, compute_retrieval_metrics

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DARN model")
    
    # Data args
    parser.add_argument("--data_root", type=str, required=True,
                      help="Path to DeepFashion dataset root")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=4,
                      help="Number of data loading workers")
    
    # Model args
    parser.add_argument("--backbone", type=str, default="vgg16",
                      choices=["vgg16", "resnet34", "resnet50"],
                      help="Backbone CNN architecture")
    parser.add_argument("--embedding_dim", type=int, default=512,
                      help="Dimension of attribute embeddings")
    parser.add_argument("--dropout", type=float, default=0.5,
                      help="Dropout rate")
    
    # Training args
    parser.add_argument("--epochs", type=int, default=50,
                      help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                      help="Weight decay")
    parser.add_argument("--margin", type=float, default=0.3,
                      help="Margin for triplet loss")
    parser.add_argument("--attribute_weight", type=float, default=1.0,
                      help="Weight for attribute loss")
    parser.add_argument("--ranking_weight", type=float, default=1.0,
                      help="Weight for ranking loss")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                      help="Label smoothing factor")
    
    # Scheduler args
    parser.add_argument("--scheduler", type=str, default="cosine",
                      choices=["cosine", "step", "none"],
                      help="Learning rate scheduler")
    parser.add_argument("--warmup_epochs", type=int, default=5,
                      help="Number of warmup epochs")
    parser.add_argument("--min_lr", type=float, default=1e-6,
                      help="Minimum learning rate")
    
    # Logging args
    parser.add_argument("--log_dir", type=str, default="logs",
                      help="Directory for tensorboard logs")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                      help="Directory for model checkpoints")
    parser.add_argument("--experiment_name", type=str, default=None,
                      help="Name of experiment for logging")
    parser.add_argument("--wandb", action="store_true",
                      help="Enable wandb logging")
    parser.add_argument("--visualize_every", type=int, default=5,
                      help="Visualize predictions every N epochs")
    
    # Debug args
    parser.add_argument("--test_mode", action="store_true",
                      help="Run in test mode with smaller dataset")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to train on (cuda/cpu)")
    
    return parser.parse_args()

def get_scheduler(optimizer: optim.Optimizer, args: argparse.Namespace):
    """Get learning rate scheduler."""
    if args.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - args.warmup_epochs,
            eta_min=args.min_lr
        )
    elif args.scheduler == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.1
        )
    else:
        return None

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace
) -> Dict[str, float]:
    """
    Train for one epoch with advanced features.
    
    Args:
        model: DARN model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        epoch: Current epoch number
        args: Training arguments
        
    Returns:
        Dict of training metrics
    """
    model.train()
    total_loss = 0
    total_attr_loss = 0
    total_rank_loss = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        images = batch["image"].to(device)
        attributes = batch["attributes"].to(device)
        
        # Forward pass
        outputs = model(images)
        loss, loss_dict = criterion(outputs, {"attributes": attributes})
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        total_loss += loss_dict["total_loss"]
        total_attr_loss += loss_dict["attr_loss"]
        total_rank_loss += loss_dict["rank_loss"]
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss_dict['total_loss']:.4f}",
            "attr_loss": f"{loss_dict['attr_loss']:.4f}",
            "rank_loss": f"{loss_dict['rank_loss']:.4f}",
            "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
        })
        
        # Log batch metrics to wandb
        if args.wandb:
            wandb.log({
                "batch/loss": loss_dict["total_loss"],
                "batch/attr_loss": loss_dict["attr_loss"],
                "batch/rank_loss": loss_dict["rank_loss"],
                "batch/lr": optimizer.param_groups[0]["lr"]
            })
    
    # Step scheduler
    if scheduler is not None:
        scheduler.step()
    
    # Compute epoch metrics
    num_batches = len(train_loader)
    metrics = {
        "train/loss": total_loss / num_batches,
        "train/attr_loss": total_attr_loss / num_batches,
        "train/rank_loss": total_rank_loss / num_batches,
        "train/lr": optimizer.param_groups[0]["lr"]
    }
    
    return metrics

def visualize_predictions(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    epoch: int
):
    """Visualize model predictions and attention patterns."""
    model.eval()
    fig_dir = output_dir / f"epoch_{epoch}" / "visualizations"
    fig_dir.mkdir(exist_ok=True, parents=True)
    
    # Get a batch of data
    batch = next(iter(val_loader))
    images = batch["image"].to(device)
    attributes = batch["attributes"].to(device)
    
    with torch.no_grad():
        outputs = model(images)
        
        # Plot attention maps
        attention_maps = outputs.get("attention_maps")
        if attention_maps is not None:
            for i, attn in enumerate(attention_maps):
                plt.figure(figsize=(10, 10))
                sns.heatmap(attn.cpu().numpy(), cmap="viridis")
                plt.title(f"Attention Map {i+1}")
                plt.savefig(fig_dir / f"attention_map_{i+1}.png")
                plt.close()
        
        # Plot embedding similarity matrix
        embeddings = outputs["embeddings"]
        similarity = torch.matmul(embeddings, embeddings.t())
        
        plt.figure(figsize=(10, 10))
        sns.heatmap(similarity.cpu().numpy(), cmap="viridis")
        plt.title("Embedding Similarity Matrix")
        plt.savefig(fig_dir / "embedding_similarity.png")
        plt.close()
        
        # Plot attribute prediction distributions
        predictions = torch.sigmoid(outputs["predictions"])
        
        plt.figure(figsize=(15, 5))
        sns.boxplot(data=predictions.cpu().numpy())
        plt.title("Attribute Prediction Distributions")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(fig_dir / "prediction_distributions.png")
        plt.close()

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace
) -> Dict[str, float]:
    """
    Validate model with comprehensive metrics.
    
    Args:
        model: DARN model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        args: Training arguments
        
    Returns:
        Dict of validation metrics
    """
    model.eval()
    total_loss = 0
    total_attr_loss = 0
    total_rank_loss = 0
    
    all_preds = []
    all_targets = []
    all_embeddings = []
    
    for batch in tqdm(val_loader, desc="Validating"):
        # Move data to device
        images = batch["image"].to(device)
        attributes = batch["attributes"].to(device)
        
        # Forward pass
        outputs = model(images)
        loss, loss_dict = criterion(outputs, {"attributes": attributes})
        
        # Update metrics
        total_loss += loss_dict["total_loss"]
        total_attr_loss += loss_dict["attr_loss"]
        total_rank_loss += loss_dict["rank_loss"]
        
        # Store predictions for metric computation
        all_preds.append(outputs["predictions"].cpu())
        all_targets.append(attributes.cpu())
        all_embeddings.append(outputs["embeddings"].cpu())
    
    # Concatenate predictions and targets
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_embeddings = torch.cat(all_embeddings)
    
    # Compute metrics
    num_batches = len(val_loader)
    metrics = {
        "val/loss": total_loss / num_batches,
        "val/attr_loss": total_attr_loss / num_batches,
        "val/rank_loss": total_rank_loss / num_batches
    }
    
    # Compute attribute prediction metrics
    attr_metrics = compute_attribute_metrics(all_preds, all_targets)
    metrics.update({f"val/{k}": v for k, v in attr_metrics.items()})
    
    # Compute retrieval metrics
    retrieval_metrics = compute_retrieval_metrics(
        all_embeddings[:1000],  # Use subset for efficiency
        all_embeddings,
        all_targets[:1000],
        all_targets
    )
    metrics.update({f"val/{k}": v for k, v in retrieval_metrics.items()})
    
    # Visualize predictions periodically
    if epoch % args.visualize_every == 0:
        visualize_predictions(
            model,
            val_loader,
            device,
            Path(args.checkpoint_dir),
            epoch
        )
    
    return metrics

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    epoch: int,
    args: argparse.Namespace,
    metrics: Dict[str, float],
    is_best: bool,
    checkpoint_dir: Path
):
    """Save model checkpoint with metadata."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "args": vars(args),
        "metrics": metrics
    }
    
    # Save latest checkpoint
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)
        logger.info(f"Saved best model to {best_path}")
    
    logger.info(f"Saved checkpoint to {checkpoint_path}")

def main():
    """Main training function with comprehensive logging and visualization."""
    args = parse_args()
    
    # Set up experiment name
    if args.experiment_name is None:
        args.experiment_name = (
            f"darn_{args.backbone}_"
            f"e{args.embedding_dim}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(f"{args.log_dir}/{args.experiment_name}.log"),
            logging.StreamHandler()
        ]
    )
    
    # Create directories
    Path(args.log_dir).mkdir(exist_ok=True, parents=True)
    Path(args.checkpoint_dir).mkdir(exist_ok=True, parents=True)
    
    # Initialize wandb
    if args.wandb:
        wandb.init(
            project="style_classifier",
            name=args.experiment_name,
            config=vars(args)
        )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load datasets
    datasets = {
        'train': DeepFashionDataset(args.data_root, split='train'),
        'val': DeepFashionDataset(args.data_root, split='val'),
        'test': DeepFashionDataset(args.data_root, split='test')
    }
    
    # Create subsets if in test mode
    if args.test_mode:
        for split in datasets:
            indices = list(range(min(100, len(datasets[split]))))
            datasets[split] = torch.utils.data.Subset(datasets[split], indices)
    
    # Create dataloaders
    dataloaders = {
        split: DataLoader(
            datasets[split],
            batch_size=args.batch_size,
            shuffle=(split == 'train'),
            num_workers=args.num_workers,
            pin_memory=True
        )
        for split in ['train', 'val', 'test']
    }
    
    # Create model
    model = DARN(
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        num_attributes=25,
        pretrained=True
    ).to(device)
    
    # Create loss function
    criterion = DARNLoss(
        margin=args.margin,
        lambda_rank=args.ranking_weight,
        lambda_attr=args.attribute_weight
    )
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create scheduler
    scheduler = get_scheduler(optimizer, args)
    
    # Training loop
    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model,
            dataloaders["train"],
            criterion,
            optimizer,
            scheduler,
            device,
            epoch,
            args
        )
        
        # Validate
        val_metrics = validate(
            model,
            dataloaders["val"],
            criterion,
            device,
            epoch,
            args
        )
        
        # Log metrics
        metrics = {**train_metrics, **val_metrics}
        if args.wandb:
            wandb.log(metrics, step=epoch)
        
        # Save checkpoint if validation loss improved
        val_loss = val_metrics["val/loss"]
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch,
            args,
            metrics,
            is_best,
            Path(args.checkpoint_dir)
        )
        
        # Log metrics
        logger.info("Metrics:")
        for name, value in metrics.items():
            logger.info(f"{name}: {value:.4f}")
    
    logger.info("Training complete!")
    
    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 
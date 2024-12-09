"""training script for style_classifier with visualization"""

import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import wandb
import json
from datetime import datetime
import argparse
from typing import Any, Tuple
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from src.models.style_classifier import StyleClassifier
from src.data.dataset import StyleDataset
from src.utils.trainer import Trainer
from src.utils.evaluator import Evaluator
from src.utils.wandb_logger import WandbLogger
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_embedding_visualization,
    plot_attribute_correlations
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train style classifier model')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                      help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='learning rate')
    parser.add_argument('--device', type=str,
                      default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='device to use')
    parser.add_argument('--wandb', action='store_true',
                      help='use wandb logging')
    parser.add_argument('--output_dir', type=str, default='outputs',
                      help='directory to save outputs')
    parser.add_argument('--data_dir', type=str, default='data_deepfashion',
                      help='path to data directory')
    parser.add_argument('--test_mode', action='store_true',
                      help='run in test mode with smaller dataset')
    return parser.parse_args()

def get_train_transform():
    """Get training data transforms with augmentation"""
    return T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transform():
    """Get validation/test data transforms"""
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def visualize_predictions(
    model: nn.Module,
    val_loader: DataLoader,
    epoch: int,
    save_dir: Path
):
    """visualize model predictions
    
    args:
        model: trained model
        val_loader: validation data loader
        epoch: current epoch
        save_dir: directory to save visualizations
    """
    model.eval()
    fig_dir = save_dir / f'epoch_{epoch}'
    fig_dir.mkdir(exist_ok=True, parents=True)
    
    # Get a batch of data
    batch = next(iter(val_loader))
    inputs, targets = batch
    
    with torch.no_grad():
        outputs = model(inputs)
        
        # Plot predictions for each attribute group
        for group_name, group_pred in outputs['group_predictions'].items():
            # Get corresponding target indices
            indices = model.ATTRIBUTE_GROUPS[group_name]
            group_targets = targets[:, indices]
            
            # Convert predictions to probabilities
            probs = torch.sigmoid(group_pred)
            
            # Create confusion matrix
            pred_labels = (probs > 0.5).float()
            conf_matrix = torch.zeros(len(indices), len(indices))
            
            for i in range(len(indices)):
                for j in range(len(indices)):
                    conf_matrix[i, j] = ((pred_labels[:, i] == 1) & (group_targets[:, j] == 1)).sum()
            
            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix.numpy(), annot=True, fmt='g')
            plt.title(f'{group_name} Attribute Predictions')
            plt.xlabel('True')
            plt.ylabel('Predicted')
            plt.savefig(fig_dir / f'{group_name}_confusion.png')
            plt.close()
            
            # Plot prediction distribution
            plt.figure(figsize=(12, 6))
            sns.boxplot(data=probs.numpy())
            plt.title(f'{group_name} Prediction Distribution')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(fig_dir / f'{group_name}_distribution.png')
            plt.close()
            
        # Visualize attention patterns using fused embeddings
        fused_emb = outputs['fused_embedding']
        similarity = torch.matmul(fused_emb, fused_emb.t())
        
        plt.figure(figsize=(10, 10))
        sns.heatmap(similarity.numpy(), cmap='viridis')
        plt.title('Sample Similarity Matrix')
        plt.savefig(fig_dir / 'similarity_matrix.png')
        plt.close()

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: str,
    epoch: int,
    wandb_run: Any = None
) -> Tuple[float, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc=f"Training")
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        targets = batch['attributes'].to(device)
        
        # Generate positive/negative pairs for contrastive learning
        # Randomly select pairs within same/different style groups
        batch_size = images.size(0)
        if batch_size > 1:  # Only do contrastive if batch has multiple samples
            # Get style attributes
            style_attrs = targets[:, 23:26]  # indices 23-25 are style attributes
            
            # Find samples with similar and different styles
            style_sims = torch.matmul(style_attrs, style_attrs.t())
            similar_pairs = torch.where(style_sims > 0.5)  # Pairs with similar styles
            different_pairs = torch.where(style_sims < 0.2)  # Pairs with different styles
            
            # Randomly select pairs if available
            if len(similar_pairs[0]) > 0 and len(different_pairs[0]) > 0:
                pos_idx = torch.randint(0, len(similar_pairs[0]), (batch_size,))
                neg_idx = torch.randint(0, len(different_pairs[0]), (batch_size,))
                
                positive_pairs = (
                    images[similar_pairs[0][pos_idx]],
                    images[similar_pairs[1][pos_idx]]
                )
                negative_pairs = (
                    images[different_pairs[0][neg_idx]],
                    images[different_pairs[1][neg_idx]]
                )
                
                # Get embeddings for pairs
                with torch.no_grad():
                    pos_out = model(positive_pairs[0])
                    neg_out = model(negative_pairs[0])
                    style_pairs = (
                        pos_out['style_embedding'],
                        neg_out['style_embedding']
                    )
            else:
                style_pairs = None
        else:
            style_pairs = None
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss with contrastive learning
        loss, loss_dict = model.compute_loss(
            outputs,
            targets,
            style_pairs=style_pairs
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        predictions = torch.sigmoid(outputs['predictions'])
        correct += ((predictions > 0.5) == targets).float().sum().item()
        total += targets.numel()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'avg_loss': total_loss / (batch_idx + 1)
        })
        
        # Log to wandb
        if wandb_run is not None:
            wandb_run.log({
                'batch_loss': loss.item(),
                'batch_accuracy': ((predictions > 0.5) == targets).float().mean().item()
            })
            
            # Log individual group losses
            for name, value in loss_dict.items():
                wandb_run.log({f'batch_{name}': value})
    
    scheduler.step()
    
    return total_loss / len(train_loader), correct / total

def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> dict:
    """validate model
    
    args:
        model: model to validate
        val_loader: validation data loader
        criterion: loss function
        device: device to validate on
        
    returns:
        dict of metrics
    """
    model.eval()
    total_loss = 0
    group_losses = {name: 0.0 for name in model.ATTRIBUTE_GROUPS}
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Compute loss
            loss, loss_dict = model.compute_loss(output, target)
            
            # Update metrics
            total_loss += loss.item()
            for name in model.ATTRIBUTE_GROUPS:
                group_losses[name] += loss_dict[f"{name}_loss"]
    
    # Compute average losses
    num_batches = len(val_loader)
    metrics = {
        'val_loss': total_loss / num_batches,
        **{f"val_{name}_loss": group_losses[name] / num_batches
           for name in model.ATTRIBUTE_GROUPS}
    }
    
    return metrics

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'using device: {device}')
    
    # Initialize wandb
    if args.wandb:
        wandb_run = wandb.init(
            project='style-classifier',
            config=vars(args)
        )
    else:
        wandb_run = None
    
    # Create datasets
    logger.info('creating datasets...')
    train_dataset = StyleDataset(
        args.data_dir,
        split='train',
        transform=get_train_transform()
    )
    val_dataset = StyleDataset(
        args.data_dir,
        split='val',
        transform=get_val_transform()
    )
    test_dataset = StyleDataset(
        args.data_dir,
        split='test',
        transform=get_val_transform()
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    logger.info('creating model...')
    model = StyleClassifier(
        num_attributes=26,
        hidden_size=256,
        num_layers=3,
        dropout=0.5,
        device=device
    ).to(device)
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=None,  # Not needed with new model
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        early_stopping_patience=10,
        wandb_run=wandb_run
    )
    
    # Train model
    logger.info('starting training...')
    trainer.train()
    
    # Evaluate on test set
    logger.info('evaluating on test set...')
    model.load_state_dict(torch.load('best_model.pt', weights_only=True))
    test_loss, test_acc = evaluate_model(
        model,
        test_loader,
        device,
        wandb_run
    )
    
    # Log final metrics
    if wandb_run is not None:
        wandb_run.log({
            'test_loss': test_loss,
            'test_accuracy': test_acc
        })
        
        # Generate visualizations
        logger.info('generating visualizations...')
        visualize_predictions(
            model,
            test_loader,
            device,
            wandb_run,
            num_samples=16
        )
    
    # Save final model
    logger.info('saving final model...')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'test_accuracy': test_acc,
        'test_loss': test_loss
    }, os.path.join(args.output_dir, 'final_model.pt'))
    
    logger.info('pipeline complete!')
    
    if wandb_run is not None:
        wandb_run.finish()

def evaluate_model(model, data_loader, device, wandb_run=None):
    """Evaluate model on dataset"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # Store per-attribute accuracies
    attr_correct = torch.zeros(26, device=device)
    attr_total = torch.zeros(26, device=device)
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            images = batch['image'].to(device)
            targets = batch['attributes'].to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss, _ = model.compute_loss(outputs, targets)
            
            # Update metrics
            total_loss += loss.item()
            predictions = torch.sigmoid(outputs['predictions'])
            pred_labels = predictions > 0.5
            
            # Overall accuracy
            correct += (pred_labels == targets).float().sum().item()
            total += targets.numel()
            
            # Per-attribute accuracy
            attr_correct += (pred_labels == targets).float().sum(dim=0)
            attr_total += targets.size(0)
    
    # Compute final metrics
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    
    # Log per-attribute accuracies
    if wandb_run is not None:
        attr_accuracies = (attr_correct / attr_total).cpu().numpy()
        for i, acc in enumerate(attr_accuracies):
            wandb_run.log({f'test_attr_{i}_accuracy': acc})
    
    return avg_loss, accuracy

def visualize_predictions(model, data_loader, device, wandb_run, num_samples=16):
    """Generate visualization of model predictions"""
    model.eval()
    all_images = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Getting predictions'):
            images = batch['image'].to(device)
            targets = batch['attributes']
            
            # Get predictions
            outputs = model(images)
            predictions = torch.sigmoid(outputs['predictions'])
            
            all_images.extend(images.cpu())
            all_preds.extend(predictions.cpu())
            all_targets.extend(targets)
            
            if len(all_images) >= num_samples:
                break
    
    # Convert to numpy arrays
    images = torch.stack(all_images[:num_samples])
    preds = torch.stack(all_preds[:num_samples])
    targets = torch.stack(all_targets[:num_samples])
    
    # Create visualization grid
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    for idx, (img, pred, target) in enumerate(zip(images, preds, targets)):
        ax = axes[idx // 4, idx % 4]
        
        # Show image
        img = img.permute(1, 2, 0)
        ax.imshow(img)
        
        # Add predictions vs targets
        text = []
        for i, (p, t) in enumerate(zip(pred, target)):
            if abs(p - t) > 0.5:  # Show incorrect predictions
                text.append(f'Attr {i}: Pred={p:.2f}, True={t:.0f}')
        
        ax.set_title('\n'.join(text[:3]))  # Show top 3 mistakes
        ax.axis('off')
    
    plt.tight_layout()
    
    # Log to wandb
    if wandb_run is not None:
        wandb_run.log({'prediction_samples': wandb.Image(plt)})
    
    plt.close()

if __name__ == '__main__':
    main() 
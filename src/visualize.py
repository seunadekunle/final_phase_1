"""visualization tools for model analysis"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional

def plot_training_curves(
    metrics_files: List[str],
    save_dir: Path,
    metrics: Optional[List[str]] = None
):
    """plot training curves comparing different models
    
    args:
        metrics_files: list of paths to metrics json files
        save_dir: directory to save plots
        metrics: optional list of metrics to plot
    """
    # load metrics
    all_metrics = {}
    for file in metrics_files:
        model_name = Path(file).stem
        with open(file) as f:
            all_metrics[model_name] = json.load(f)
    
    if metrics is None:
        # get all metrics that are common between models
        metrics = set.intersection(*[
            set(m.keys()) for m in all_metrics.values()
        ])
        metrics = [m for m in metrics if not m.startswith('_')]
    
    # create plots
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for model_name, model_metrics in all_metrics.items():
            if metric in model_metrics:
                values = model_metrics[metric]
                epochs = range(1, len(values) + 1)
                plt.plot(epochs, values, label=model_name)
        
        plt.title(f'{metric} vs Epoch')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.savefig(save_dir / f'{metric}_comparison.png')
        plt.close()

def plot_attention_weights(
    attention_weights: torch.Tensor,
    save_path: Path,
    layer_name: str = "attention"
):
    """plot attention weight heatmap
    
    args:
        attention_weights: attention weights tensor
        save_path: path to save plot
        layer_name: name of attention layer
    """
    weights = attention_weights.cpu().numpy()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        weights,
        cmap='viridis',
        center=0,
        annot=True,
        fmt='.2f'
    )
    plt.title(f'{layer_name} Attention Weights')
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    plt.savefig(save_path)
    plt.close()

def plot_feature_importance(
    model_outputs: Dict[str, torch.Tensor],
    attribute_names: List[str],
    save_dir: Path
):
    """plot feature importance analysis
    
    args:
        model_outputs: dict containing model predictions and features
        attribute_names: list of attribute names
        save_dir: directory to save plots
    """
    # get predictions and features
    predictions = model_outputs['attribute_predictions']
    features = model_outputs['embeddings']
    
    # compute feature importance using gradient-based attribution
    importance = torch.autograd.grad(
        predictions.sum(),
        features,
        create_graph=True
    )[0]
    
    importance = importance.abs().mean(0).cpu().numpy()
    
    # plot overall feature importance
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(importance)), importance)
    plt.title('Overall Feature Importance')
    plt.xlabel('Feature Index')
    plt.ylabel('Importance Score')
    plt.savefig(save_dir / 'feature_importance.png')
    plt.close()
    
    # plot per-attribute feature importance
    for i, attr_name in enumerate(attribute_names):
        attr_importance = torch.autograd.grad(
            predictions[:, i].sum(),
            features,
            create_graph=True
        )[0]
        
        attr_importance = attr_importance.abs().mean(0).cpu().numpy()
        
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(attr_importance)), attr_importance)
        plt.title(f'Feature Importance for {attr_name}')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance Score')
        plt.savefig(save_dir / f'feature_importance_{attr_name}.png')
        plt.close()

def plot_interactive_embeddings(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    attribute_names: List[str],
    save_path: Path
):
    """create interactive t-SNE plot of embeddings
    
    args:
        embeddings: feature embeddings tensor
        labels: attribute labels tensor
        attribute_names: list of attribute names
        save_path: path to save plot
    """
    # reduce dimensionality
    tsne = TSNE(n_components=3, random_state=42)
    embeddings_3d = tsne.fit_transform(embeddings.cpu().numpy())
    
    # create dataframe
    df = pd.DataFrame(embeddings_3d, columns=['x', 'y', 'z'])
    
    # add attribute information
    for i, attr_name in enumerate(attribute_names):
        df[attr_name] = labels[:, i].cpu().numpy()
    
    # create interactive plot
    fig = px.scatter_3d(
        df,
        x='x',
        y='y',
        z='z',
        color=attribute_names[0],  # color by first attribute
        hover_data=attribute_names,
        title='3D t-SNE Visualization of Feature Embeddings'
    )
    
    # update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='t-SNE 1',
            yaxis_title='t-SNE 2',
            zaxis_title='t-SNE 3'
        )
    )
    
    # save
    fig.write_html(str(save_path))

def plot_attribute_relationships(
    predictions: torch.Tensor,
    attribute_names: List[str],
    save_dir: Path
):
    """analyze and plot attribute relationships
    
    args:
        predictions: model predictions tensor
        attribute_names: list of attribute names
        save_dir: directory to save plots
    """
    # compute correlation matrix
    probs = torch.sigmoid(predictions).cpu().numpy()
    correlations = np.corrcoef(probs.T)
    
    # plot correlation heatmap
    plt.figure(figsize=(15, 12))
    sns.heatmap(
        correlations,
        xticklabels=attribute_names,
        yticklabels=attribute_names,
        cmap='RdBu',
        center=0,
        annot=True,
        fmt='.2f'
    )
    plt.title('Attribute Correlations')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_dir / 'attribute_correlations.png')
    plt.close()
    
    # find most correlated pairs
    n_attrs = len(attribute_names)
    pairs = []
    for i in range(n_attrs):
        for j in range(i+1, n_attrs):
            pairs.append((
                attribute_names[i],
                attribute_names[j],
                correlations[i, j]
            ))
    
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # plot top correlations
    top_n = 10
    plt.figure(figsize=(12, 6))
    
    names = [f"{p[0]}\n{p[1]}" for p in pairs[:top_n]]
    values = [p[2] for p in pairs[:top_n]]
    
    plt.bar(range(top_n), values)
    plt.xticks(range(top_n), names, rotation=45, ha='right')
    plt.title(f'Top {top_n} Attribute Correlations')
    plt.ylabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig(save_dir / 'top_correlations.png')
    plt.close()

def create_analysis_report(
    model_outputs: Dict[str, torch.Tensor],
    attribute_names: List[str],
    metrics: Dict[str, float],
    save_dir: Path
):
    """create comprehensive analysis report
    
    args:
        model_outputs: dict containing model predictions and features
        attribute_names: list of attribute names
        metrics: dict of evaluation metrics
        save_dir: directory to save report
    """
    # create report directory
    report_dir = save_dir / 'analysis_report'
    report_dir.mkdir(exist_ok=True)
    
    # save metrics summary
    with open(report_dir / 'metrics_summary.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # plot feature visualizations
    plot_feature_importance(
        model_outputs,
        attribute_names,
        report_dir
    )
    
    # plot interactive embeddings
    plot_interactive_embeddings(
        model_outputs['embeddings'],
        model_outputs['targets'],
        attribute_names,
        report_dir / 'embeddings_visualization.html'
    )
    
    # plot attribute relationships
    plot_attribute_relationships(
        model_outputs['predictions'],
        attribute_names,
        report_dir
    )
    
    # create performance breakdown
    performance = {}
    predictions = model_outputs['predictions']
    targets = model_outputs['targets']
    
    for i, attr_name in enumerate(attribute_names):
        # compute metrics for this attribute
        pred = predictions[:, i]
        target = targets[:, i]
        
        true_pos = ((torch.sigmoid(pred) > 0.5) & (target == 1)).sum().item()
        false_pos = ((torch.sigmoid(pred) > 0.5) & (target == 0)).sum().item()
        true_neg = ((torch.sigmoid(pred) <= 0.5) & (target == 0)).sum().item()
        false_neg = ((torch.sigmoid(pred) <= 0.5) & (target == 1)).sum().item()
        
        total = len(target)
        accuracy = (true_pos + true_neg) / total
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        performance[attr_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': int(target.sum().item())
        }
    
    # save performance breakdown
    with open(report_dir / 'attribute_performance.json', 'w') as f:
        json.dump(performance, f, indent=2)
    
    # plot performance summary
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    
    for metric in metrics_to_plot:
        values = [perf[metric] for perf in performance.values()]
        
        plt.figure(figsize=(15, 6))
        plt.bar(attribute_names, values)
        plt.title(f'{metric.title()} per Attribute')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel(metric.title())
        plt.tight_layout()
        plt.savefig(report_dir / f'{metric}_per_attribute.png')
        plt.close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing model outputs and metrics')
    parser.add_argument('--save_dir', type=str, default='visualizations',
                       help='Directory to save visualizations')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # plot training curves
    metrics_files = list(results_dir.glob('**/metrics.json'))
    plot_training_curves(metrics_files, save_dir)
    
    # load model outputs and create analysis report
    for model_dir in results_dir.iterdir():
        if model_dir.is_dir():
            outputs_file = model_dir / 'model_outputs.pt'
            if outputs_file.exists():
                print(f"\nAnalyzing {model_dir.name}")
                
                # load data
                model_outputs = torch.load(outputs_file)
                with open(model_dir / 'metrics.json') as f:
                    metrics = json.load(f)
                with open(model_dir / 'attribute_names.json') as f:
                    attribute_names = json.load(f)
                
                # create report
                create_analysis_report(
                    model_outputs,
                    attribute_names,
                    metrics,
                    save_dir / model_dir.name
                ) 
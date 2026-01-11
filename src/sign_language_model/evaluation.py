"""
Evaluation utilities for ASL classification.
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names
    
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision_weighted': precision,
        'recall_weighted': recall,
        'f1_weighted': f1,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
    }
    
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (12, 10),
    normalize: bool = False,
    save_path: Optional[str] = None,
):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Optional list of class names
        title: Plot title
        figsize: Figure size
        normalize: Whether to normalize the matrix
        save_path: Optional path to save figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    # Generate default class names if not provided
    if class_names is None:
        num_classes = cm.shape[0]
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        linewidths=0.5,
    )
    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
):
    """
    Print detailed classification report.
    """
    print("\n" + "="*60)
    print("Classification Report")
    print("="*60)
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
            digits=4,
        )
    )


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: Optional[torch.nn.Module] = None,
    return_predictions: bool = False,
) -> Dict:
    """
    Evaluate a model on a dataloader.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to run on
        criterion: Optional loss function
        return_predictions: Whether to return predictions and labels
    
    Returns:
        Dictionary with metrics and optionally predictions
    """
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            # Handle different batch formats
            if isinstance(batch, dict):
                labels = batch["y"]
                # Extract features based on what's available
                feats = []
                if "rgb" in batch:
                    feats.append(batch["rgb"].to(device))
                if "flow" in batch:
                    feats.append(batch["flow"].to(device))
                if "kps" in batch:
                    feats.append(batch["kps"].to(device))
                
                # Forward pass depends on model structure
                if len(feats) == 1:
                    logits = model(feats[0])
                elif len(feats) == 2:
                    logits = model(feats[0], feats[1])
                elif len(feats) == 3:
                    logits = model(feats[0], feats[1], feats[2])
                else:
                    # Try dict input
                    batch_dict = {k: v.to(device) for k, v in batch.items() if k != "y"}
                    logits = model(batch_dict)
            else:
                # Assume (x, y) tuple
                x, labels = batch
                x = x.to(device)
                logits = model(x)
            
            labels = labels.to(device)
            
            if criterion is not None:
                # Handle one-hot labels for MixUp
                if labels.dim() > 1 and labels.size(1) > 1:
                    # One-hot labels - convert to class indices for loss
                    labels_idx = labels.argmax(dim=1)
                    loss = criterion(logits, labels_idx)
                else:
                    loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
            
            preds = logits.argmax(dim=1).cpu().numpy()
            
            # Convert labels to numpy
            if labels.dim() > 1 and labels.size(1) > 1:
                labels_np = labels.argmax(dim=1).cpu().numpy()
            else:
                labels_np = labels.cpu().numpy()
            
            all_preds.append(preds)
            all_labels.append(labels_np)
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    metrics = compute_metrics(all_labels, all_preds)
    
    if criterion is not None:
        metrics['loss'] = total_loss / len(all_labels)
    
    if return_predictions:
        metrics['predictions'] = all_preds
        metrics['labels'] = all_labels
    
    return metrics


def compare_models(
    results: Dict[str, Dict],
    metric: str = 'accuracy',
    save_path: Optional[str] = None,
):
    """
    Compare multiple models on a given metric.
    
    Args:
        results: Dictionary mapping model names to result dictionaries
        metric: Metric to compare
        save_path: Optional path to save figure
    """
    model_names = list(results.keys())
    metric_values = [results[name].get(metric, 0) for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, metric_values)
    plt.ylabel(metric.capitalize())
    plt.title(f'Model Comparison: {metric.capitalize()}')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, metric_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{val:.4f}',
            ha='center',
            va='bottom',
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


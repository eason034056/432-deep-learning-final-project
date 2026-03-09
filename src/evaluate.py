"""
Evaluation script for trained models.

This script provides:
- Model evaluation on test set
- Confusion matrix visualization
- Per-class accuracy metrics
- Model comparison across architectures
- Feature visualization (for PointNet)

Usage:
    python evaluate.py --model pointnet --checkpoint results/checkpoints/pointnet/model_best.pth
    python evaluate.py --compare --models mlp cnn1d pointnet
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import load_processed_dataset, stratified_split_grouped, create_dataloaders
from models import MLPBaseline, CNN1DModel, TinyPointNet
from train import load_config, create_model


def evaluate_model(model: nn.Module,
                  test_loader: DataLoader,
                  device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model and return predictions and true labels.
    
    Args:
        model: trained model to evaluate
        test_loader: DataLoader for test set
        device: 'cuda' or 'cpu'
        
    Returns:
        y_true: true labels, shape (N,)
        y_pred: predicted labels, shape (N,)
        y_prob: prediction probabilities, shape (N, num_classes)
        
    Example:
        >>> y_true, y_pred, y_prob = evaluate_model(model, test_loader, 'cuda')
        >>> accuracy = (y_true == y_pred).mean()
    """
    # Set model to evaluation mode
    model.eval()
    
    # Lists to collect results
    all_labels = []  # True labels
    all_preds = []   # Predicted labels
    all_probs = []   # Prediction probabilities
    
    # Disable gradient computation
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        
        for data, labels in pbar:
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(data)  # (B, num_classes) logits
            
            # Get probabilities via softmax
            probs = torch.softmax(outputs, dim=1)  # (B, num_classes)
            
            # Get predicted class (highest probability)
            _, predicted = torch.max(outputs, 1)
            
            # Collect results
            all_labels.append(labels.cpu().numpy())
            all_preds.append(predicted.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # Concatenate all batches
    y_true = np.concatenate(all_labels)  # (N,)
    y_pred = np.concatenate(all_preds)   # (N,)
    y_prob = np.concatenate(all_probs)   # (N, num_classes)
    
    return y_true, y_pred, y_prob


def plot_confusion_matrix(y_true: np.ndarray,
                         y_pred: np.ndarray,
                         class_names: List[str],
                         save_path: str,
                         title: str = "Confusion Matrix") -> None:
    """
    Plot and save confusion matrix.
    
    Confusion matrix shows:
    - Rows: true labels
    - Columns: predicted labels
    - Diagonal: correct predictions
    - Off-diagonal: misclassifications
    
    Args:
        y_true: true labels
        y_pred: predicted labels
        class_names: list of class names for labels
        save_path: where to save the figure
        title: plot title
        
    Example:
        >>> plot_confusion_matrix(y_true, y_pred, 
        ...                      class_names=['Subject 0', ..., 'Subject 9'],
        ...                      save_path='results/confusion_matrix.png')
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize by true labels (row-wise) to get percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title(f'{title} (Counts)')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # Plot 2: Normalized percentages
    sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=axes[1], cbar_kws={'label': 'Percentage (%)'})
    axes[1].set_title(f'{title} (Normalized %)')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_roc_curves(y_true: np.ndarray,
                   y_prob: np.ndarray,
                   class_names: List[str],
                   save_path: str,
                   num_classes: int = 10) -> float:
    """
    Plot ROC curves for multi-class classification (one-vs-rest) and compute AUC.
    
    Args:
        y_true: true labels, shape (N,)
        y_prob: prediction probabilities, shape (N, num_classes)
        class_names: list of class names
        save_path: where to save the figure
        num_classes: number of classes
        
    Returns:
        auc_macro: macro-averaged AUC score
    """
    # Binarize labels for one-vs-rest
    y_true_bin = label_binarize(y_true, classes=range(num_classes))
    
    # Compute ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
    
    # Macro-averaged AUC
    auc_macro = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    
    # Plot all ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    
    for i in range(num_classes):
        ax.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc[i]:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curves (One-vs-Rest) - Macro AUC = {auc_macro:.3f}')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curves saved to {save_path}")
    plt.close()
    
    return auc_macro


def plot_per_class_metrics(y_true: np.ndarray,
                           y_pred: np.ndarray,
                           class_names: List[str],
                           save_path: str) -> None:
    """
    Plot per-class precision, recall, and F1-score.
    
    Args:
        y_true: true labels
        y_pred: predicted labels
        class_names: list of class names
        save_path: where to save the figure
        
    Example:
        >>> plot_per_class_metrics(y_true, y_pred, class_names, 
        ...                        save_path='results/per_class_metrics.png')
    """
    # Generate classification report as dictionary
    report = classification_report(y_true, y_pred, output_dict=True,
                                   target_names=class_names, zero_division=0)
    
    # Extract per-class metrics
    metrics_dict = {
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }
    
    for class_name in class_names:
        metrics_dict['Precision'].append(report[class_name]['precision'] * 100)
        metrics_dict['Recall'].append(report[class_name]['recall'] * 100)
        metrics_dict['F1-Score'].append(report[class_name]['f1-score'] * 100)
    
    # Create DataFrame for easy plotting
    df = pd.DataFrame(metrics_dict, index=class_names)
    
    # Plot grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(kind='bar', ax=ax, width=0.8)
    ax.set_xlabel('Class')
    ax.set_ylabel('Score (%)')
    ax.set_title('Per-Class Metrics')
    ax.set_ylim([0, 105])
    ax.legend(loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-class metrics saved to {save_path}")
    plt.close()


def print_evaluation_summary(y_true: np.ndarray,
                            y_pred: np.ndarray,
                            class_names: List[str],
                            y_prob: Optional[np.ndarray] = None,
                            num_classes: int = 10) -> Dict[str, float]:
    """
    Print comprehensive evaluation summary.
    
    Args:
        y_true: true labels
        y_pred: predicted labels
        class_names: list of class names
        
    Returns:
        metrics: dictionary of overall metrics
        
    Prints:
        - Overall accuracy
        - Macro/Micro averaged precision, recall, F1
        - Per-class detailed report
    """
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred) * 100
    
    # Macro average: average of per-class metrics (treats all classes equally)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    
    # Micro average: aggregate contributions of all classes (weighted by support)
    precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0) * 100
    recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0) * 100
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0) * 100
    
    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    print(f"\nMacro Averaged Metrics:")
    print(f"  Precision: {precision_macro:.2f}%")
    print(f"  Recall: {recall_macro:.2f}%")
    print(f"  F1-Score: {f1_macro:.2f}%")
    print(f"\nMicro Averaged Metrics:")
    print(f"  Precision: {precision_micro:.2f}%")
    print(f"  Recall: {recall_micro:.2f}%")
    print(f"  F1-Score: {f1_micro:.2f}%")
    
    # ROC-AUC if probabilities provided
    auc_macro = 0.0
    if y_prob is not None:
        try:
            auc_macro = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            print(f"\nROC-AUC (macro, one-vs-rest): {auc_macro:.4f}")
        except ValueError:
            print("\nROC-AUC: Could not compute (e.g., single class in test set)")
    
    # Detailed per-class report
    print("\n" + "=" * 80)
    print("PER-CLASS CLASSIFICATION REPORT")
    print("=" * 80)
    print(classification_report(y_true, y_pred, target_names=class_names, 
                                digits=4, zero_division=0))
    
    # Return metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_micro': precision_micro,
        'recall_micro': recall_micro,
        'f1_micro': f1_micro,
        'auc_macro': auc_macro if y_prob is not None else 0.0
    }
    
    return metrics


def compare_models(model_names: List[str],
                  checkpoint_paths: List[str],
                  config: Dict,
                  test_loader: DataLoader,
                  device: str,
                  save_path: str) -> pd.DataFrame:
    """
    Compare multiple trained models.
    
    Args:
        model_names: list of model types ['mlp', 'cnn1d', 'pointnet']
        checkpoint_paths: list of checkpoint paths for each model
        config: configuration dictionary
        test_loader: DataLoader for test set
        device: 'cuda' or 'cpu'
        save_path: where to save comparison table
        
    Returns:
        comparison_df: DataFrame with comparison results
        
    Example:
        >>> models = ['mlp', 'cnn1d', 'pointnet']
        >>> checkpoints = ['results/checkpoints/mlp/model_best.pth', ...]
        >>> df = compare_models(models, checkpoints, config, test_loader, 'cuda',
        ...                     save_path='results/comparison.csv')
    """
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    
    comparison_results = []
    
    for model_name, checkpoint_path in zip(model_names, checkpoint_paths):
        print(f"\n{'-' * 80}")
        print(f"Evaluating {model_name.upper()} model")
        print(f"{'-' * 80}")
        
        # Load model
        num_classes = 10  # FAUST has 10 subjects
        model = create_model(model_name, num_classes, config)
        model = model.to(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Checkpoint epoch: {checkpoint['epoch']}")
        print(f"Checkpoint val_acc: {checkpoint['val_acc']:.2f}%")
        
        # Evaluate
        y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred) * 100
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
        auc_macro = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Store results
        comparison_results.append({
            'Model': model_name.upper(),
            'Test Accuracy (%)': f"{accuracy:.2f}",
            'F1-Score (%)': f"{f1_macro:.2f}",
            'ROC-AUC': f"{auc_macro:.4f}",
            'Precision (%)': f"{precision:.2f}",
            'Recall (%)': f"{recall:.2f}",
            'Parameters': f"{num_params:,}",
            'Checkpoint Epoch': checkpoint['epoch'],
            'Checkpoint Val Acc (%)': f"{checkpoint['val_acc']:.2f}"
        })
        
        print(f"\nResults:")
        print(f"  Test Accuracy: {accuracy:.2f}%")
        print(f"  F1-Score: {f1_macro:.2f}%")
        print(f"  ROC-AUC: {auc_macro:.4f}")
        print(f"  Parameters: {num_params:,}")
    
    # Create DataFrame
    comparison_df = pd.DataFrame(comparison_results)
    
    # Save to CSV
    comparison_df.to_csv(save_path, index=False)
    print(f"\n{'-' * 80}")
    print(f"Comparison table saved to {save_path}")
    print(f"{'-' * 80}\n")
    print(comparison_df.to_string(index=False))
    
    return comparison_df


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained point cloud classification models'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['mlp', 'cnn1d', 'pointnet', 'pointnet2'],
        help='Model type to evaluate (single model mode)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to model checkpoint (single model mode)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple models'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['mlp', 'cnn1d', 'pointnet2'],
        help='Models to compare (comparison mode)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test dataset
    print("\nLoading test dataset...")
    processed_path = Path(config['data']['processed_dir']) / 'faust_pc.npz'
    data, labels, filenames, metadata = load_processed_dataset(str(processed_path))
    
    # Get samples_per_mesh from metadata or config
    samples_per_mesh = metadata.get('samples_per_mesh', config['data'].get('samples_per_mesh', 100))
    
    # Split dataset (same split as training, grouped by mesh to avoid data leakage)
    _, _, _, _, X_test, y_test = stratified_split_grouped(
        data, labels, filenames, samples_per_mesh,
        train_ratio=config['split']['train_ratio'],
        val_ratio=config['split']['val_ratio'],
        test_ratio=config['split']['test_ratio'],
        random_seed=config['split']['seed']
    )
    
    # Create test loader
    from dataset import FAUSTPointCloudDataset
    test_dataset = FAUSTPointCloudDataset(X_test, y_test, augment=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    # Class names for visualization
    class_names = [f'Subject {i}' for i in range(10)]
    
    # Results directory
    results_dir = Path(config['logging']['log_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Mode 1: Single model evaluation
    if not args.compare:
        if not args.model or not args.checkpoint:
            parser.error("--model and --checkpoint required for single model evaluation")
        
        print(f"\n{'=' * 80}")
        print(f"Evaluating {args.model.upper()} model")
        print(f"{'=' * 80}")
        
        # Load model
        num_classes = 10
        model = create_model(args.model, num_classes, config)
        model = model.to(device)
        
        # Load checkpoint
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded checkpoint from {args.checkpoint}")
        
        # Evaluate
        y_true, y_pred, y_prob = evaluate_model(model, test_loader, device)
        
        # Print summary (with ROC-AUC)
        num_classes = len(class_names)
        metrics = print_evaluation_summary(y_true, y_pred, class_names,
                                          y_prob=y_prob, num_classes=num_classes)
        
        # Save visualizations
        cm_path = results_dir / f'confusion_matrix_{args.model}.png'
        plot_confusion_matrix(y_true, y_pred, class_names, str(cm_path),
                            title=f"{args.model.upper()} Confusion Matrix")
        
        metrics_path = results_dir / f'per_class_metrics_{args.model}.png'
        plot_per_class_metrics(y_true, y_pred, class_names, str(metrics_path))
        
        # ROC curves
        roc_path = results_dir / f'roc_curves_{args.model}.png'
        plot_roc_curves(y_true, y_prob, class_names, str(roc_path), num_classes=num_classes)
        
        print(f"\n{'=' * 80}")
        print("Evaluation complete!")
        print(f"{'=' * 80}")
    
    # Mode 2: Multi-model comparison
    else:
        print(f"\n{'=' * 80}")
        print("Multi-Model Comparison Mode")
        print(f"{'=' * 80}")
        
        # Find checkpoints for each model
        checkpoint_dir = Path(config['logging']['log_dir']) / 'checkpoints'
        checkpoint_paths = []
        
        for model_name in args.models:
            checkpoint_path = checkpoint_dir / model_name / 'model_best.pth'
            if not checkpoint_path.exists():
                print(f"Warning: Checkpoint not found for {model_name} at {checkpoint_path}")
                print(f"Skipping {model_name}")
                continue
            checkpoint_paths.append(str(checkpoint_path))
        
        if len(checkpoint_paths) == 0:
            print("Error: No valid checkpoints found")
            return
        
        # Compare models
        comparison_path = results_dir / 'model_comparison.csv'
        comparison_df = compare_models(
            args.models, checkpoint_paths, config,
            test_loader, device, str(comparison_path)
        )
        
        print(f"\n{'=' * 80}")
        print("Comparison complete!")
        print(f"{'=' * 80}")


if __name__ == '__main__':
    main()


"""
Training script for point cloud classification models.

This script handles:
- Model initialization from config
- Training loop with progress tracking
- Validation during training
- Model checkpointing
- TensorBoard logging
- Early stopping

Usage:
    python train.py --config config.yaml --model pointnet
    python train.py --config config.yaml --model mlp --epochs 100
"""

import os
import sys
import argparse
import yaml
import time
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import (
    load_faust_dataset,
    stratified_split_grouped,
    create_dataloaders,
    save_processed_dataset,
    load_processed_dataset
)

from models import MLPBaseline, CNN1DModel, TinyPointNet


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: path to config.yaml file
        
    Returns:
        config: dictionary containing all configuration parameters
        
    Example:
        >>> config = load_config('config.yaml')
        >>> batch_size = config['training']['batch_size']
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(model_type: str, num_classes: int, config: Dict) -> nn.Module:
    """
    Create model instance based on model type.
    
    Args:
        model_type: 'mlp', 'cnn1d', or 'pointnet'
        num_classes: number of output classes
        config: configuration dictionary
        
    Returns:
        model: initialized PyTorch model
        
    Example:
        >>> model = create_model('pointnet', num_classes=10, config=cfg)
    """
    num_points = config['data']['num_points']  # 200
    dropout = config['model']['dropout']  # 0.3
    
    if model_type == 'mlp':
        # MLP baseline model
        model = MLPBaseline(
            num_points=num_points,
            num_channels=3,
            num_classes=num_classes,
            hidden_dims=(256, 128),
            dropout=dropout
        )
        print("Created MLP Baseline model")
        
    elif model_type == 'cnn1d':
        # 1D-CNN model
        kernel_size = config['model'].get('cnn1d_kernel_size', 1)  # Default to 1 (pointwise)
        model = CNN1DModel(
            num_points=num_points,
            num_channels=3,
            num_classes=num_classes,
            conv_channels=(64, 128, 256),
            fc_dims=(128,),
            dropout=dropout,
            kernel_size=kernel_size
        )
        print("Created CNN1D model")
        
    elif model_type == 'pointnet':
        # Tiny PointNet model
        model = TinyPointNet(
            num_points=num_points,
            num_channels=3,
            num_classes=num_classes,
            use_tnet=True,
            channel_dims=(64, 128, 1024),
            fc_dims=(512, 256),
            dropout=dropout
        )
        print("Created Tiny PointNet model")

    elif model_type == 'mmidnet':
        # MMIDNet: Transform Block + Residual CNN + Global Max Pool + Dense
        model = MMIDNet(
            num_points=num_points,
            num_channels=3,
            num_classes=num_classes,
            dropout=dropout
        )
        print("Created MMIDNet model")
        
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Choose from: mlp, cnn1d, pointnet, mmidnet")
    
    # Print model parameter count
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {num_params:,}")
    
    return model


def train_one_epoch(model: nn.Module,
                   train_loader: DataLoader,
                   criterion: nn.Module,
                   optimizer: optim.Optimizer,
                   device: str,
                   epoch: int) -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        criterion: loss function (e.g., CrossEntropyLoss)
        optimizer: optimizer (e.g., Adam)
        device: 'cuda' or 'cpu'
        epoch: current epoch number (for display)
        
    Returns:
        avg_loss: average loss over epoch
        accuracy: classification accuracy (0-100%)
        
    Process:
    1. Set model to train mode
    2. Iterate through batches
    3. Forward pass → compute loss
    4. Backward pass → update weights
    5. Track metrics
    """
    # Set model to training mode
    # This enables dropout and batch normalization training behavior
    model.train()
    
    # Initialize metrics tracking
    total_loss = 0.0  # Accumulate loss
    correct = 0  # Count correct predictions
    total = 0  # Count total samples
    
    # Progress bar for visualization
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (data, labels) in enumerate(pbar):
        # Move data to device (GPU or CPU)
        data = data.to(device)  # (B, N, 3)
        labels = labels.to(device)  # (B,)
        
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        
        # Forward pass: compute predictions
        outputs = model(data)  # (B, num_classes)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass: compute gradients
        loss.backward()
        
        # Update model parameters
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        
        # Get predicted class (highest score)
        _, predicted = torch.max(outputs.data, 1)  # Returns (values, indices)
        total += labels.size(0)  # Batch size
        correct += (predicted == labels).sum().item()  # Count matches
        
        # Update progress bar
        current_accuracy = 100.0 * correct / total
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{current_accuracy:.2f}%'
        })
    
    # Calculate epoch metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def validate(model: nn.Module,
            val_loader: DataLoader,
            criterion: nn.Module,
            device: str) -> Tuple[float, float]:
    """
    Evaluate model on validation set.
    
    Args:
        model: PyTorch model to evaluate
        val_loader: DataLoader for validation data
        criterion: loss function
        device: 'cuda' or 'cpu'
        
    Returns:
        avg_loss: average validation loss
        accuracy: validation accuracy (0-100%)
        
    Note: No gradient computation needed for validation (saves memory)
    """
    # Set model to evaluation mode
    # This disables dropout and uses batch norm running statistics
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="[Validation]")
        
        for data, labels in pbar:
            data = data.to(device)
            labels = labels.to(device)
            
            # Forward pass only
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            current_accuracy = 100.0 * correct / total
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_accuracy:.2f}%'
            })
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    
    Monitors validation loss and stops training if no improvement
    for a specified number of epochs (patience).
    
    Example:
        >>> early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        >>> for epoch in range(num_epochs):
        ...     val_loss = validate(...)
        ...     if early_stopping(val_loss):
        ...         print("Early stopping triggered!")
        ...         break
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Initialize early stopping.
        
        Args:
            patience: number of epochs to wait before stopping
            min_delta: minimum change to qualify as improvement
        """
        self.patience = patience  # How many epochs to wait
        self.min_delta = min_delta  # Minimum improvement threshold
        self.counter = 0  # Count epochs without improvement
        self.best_loss = None  # Track best validation loss
        self.early_stop = False  # Flag to signal stopping
        
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: current validation loss
            
        Returns:
            should_stop: True if training should stop, False otherwise
        """
        if self.best_loss is None:
            # First epoch: initialize best loss
            self.best_loss = val_loss
            
        elif val_loss > self.best_loss - self.min_delta:
            # No significant improvement
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                
        else:
            # Improvement detected
            self.best_loss = val_loss
            self.counter = 0  # Reset counter
        
        return self.early_stop


def save_checkpoint(model: nn.Module,
                   optimizer: optim.Optimizer,
                   epoch: int,
                   val_loss: float,
                   val_acc: float,
                   save_path: str) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: model to save
        optimizer: optimizer state to save
        epoch: current epoch number
        val_loss: validation loss
        val_acc: validation accuracy
        save_path: path to save checkpoint
        
    Example:
        >>> save_checkpoint(model, optimizer, epoch=50, 
        ...                 val_loss=0.5, val_acc=85.2,
        ...                 save_path='checkpoints/model_best.pth')
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model: nn.Module,
                   optimizer: optim.Optimizer,
                   checkpoint_path: str,
                   device: str) -> Tuple[int, float, float]:
    """
    Load model checkpoint.
    
    Args:
        model: model to load weights into
        optimizer: optimizer to load state into
        checkpoint_path: path to checkpoint file
        device: device to load model to
        
    Returns:
        epoch: epoch number when saved
        val_loss: validation loss when saved
        val_acc: validation accuracy when saved
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    val_acc = checkpoint['val_acc']
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {epoch}, val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%")
    return epoch, val_loss, val_acc


def train(config: Dict, model_type: str, resume_from: Optional[str] = None) -> None:
    """
    Complete training pipeline.
    
    This is the main training function that orchestrates:
    1. Data loading
    2. Model creation
    3. Training loop
    4. Validation
    5. Checkpointing
    6. Logging
    
    Args:
        config: configuration dictionary
        model_type: 'mlp', 'cnn1d', or 'pointnet'
        resume_from: optional checkpoint path to resume training
    """
    print("=" * 80)
    print(f"Training {model_type.upper()} model")
    print("=" * 80)
    
    # Set device (GPU if available, else CPU)
    # Check for MPS (Mac M1/M2 GPU) availability
    if config['device'] == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif config['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    log_dir = Path(config['logging']['log_dir'])
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = log_dir / 'checkpoints' / model_type
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize TensorBoard writer for logging
    writer = SummaryWriter(log_dir=str(log_dir / 'tensorboard' / model_type))
    
    # ========== Data Loading ==========
    print("\n" + "=" * 80)
    print("Loading dataset...")
    print("=" * 80)
    
    # Use grouped split
    processed_filename = 'faust_pc.npz'
    processed_path = Path(config['data']['processed_dir']) / processed_filename
    
    samples_per_mesh = config['data'].get('samples_per_mesh', 100)
    normalize_center = config['data'].get('normalize_center', True)
    normalize_scale = config['data'].get('normalize_scale', True)
    
    print(f"Split strategy: grouped")
    print(f"Processed data will be saved/loaded from: {processed_path}")
    
    if processed_path.exists():
        # Load preprocessed data
        print(f"Loading processed dataset from {processed_path}")
        data, labels, filenames, metadata = load_processed_dataset(str(processed_path))
        
        needs_reprocess = False

        if filenames is None:
            print("⚠️  Old format detected (missing filenames), reprocessing...")
            needs_reprocess = True
        elif metadata.get('normalized', True) != normalize_scale:
            print("⚠️  Dataset normalization mode mismatch. Reprocessing to match current config...")
            needs_reprocess = True
        elif metadata.get('samples_per_mesh') != samples_per_mesh:
            print("⚠️  samples_per_mesh mismatch. Reprocessing to match current config...")
            needs_reprocess = True

        if needs_reprocess:
            processed_path.unlink(missing_ok=True)
            data, labels, filenames = load_faust_dataset(
                config['data']['raw_dir'],
                num_points=config['data']['num_points'],
                samples_per_mesh=samples_per_mesh,
                use_fps=True,
                normalize_center=normalize_center,
                normalize_scale=normalize_scale
            )
            processed_path.parent.mkdir(parents=True, exist_ok=True)
            save_processed_dataset(
                data, labels, str(processed_path),
                filenames=filenames,
                normalized=normalize_scale,
                samples_per_mesh=samples_per_mesh
            )
    else:
        # Load and preprocess FAUST dataset
        print(f"Processing FAUST dataset from {config['data']['raw_dir']}")
        data, labels, filenames = load_faust_dataset(
            config['data']['raw_dir'],
            num_points=config['data']['num_points'],
            samples_per_mesh=samples_per_mesh,
            use_fps=True,
            normalize_center=normalize_center,
            normalize_scale=normalize_scale
        )
        # Save for future use
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        save_processed_dataset(
            data, labels, str(processed_path),
            filenames=filenames,
            normalized=normalize_scale,
            samples_per_mesh=samples_per_mesh
        )
    
    # Split dataset using grouped split
    # Samples from the same mesh stay together to avoid data leakage
    print("\nSplitting dataset (grouped by mesh)...")
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_split_grouped(
        data, labels, filenames, samples_per_mesh,
        train_ratio=config['split']['train_ratio'],
        val_ratio=config['split']['val_ratio'],
        test_ratio=config['split']['test_ratio'],
        random_seed=config['split']['seed']
    )
    
    # Create DataLoaders
    print("\nCreating DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=config['training']['batch_size'],
        num_workers=4,
        augment_train=True,  # Using augmentation with fixed normalization
        device=str(device),  # Pass device to control pin_memory
        rotation_range=config['augmentation']['rotation_range'],
        translation_range=config['augmentation']['translation_range'],
        normalize_center=normalize_center,
        normalize_scale=normalize_scale
    )
    
    # ========== Model Setup ==========
    print("\n" + "=" * 80)
    print("Creating model...")
    print("=" * 80)
    
    num_classes = len(np.unique(labels))  # Number of subjects
    model = create_model(model_type, num_classes, config)
    model = model.to(device)
    
    # ========== Training Setup ==========
    # Loss function: CrossEntropyLoss for multi-class classification
    # Combines LogSoftmax and NLLLoss
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: Adam with weight decay for regularization
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler: reduce LR when validation plateaus
    # With high initial LR (0.01), use more aggressive reduction
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # Minimize validation loss
        factor=0.7,  # Reduce LR by 50% when plateau
        patience=5,  # Wait 5 epochs before reducing
        min_lr=1e-6  # Don't go below this LR
    )
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        min_delta=0.001
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_acc = 0.0
    
    if resume_from:
        start_epoch, _, best_val_acc = load_checkpoint(
            model, optimizer, resume_from, device
        )
        start_epoch += 1  # Start from next epoch
    
    # ========== Training Loop ==========
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)
    
    num_epochs = config['training']['num_epochs']
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 80)
        
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = checkpoint_dir / 'model_best.pth'
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, str(save_path))
        
        # Save periodic checkpoint
        if (epoch + 1) % config['logging']['save_interval'] == 0:
            save_path = checkpoint_dir / f'model_epoch_{epoch+1}.pth'
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, str(save_path))
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"No improvement in validation loss for {early_stopping.patience} epochs")
            break
    
    # ========== Final Evaluation ==========
    print("\n" + "=" * 80)
    print("Training completed! Evaluating on test set...")
    print("=" * 80)
    
    # Load best model
    best_model_path = checkpoint_dir / 'model_best.pth'
    load_checkpoint(model, optimizer, str(best_model_path), device)
    
    # Test evaluation
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    print(f"\nFinal Test Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Acc: {test_acc:.2f}%")
    
    # Log test results
    writer.add_text('Test Results', 
                    f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    
    writer.close()
    print(f"\nTraining logs saved to {log_dir / 'tensorboard' / model_type}")
    print(f"Checkpoints saved to {checkpoint_dir}")


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description='Train point cloud classification models'
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
        default='pointnet',
        choices=['mlp', 'cnn1d', 'pointnet', 'mmidnet'],
        help='Model type to train'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Start training
    train(config, args.model, args.resume)


if __name__ == '__main__':
    main()


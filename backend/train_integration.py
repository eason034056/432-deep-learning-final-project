"""
Integration wrapper for existing training code

This module provides a bridge between the GUI backend and the existing
training scripts without heavily modifying the original code.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Callable, Dict
import yaml

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import existing training functions
from train import (
    load_config as load_train_config,
    create_model,
    train_one_epoch,
    validate,
    EarlyStopping,
    save_checkpoint,
    load_checkpoint
)

from dataset import (
    load_faust_dataset,
    stratified_split_grouped,
    create_dataloaders,
    save_processed_dataset,
    load_processed_dataset,
    FAUSTPointCloudDataset
)
from models import chamfer_distance, create_autoencoder_from_config, get_autoencoder_config

from evaluate import evaluate_model as eval_model_original

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


from backend.utils import get_project_root

def _is_autoencoder(model_type: str) -> bool:
    return model_type in ('mlp_ae', 'pointnet_ae')


def train_ae_model(model_type: str, config: dict, progress_callback=None) -> str:
    """Train autoencoder with progress callback for GUI."""
    ae_cfg = get_autoencoder_config(config)
    train_cfg = ae_cfg['train']
    project_root = Path(get_project_root())
    processed_path = project_root / config['data']['processed_dir'] / 'faust_pc.npz'
    raw_dir = project_root / config['data']['raw_dir']
    samples_per_mesh = config['data'].get('samples_per_mesh', 100)
    normalize_center = config['data'].get('normalize_center', True)
    normalize_scale = config['data'].get('normalize_scale', True)

    if processed_path.exists():
        data, labels, filenames, metadata = load_processed_dataset(str(processed_path))
        samples_per_mesh = metadata.get('samples_per_mesh', samples_per_mesh)
    else:
        data, labels, filenames = load_faust_dataset(
            str(raw_dir),
            num_points=config['data']['num_points'],
            samples_per_mesh=samples_per_mesh,
            use_fps=True,
            normalize_center=normalize_center,
            normalize_scale=normalize_scale
        )
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        save_processed_dataset(data, labels, str(processed_path),
                              filenames=filenames, normalized=normalize_scale,
                              samples_per_mesh=samples_per_mesh)

    X_train, y_train, X_val, y_val, X_test, y_test = stratified_split_grouped(
        data, labels, filenames, samples_per_mesh,
        train_ratio=config['split']['train_ratio'],
        val_ratio=config['split']['val_ratio'],
        test_ratio=config['split']['test_ratio'],
        random_seed=config['split']['seed']
    )

    train_dataset = FAUSTPointCloudDataset(X_train, y_train, augment=train_cfg['augment'],
        rotation_range=config['augmentation']['rotation_range'],
        translation_range=config['augmentation']['translation_range'])
    val_dataset = FAUSTPointCloudDataset(X_val, y_val, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=train_cfg['batch_size'],
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg['batch_size'],
                           shuffle=False, num_workers=2)

    if config.get('device') == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif config.get('device') == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = create_autoencoder_from_config(model_type, config).to(device)

    criterion = lambda pred, target: chamfer_distance(pred, target, reduce='mean')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=train_cfg['learning_rate'],
                                 weight_decay=train_cfg['weight_decay'])

    checkpoint_dir = project_root / config['logging']['log_dir'] / 'checkpoints' / model_type
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    num_epochs = train_cfg['num_epochs']
    best_val_loss = float('inf')

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for data, _ in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon, _ = model(data)
            loss = criterion(recon, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        model.eval()
        total_val = 0.0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                recon, _ = model(data)
                total_val += criterion(recon, data).item()
        val_loss = total_val / len(val_loader)

        if progress_callback:
            progress_callback(epoch, {'train_loss': train_loss, 'val_loss': val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, checkpoint_dir / 'model_best.pth')

    return str(checkpoint_dir / 'model_best.pth')


def evaluate_ae_model(model_type: str, checkpoint_path: str, config: dict) -> dict:
    """Evaluate autoencoder and return Chamfer Distance."""
    ae_cfg = get_autoencoder_config(config)
    project_root = Path(get_project_root())
    processed_path = project_root / config['data']['processed_dir'] / 'faust_pc.npz'
    data, labels, filenames, metadata = load_processed_dataset(str(processed_path))
    samples_per_mesh = metadata.get('samples_per_mesh', config['data'].get('samples_per_mesh', 100))

    _, _, _, _, X_test, y_test = stratified_split_grouped(
        data, labels, filenames, samples_per_mesh,
        train_ratio=config['split']['train_ratio'],
        val_ratio=config['split']['val_ratio'],
        test_ratio=config['split']['test_ratio'],
        random_seed=config['split']['seed']
    )

    test_dataset = FAUSTPointCloudDataset(X_test, y_test, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=ae_cfg['eval']['batch_size'], shuffle=False)

    if config.get('device') == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif config.get('device') == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = create_autoencoder_from_config(model_type, config).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    model.eval()
    total_cd = 0.0
    count = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, _ = model(data)
            total_cd += chamfer_distance(recon, data, reduce='sum').item()
            count += data.size(0)

    return {
        'chamfer_distance': total_cd / count,
        'accuracy': 0,
        'precision': 0,
        'recall': 0,
        'f1_score': 0
    }


def train_model(model_type: str, 
                config: dict,
                progress_callback: Optional[Callable] = None) -> str:
    """
    Train model with progress callback support
    
    This is a wrapper around the existing train() function that adds
    callback support for the GUI to track progress in real-time.
    
    Args:
        model_type: Model type ('mlp', 'cnn1d', or 'pointnet')
        config: Configuration dictionary
        progress_callback: Optional callback function(epoch, metrics)
        
    Returns:
        model_path: Path to saved model checkpoint
    """
    # Setup device
    if config['device'] == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif config['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Create output directories (Absolute paths)
    project_root = Path(get_project_root())
    log_dir = project_root / config['logging']['log_dir']
    checkpoint_dir = log_dir / 'checkpoints' / model_type
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load/process dataset (Absolute paths)
    processed_filename = 'faust_pc.npz'
    processed_dir = project_root / config['data']['processed_dir']
    processed_path = processed_dir / processed_filename
    raw_dir = project_root / config['data']['raw_dir']
    
    samples_per_mesh = config['data'].get('samples_per_mesh', 100)
    normalize_center = config['data'].get('normalize_center', False)
    normalize_scale = config['data'].get('normalize_scale', False)
    
    if processed_path.exists():
        data, labels, filenames, metadata = load_processed_dataset(str(processed_path))
        needs_reprocess = (
            filenames is None or
            metadata.get('normalized', True) != normalize_scale or
            metadata.get('samples_per_mesh') != samples_per_mesh
        )
        
        if needs_reprocess:
            processed_path.unlink(missing_ok=True)
            data, labels, filenames = load_faust_dataset(
                str(raw_dir),
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
        data, labels, filenames = load_faust_dataset(
            str(raw_dir),
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
    
    # Split dataset
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_split_grouped(
        data, labels, filenames, samples_per_mesh,
        train_ratio=config['split']['train_ratio'],
        val_ratio=config['split']['val_ratio'],
        test_ratio=config['split']['test_ratio'],
        random_seed=config['split']['seed']
    )
    
    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test,
        batch_size=config['training']['batch_size'],
        num_workers=2,  # Reduce for better GUI responsiveness
        augment_train=True,
        device=str(device),
        rotation_range=config['augmentation']['rotation_range'],
        translation_range=config['augmentation']['translation_range'],
        normalize_center=normalize_center,
        normalize_scale=normalize_scale
    )
    
    # Create model
    num_classes = len(np.unique(labels))
    model = create_model(model_type, num_classes, config)
    model = model.to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=5, min_lr=1e-6
    )
    
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping_patience'],
        min_delta=0.001
    )
    
    # Training loop with callbacks
    num_epochs = config['training']['num_epochs']
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch+1
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Call progress callback
        if progress_callback:
            metrics = {
                'train_loss': train_loss,
                'train_acc': train_acc / 100.0,  # Normalize to 0-1
                'val_loss': val_loss,
                'val_acc': val_acc / 100.0  # Normalize to 0-1
            }
            progress_callback(epoch + 1, metrics)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = checkpoint_dir / 'model_best.pth'
            save_checkpoint(model, optimizer, epoch, val_loss, val_acc, str(save_path))
        
        # Early stopping
        if early_stopping(val_loss):
            break
    
    # Return path to best model
    return str(checkpoint_dir / 'model_best.pth')


def evaluate_model(model_type: str, 
                   checkpoint_path: str,
                   config: dict) -> Dict:
    """
    Evaluate a trained model and return structured results
    
    Args:
        model_type: Model type
        checkpoint_path: Path to checkpoint file
        config: Configuration dictionary
        
    Returns:
        results: Dictionary with evaluation metrics
    """
    # Setup device
    if config['device'] == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif config['device'] == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Load dataset (Absolute paths)
    project_root = Path(get_project_root())
    processed_filename = 'faust_pc.npz'
    processed_path = project_root / config['data']['processed_dir'] / processed_filename
    
    data, labels, filenames, metadata = load_processed_dataset(str(processed_path))
    
    samples_per_mesh = config['data'].get('samples_per_mesh', 100)
    
    # Split dataset
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_split_grouped(
        data, labels, filenames, samples_per_mesh,
        train_ratio=config['split']['train_ratio'],
        val_ratio=config['split']['val_ratio'],
        test_ratio=config['split']['test_ratio'],
        random_seed=config['split']['seed']
    )
    
    # Create test DataLoader
    batch_size = config['training']['batch_size']
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Load model
    num_classes = len(np.unique(labels))
    model = create_model(model_type, num_classes, config)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Evaluate
    y_true, y_pred, y_prob = eval_model_original(model, test_loader, str(device))
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'y_true': y_true.tolist(),
        'y_pred': y_pred.tolist()
    }


def preprocess_faust_dataset(config: dict) -> None:
    """
    Preprocess FAUST dataset according to config
    
    Args:
        config: Configuration dictionary
    """
    project_root = Path(get_project_root())
    processed_filename = 'faust_pc.npz'
    processed_path = project_root / config['data']['processed_dir'] / processed_filename
    raw_dir = project_root / config['data']['raw_dir']
    
    samples_per_mesh = config['data'].get('samples_per_mesh', 100)
    normalize_center = config['data'].get('normalize_center', False)
    normalize_scale = config['data'].get('normalize_scale', False)
    
    # Always reprocess when explicitly called
    if processed_path.exists():
        processed_path.unlink()
    
    data, labels, filenames = load_faust_dataset(
        str(raw_dir),
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

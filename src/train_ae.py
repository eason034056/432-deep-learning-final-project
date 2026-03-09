"""
Training script for Point Cloud Autoencoder models.

Trains MLP-based and PointNet-based autoencoders for the compression task.
Uses Chamfer Distance as the reconstruction loss.

Usage:
    python train_ae.py --config config.yaml --model mlp_ae
    python train_ae.py --config config.yaml --model pointnet_ae
"""

# First thing: ensure we produce output (diagnose "no output" on HPC)
import sys
sys.stdout.write("train_ae: script starting\n")
sys.stdout.flush()

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import (
    load_processed_dataset,
    load_faust_dataset,
    stratified_split_grouped,
    save_processed_dataset,
)
from models import MLPAutoencoder, PointNetAutoencoder, chamfer_distance


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_ae_model(model_type: str, config: Dict) -> nn.Module:
    num_points = config['data']['num_points']
    latent_dim = config.get('autoencoder', {}).get('latent_dim', 128)
    dropout = config['model'].get('dropout', 0.1)
    
    if model_type == 'mlp_ae':
        print("Creating MLP Autoencoder...", flush=True)
        model = MLPAutoencoder(
            num_points=num_points,
            num_channels=3,
            latent_dim=latent_dim,
            hidden_dims=(256, 128),
            dropout=dropout
        )
    elif model_type == 'pointnet_ae':
        print("Creating PointNet Autoencoder...", flush=True)
        model = PointNetAutoencoder(
            num_points=num_points,
            num_channels=3,
            latent_dim=128,
            dropout=0.5,
            use_tnet=False,
            channel_dims=(64, 128, 256)
        )
    else:
        raise ValueError(f"Unknown model: {model_type}. Use mlp_ae or pointnet_ae")
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}", flush=True)
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for data, _ in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        recon, _ = model(data)
        loss = criterion(recon, data)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for data, _ in val_loader:
            data = data.to(device)
            recon, _ = model(data)
            loss = criterion(recon, data)
            total_loss += loss.item()
    return total_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--model', choices=['mlp_ae', 'pointnet_ae'], default='mlp_ae')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use (e.g. 3). If not set, uses default cuda:0. Use when other GPUs are busy.')
    args = parser.parse_args()
    
    print("train_ae: Starting...", flush=True)
    config = load_config(args.config)
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}' if args.gpu is not None else 'cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}", flush=True)
    
    processed_path = Path(config['data']['processed_dir']) / 'faust_pc.npz'
    samples_per_mesh = config['data'].get('samples_per_mesh', 100)
    
    if processed_path.exists():
        print("Loading processed dataset...", flush=True)
        data, labels, filenames, metadata = load_processed_dataset(str(processed_path))
        samples_per_mesh = metadata.get('samples_per_mesh', samples_per_mesh)
    else:
        print("Loading raw FAUST dataset (this may take a while)...", flush=True)
        data, labels, filenames = load_faust_dataset(
            config['data']['raw_dir'],
            num_points=config['data']['num_points'],
            samples_per_mesh=samples_per_mesh,
            use_fps=True,
            normalize_center=True,
            normalize_scale=True
        )
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        save_processed_dataset(data, labels, str(processed_path), filenames=filenames,
                              normalized=True, samples_per_mesh=samples_per_mesh)
    
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_split_grouped(
        data, labels, filenames, samples_per_mesh,
        train_ratio=config['split']['train_ratio'],
        val_ratio=config['split']['val_ratio'],
        test_ratio=config['split']['test_ratio'],
        random_seed=config['split']['seed']
    )
    
    from dataset import FAUSTPointCloudDataset
    train_dataset = FAUSTPointCloudDataset(X_train, y_train, augment=True,
                                          rotation_range=config['augmentation']['rotation_range'],
                                          translation_range=config['augmentation']['translation_range'])
    val_dataset = FAUSTPointCloudDataset(X_val, y_val, augment=False)
    
    # num_workers=0 avoids multiprocessing hangs on HPC/NFS; use 4 for faster local training
    num_workers = 0
    print(f"Creating DataLoaders (num_workers={num_workers})...", flush=True)
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                             shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                           shuffle=False, num_workers=num_workers)
    
    print(f"Creating {args.model} model...", flush=True)
    model = create_ae_model(args.model, config).to(device)
    criterion = lambda pred, target: chamfer_distance(pred, target, reduce='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'],
                                 weight_decay=config['training']['weight_decay'])
    
    log_dir = Path(config['logging']['log_dir']) / 'tensorboard' / args.model
    ckpt_dir = Path(config['logging']['log_dir']) / 'checkpoints' / args.model
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = validate(model, val_loader, criterion, device)
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}", flush=True)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss
            }, ckpt_dir / 'model_best.pth')
            print(f"  Saved best model (val_loss={val_loss:.6f})", flush=True)
    
    writer.close()
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.6f}")
    print(f"Checkpoints: {ckpt_dir}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        import traceback
        sys.stderr.write(f"train_ae ERROR: {e}\n")
        traceback.print_exc()
        sys.exit(1)

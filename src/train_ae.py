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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import (
    FAUSTPointCloudDataset,
    load_processed_dataset,
    load_faust_dataset,
    stratified_split_grouped,
    save_processed_dataset,
)
from models import chamfer_distance, create_autoencoder_from_config, get_autoencoder_config


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_ae_model(model_type: str, config: Dict) -> nn.Module:
    ae_cfg = get_autoencoder_config(config)

    if model_type == 'mlp_ae':
        print("Creating MLP Autoencoder...", flush=True)
    elif model_type == 'pointnet_ae':
        print("Creating PointNet Autoencoder...", flush=True)
    else:
        raise ValueError(f"Unknown model: {model_type}. Use mlp_ae or pointnet_ae")

    model = create_autoencoder_from_config(model_type, config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}", flush=True)
    print(f"Autoencoder config: {ae_cfg[model_type.replace('_ae', '')]}", flush=True)
    return model


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def should_reprocess_dataset(
    filenames,
    metadata: Dict,
    num_points: int,
    samples_per_mesh: int,
    normalize_center: bool,
    normalize_scale: bool,
) -> bool:
    """Check whether cached processed data matches the active config."""
    if filenames is None:
        print("  Reprocessing: cached dataset is missing filenames metadata.", flush=True)
        return True

    if metadata.get('num_points') != num_points:
        print("  Reprocessing: cached dataset num_points does not match config.", flush=True)
        return True

    if metadata.get('samples_per_mesh') != samples_per_mesh:
        print("  Reprocessing: cached dataset samples_per_mesh does not match config.", flush=True)
        return True

    if metadata.get('normalize_center') != normalize_center:
        print("  Reprocessing: cached dataset normalize_center does not match config.", flush=True)
        return True

    if metadata.get('normalize_scale') != normalize_scale:
        print("  Reprocessing: cached dataset normalize_scale does not match config.", flush=True)
        return True

    return False


def build_processed_dataset(
    config: Dict,
    processed_path: Path,
    num_points: int,
    samples_per_mesh: int,
    normalize_center: bool,
    normalize_scale: bool,
):
    """Load cached dataset if compatible, otherwise rebuild it."""
    if processed_path.exists():
        print("Loading processed dataset...", flush=True)
        data, labels, filenames, metadata = load_processed_dataset(str(processed_path))
        if not should_reprocess_dataset(
            filenames=filenames,
            metadata=metadata,
            num_points=num_points,
            samples_per_mesh=samples_per_mesh,
            normalize_center=normalize_center,
            normalize_scale=normalize_scale,
        ):
            return data, labels, filenames, metadata
        processed_path.unlink(missing_ok=True)

    print("Loading raw FAUST dataset (this may take a while)...", flush=True)
    data, labels, filenames = load_faust_dataset(
        config['data']['raw_dir'],
        num_points=num_points,
        samples_per_mesh=samples_per_mesh,
        use_fps=True,
        normalize_center=normalize_center,
        normalize_scale=normalize_scale
    )
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    save_processed_dataset(
        data,
        labels,
        str(processed_path),
        filenames=filenames,
        normalized=normalize_scale,
        samples_per_mesh=samples_per_mesh,
        normalize_center=normalize_center,
        normalize_scale=normalize_scale,
        num_points=num_points,
    )
    metadata = {
        'normalized': normalize_scale,
        'normalize_center': normalize_center,
        'normalize_scale': normalize_scale,
        'num_points': num_points,
        'samples_per_mesh': samples_per_mesh,
    }
    return data, labels, filenames, metadata


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
    parser.add_argument('--epochs', type=int, default=None,
                        help='Optional override for autoencoder.train.num_epochs')
    parser.add_argument(
        '--overfit-samples',
        type=int,
        default=0,
        help='If > 0, train on only the first N training samples and reuse them for validation.'
    )
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use (e.g. 3). If not set, uses default cuda:0. Use when other GPUs are busy.')
    args = parser.parse_args()
    
    print("train_ae: Starting...", flush=True)
    config = load_config(args.config)
    ae_cfg = get_autoencoder_config(config)
    ae_train_cfg = ae_cfg['train']
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}' if args.gpu is not None else 'cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}", flush=True)
    
    processed_path = Path(config['data']['processed_dir']) / 'faust_pc.npz'
    num_points = config['data']['num_points']
    samples_per_mesh = config['data'].get('samples_per_mesh', 100)
    normalize_center = config['data'].get('normalize_center', True)
    normalize_scale = config['data'].get('normalize_scale', True)
    data, labels, filenames, metadata = build_processed_dataset(
        config=config,
        processed_path=processed_path,
        num_points=num_points,
        samples_per_mesh=samples_per_mesh,
        normalize_center=normalize_center,
        normalize_scale=normalize_scale,
    )
    samples_per_mesh = metadata.get('samples_per_mesh', samples_per_mesh)
    
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_split_grouped(
        data, labels, filenames, samples_per_mesh,
        train_ratio=config['split']['train_ratio'],
        val_ratio=config['split']['val_ratio'],
        test_ratio=config['split']['test_ratio'],
        random_seed=config['split']['seed']
    )

    if args.overfit_samples > 0:
        overfit_n = min(args.overfit_samples, len(X_train))
        if overfit_n <= 0:
            raise ValueError("--overfit-samples must be positive")
        print(
            f"Overfit debug mode: using first {overfit_n} training samples for both train and validation.",
            flush=True
        )
        print(
            "Recommendation: use low dropout and near-zero weight decay to test memorization capacity.",
            flush=True
        )
        X_train = X_train[:overfit_n]
        y_train = y_train[:overfit_n]
        X_val = X_train.copy()
        y_val = y_train.copy()
    
    train_dataset = FAUSTPointCloudDataset(
        X_train,
        y_train,
        augment=ae_train_cfg['augment'] and args.overfit_samples <= 0,
        rotation_range=config['augmentation']['rotation_range'],
        translation_range=config['augmentation']['translation_range'],
        normalize_center=normalize_center,
        normalize_scale=normalize_scale,
    )
    val_dataset = FAUSTPointCloudDataset(
        X_val,
        y_val,
        augment=False,
        normalize_center=normalize_center,
        normalize_scale=normalize_scale,
    )
    
    # num_workers=0 avoids multiprocessing hangs on HPC/NFS; use 4 for faster local training
    num_workers = 0
    print(f"Creating DataLoaders (num_workers={num_workers})...", flush=True)
    train_loader = DataLoader(train_dataset, batch_size=ae_train_cfg['batch_size'],
                             shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=ae_train_cfg['batch_size'],
                           shuffle=False, num_workers=num_workers)
    
    print(f"Creating {args.model} model...", flush=True)
    model = create_ae_model(args.model, config).to(device)
    criterion = lambda pred, target: chamfer_distance(pred, target, reduce='mean')
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=ae_train_cfg['learning_rate'],
        weight_decay=ae_train_cfg['weight_decay']
    )
    
    run_suffix = f"overfit_{args.overfit_samples}" if args.overfit_samples > 0 else None
    log_dir = Path(config['logging']['log_dir']) / 'tensorboard' / args.model
    ckpt_dir = Path(config['logging']['log_dir']) / 'checkpoints' / args.model
    if run_suffix is not None:
        log_dir = log_dir / run_suffix
        ckpt_dir = ckpt_dir / run_suffix
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    
    num_epochs = args.epochs if args.epochs is not None else ae_train_cfg['num_epochs']
    early_cfg = ae_train_cfg['early_stopping']
    early_stopping = None
    if early_cfg['enabled']:
        early_stopping = EarlyStopping(
            patience=early_cfg['patience'],
            min_delta=early_cfg['min_delta']
        )

    best_val_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
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

        if early_stopping is not None and early_stopping.step(val_loss):
            print(
                f"  Early stopping triggered at epoch {epoch} "
                f"(patience={early_cfg['patience']}, min_delta={early_cfg['min_delta']})",
                flush=True
            )
            break
    
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

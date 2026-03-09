"""
Evaluation script for Point Cloud Autoencoder models.

Computes Chamfer Distance, generates reconstruction visualizations,
and compares MLP vs PointNet++ autoencoders.

Usage:
    python evaluate_ae.py --config config.yaml --model pointnet2_ae
    python evaluate_ae.py --config config.yaml --compare
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import load_processed_dataset, stratified_split_grouped, FAUSTPointCloudDataset
from models import MLPAutoencoder, PointNet2Autoencoder, chamfer_distance


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_ae_model(model_type: str, config: Dict) -> torch.nn.Module:
    num_points = config['data']['num_points']
    dropout = config['model'].get('dropout', 0.1)
    
    if model_type == 'mlp_ae':
        return MLPAutoencoder(num_points=num_points, num_channels=3, latent_dim=128,
                              hidden_dims=(256, 128), dropout=dropout)
    elif model_type == 'pointnet2_ae':
        return PointNet2Autoencoder(num_points=num_points, num_channels=3, latent_dim=1024,
                                   dropout=dropout, use_xyz=True)
    else:
        raise ValueError(f"Unknown model: {model_type}")


def evaluate_ae(model, test_loader, device) -> tuple:
    """Returns (avg_chamfer_distance, all_recon, all_original)"""
    model.eval()
    total_cd = 0.0
    count = 0
    all_recon = []
    all_orig = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            recon, _ = model(data)
            cd = chamfer_distance(recon, data, reduce='sum')
            total_cd += cd.item()
            count += data.size(0)
            all_recon.append(recon.cpu().numpy())
            all_orig.append(data.cpu().numpy())
    
    avg_cd = total_cd / count
    all_recon = np.concatenate(all_recon, axis=0)
    all_orig = np.concatenate(all_orig, axis=0)
    return avg_cd, all_recon, all_orig


def plot_reconstruction(original: np.ndarray, reconstructed: np.ndarray, save_path: str, title: str = ""):
    """Plot original vs reconstructed point clouds side by side."""
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2], s=5, alpha=0.6, c='blue')
    ax1.set_title('Original')
    
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    ax2.scatter(reconstructed[:, 0], reconstructed[:, 1], reconstructed[:, 2], s=5, alpha=0.6, c='green')
    ax2.set_title('Reconstructed')
    
    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--model', choices=['mlp_ae', 'pointnet2_ae'], help='Single model mode')
    parser.add_argument('--checkpoint', help='Path to checkpoint (single model)')
    parser.add_argument('--compare', action='store_true', help='Compare both models')
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
    
    processed_path = Path(config['data']['processed_dir']) / 'faust_pc.npz'
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
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    results_dir = Path(config['logging']['log_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if args.compare:
        print("=" * 60)
        print("Autoencoder Comparison: MLP vs PointNet++")
        print("=" * 60)
        
        results = []
        for model_name in ['mlp_ae', 'pointnet2_ae']:
            ckpt_path = results_dir / 'checkpoints' / model_name / 'model_best.pth'
            if not ckpt_path.exists():
                print(f"  Skip {model_name}: checkpoint not found")
                continue
            
            model = create_ae_model(model_name, config).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            
            avg_cd, all_recon, all_orig = evaluate_ae(model, test_loader, device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            results.append({
                'model': model_name,
                'chamfer_distance': avg_cd,
                'parameters': n_params
            })
            print(f"\n{model_name}:")
            print(f"  Chamfer Distance: {avg_cd:.6f}")
            print(f"  Parameters: {n_params:,}")
            
            # Save sample reconstruction
            idx = 0
            plot_reconstruction(all_orig[idx], all_recon[idx],
                               str(results_dir / f'ae_reconstruction_{model_name}.png'),
                               f'{model_name} - CD={avg_cd:.4f}')
        
        if results:
            print("\n" + "-" * 60)
            best = min(results, key=lambda x: x['chamfer_distance'])
            print(f"Best: {best['model']} (CD={best['chamfer_distance']:.6f})")
    
    else:
        if not args.model or not args.checkpoint:
            parser.error("--model and --checkpoint required for single model evaluation")
        
        model = create_ae_model(args.model, config).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        
        avg_cd, all_recon, all_orig = evaluate_ae(model, test_loader, device)
        print(f"\n{args.model}: Chamfer Distance = {avg_cd:.6f}")
        
        plot_reconstruction(all_orig[0], all_recon[0],
                           str(results_dir / f'ae_reconstruction_{args.model}.png'),
                           f'{args.model} - CD={avg_cd:.4f}')
        print(f"Saved reconstruction plot to {results_dir / f'ae_reconstruction_{args.model}.png'}")


if __name__ == '__main__':
    main()

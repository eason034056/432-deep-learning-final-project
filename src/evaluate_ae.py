"""
Evaluation script for Point Cloud Autoencoder models.

Computes Chamfer Distance, inference time, and generates reconstruction visualizations.
Exports comparison table (CSV + Markdown) and multi-sample plots.

Usage:
    python evaluate_ae.py --config config.yaml --model mlp_ae --checkpoint results/checkpoints/mlp_ae/model_best.pth
    python evaluate_ae.py --config config.yaml --model pointnet_ae --checkpoint results/checkpoints/pointnet_ae/model_best.pth
    python evaluate_ae.py --config config.yaml --compare [--num_samples 4]
"""

import os
import sys
import argparse
import time
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import load_processed_dataset, stratified_split_grouped, FAUSTPointCloudDataset
from models import chamfer_distance, create_autoencoder_from_config, get_autoencoder_config


def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_ae_model(model_type: str, config: Dict) -> torch.nn.Module:
    return create_autoencoder_from_config(model_type, config)


def evaluate_ae(model, test_loader, device) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """Returns (avg_chamfer_distance, all_recon, all_original, inference_time_sec)."""
    model.eval()
    total_cd = 0.0
    count = 0
    all_recon = []
    all_orig = []
    total_inference_time = 0.0

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps' and hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
            t0 = time.perf_counter()
            recon, _ = model(data)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'mps' and hasattr(torch.mps, 'synchronize'):
                torch.mps.synchronize()
            total_inference_time += time.perf_counter() - t0

            cd = chamfer_distance(recon, data, reduce='sum')
            total_cd += cd.item()
            count += data.size(0)
            all_recon.append(recon.cpu().numpy())
            all_orig.append(data.cpu().numpy())

    avg_cd = total_cd / count
    all_recon = np.concatenate(all_recon, axis=0)
    all_orig = np.concatenate(all_orig, axis=0)
    return avg_cd, all_recon, all_orig, total_inference_time


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


def save_comparison_table(results: List[Dict], results_dir: Path) -> None:
    """Save comparison results to CSV and Markdown table."""
    if not results:
        return

    # CSV
    csv_path = results_dir / 'ae_comparison.csv'
    with open(csv_path, 'w') as f:
        f.write('model,chamfer_distance,parameters,inference_time_sec,inference_ms_per_sample\n')
        for r in results:
            n_samples = r.get('n_samples', 1)
            ms_per = (r['inference_time_sec'] / n_samples * 1000) if n_samples else 0
            f.write(f"{r['model']},{r['chamfer_distance']:.6f},{r['parameters']},{r['inference_time_sec']:.4f},{ms_per:.2f}\n")
    print(f"Saved comparison table to {csv_path}")

    # Markdown
    md_path = results_dir / 'ae_comparison.md'
    with open(md_path, 'w') as f:
        f.write('# Autoencoder Comparison\n\n')
        f.write('| Model | Chamfer Distance | Parameters | Inference (s) | ms/sample |\n')
        f.write('|-------|------------------|------------|---------------|----------|\n')
        for r in results:
            n_samples = r.get('n_samples', 1)
            ms_per = (r['inference_time_sec'] / n_samples * 1000) if n_samples else 0
            f.write(f"| {r['model']} | {r['chamfer_distance']:.6f} | {r['parameters']:,} | "
                    f"{r['inference_time_sec']:.4f} | {ms_per:.2f} |\n")
    print(f"Saved comparison table to {md_path}")


def plot_multi_sample_reconstruction(
    all_orig: np.ndarray,
    recon_dict: Dict[str, np.ndarray],
    save_path: str,
    num_samples: int = 4,
    title: str = "Multi-Sample Reconstruction Comparison",
) -> None:
    """
    Plot multiple samples: each row = one sample, columns = Original | Model1 | Model2 | ...
    recon_dict: {model_name: all_recon array}
    """
    num_samples = min(num_samples, all_orig.shape[0])
    model_names = list(recon_dict.keys())
    n_cols = 1 + len(model_names)
    fig = plt.figure(figsize=(5 * n_cols, 5 * num_samples))

    for row, idx in enumerate(range(num_samples)):
        for col, name in enumerate(['Original'] + model_names):
            ax = fig.add_subplot(num_samples, n_cols, row * n_cols + col + 1, projection='3d')
            if col == 0:
                pts = all_orig[idx]
                color = 'blue'
            else:
                pts = recon_dict[model_names[col - 1]][idx]
                color = 'green'
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5, alpha=0.6, c=color)
            ax.set_title(f'Sample {idx + 1}: {name}')
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_zlim(-1.2, 1.2)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved multi-sample plot to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--model', choices=['mlp_ae', 'pointnet_ae'], help='Single model mode')
    parser.add_argument('--checkpoint', help='Path to checkpoint (single model)')
    parser.add_argument('--compare', action='store_true', help='Compare both models')
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Optional override for autoencoder.eval.num_samples'
    )
    args = parser.parse_args()
    
    config = load_config(args.config)
    ae_cfg = get_autoencoder_config(config)
    ae_eval_cfg = ae_cfg['eval']
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
    test_loader = DataLoader(test_dataset, batch_size=ae_eval_cfg['batch_size'], shuffle=False)
    num_samples = args.num_samples if args.num_samples is not None else ae_eval_cfg['num_samples']
    
    results_dir = Path(config['logging']['log_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if args.compare:
        print("=" * 60)
        print("Autoencoder Comparison: MLP vs PointNet")
        print("=" * 60)

        results = []
        recon_dict = {}
        all_orig_shared = None

        for model_name in ['mlp_ae', 'pointnet_ae']:
            ckpt_path = results_dir / 'checkpoints' / model_name / 'model_best.pth'
            if not ckpt_path.exists():
                print(f"  Skip {model_name}: checkpoint not found")
                continue

            model = create_ae_model(model_name, config).to(device)
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])

            avg_cd, all_recon, all_orig, inference_time = evaluate_ae(model, test_loader, device)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_samples = all_orig.shape[0]

            if all_orig_shared is None:
                all_orig_shared = all_orig

            results.append({
                'model': model_name,
                'chamfer_distance': avg_cd,
                'parameters': n_params,
                'inference_time_sec': inference_time,
                'n_samples': n_samples,
            })
            recon_dict[model_name] = all_recon

            print(f"\n{model_name}:")
            print(f"  Chamfer Distance: {avg_cd:.6f}")
            print(f"  Parameters: {n_params:,}")
            print(f"  Inference Time: {inference_time:.4f}s ({inference_time / n_samples * 1000:.2f} ms/sample)")

            # Single-sample reconstruction (per model)
            plot_reconstruction(all_orig[0], all_recon[0],
                               str(results_dir / f'ae_reconstruction_{model_name}.png'),
                               f'{model_name} - CD={avg_cd:.4f}')

        if results:
            # Comparison table (CSV + Markdown)
            save_comparison_table(results, results_dir)

            # Multi-sample visualization
            if recon_dict and all_orig_shared is not None:
                plot_multi_sample_reconstruction(
                    all_orig_shared,
                    recon_dict,
                    str(results_dir / 'ae_reconstruction_multi_sample.png'),
                    num_samples=num_samples,
                    title="Multi-Sample Reconstruction: Original vs MLP AE vs PointNet AE",
                )

            print("\n" + "-" * 60)
            best = min(results, key=lambda x: x['chamfer_distance'])
            print(f"Best: {best['model']} (CD={best['chamfer_distance']:.6f})")
    
    else:
        if not args.model or not args.checkpoint:
            parser.error("--model and --checkpoint required for single model evaluation")

        model = create_ae_model(args.model, config).to(device)
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])

        avg_cd, all_recon, all_orig, inference_time = evaluate_ae(model, test_loader, device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_samples = all_orig.shape[0]

        print(f"\n{args.model}: Chamfer Distance = {avg_cd:.6f}")
        print(f"  Parameters: {n_params:,}")
        print(f"  Inference Time: {inference_time:.4f}s ({inference_time / n_samples * 1000:.2f} ms/sample)")

        plot_reconstruction(all_orig[0], all_recon[0],
                           str(results_dir / f'ae_reconstruction_{args.model}.png'),
                           f'{args.model} - CD={avg_cd:.4f}')
        print(f"Saved reconstruction plot to {results_dir / f'ae_reconstruction_{args.model}.png'}")

        # Multi-sample visualization (single model: Original vs Reconstructed)
        plot_multi_sample_reconstruction(
            all_orig,
            {args.model: all_recon},
            str(results_dir / f'ae_reconstruction_multi_sample_{args.model}.png'),
            num_samples=num_samples,
            title=f"Multi-Sample Reconstruction: {args.model} (CD={avg_cd:.4f})",
        )


if __name__ == '__main__':
    main()

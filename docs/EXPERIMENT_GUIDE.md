# Experiment Guide

## Classification Experiments

- **Centering**: Toggle `normalize_center` in config.yaml
- **Models**: mlp, cnn1d, pointnet2
- **Run**: `bash run_experiments.sh` (if available) or train each model separately

## Autoencoder Experiments

- **Models**: mlp_ae, pointnet2_ae
- **Loss**: Chamfer Distance
- **Run**: `python src/train_ae.py --model pointnet2_ae --epochs 100`

## Analysis

- `python analyze_results.py` — summarizes experiment results
- TensorBoard: `tensorboard --logdir results/tensorboard`

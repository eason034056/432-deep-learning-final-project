# Experiment Guide

## Goals
This repository currently supports two experiment families:

1. Classification comparison across `mlp`, `cnn1d`, and `pointnet`
2. Autoencoder comparison across `mlp_ae` and `pointnet_ae`

It also includes a focused ablation on point-cloud centering.

## Classification Comparison
Train or compare the implemented classifiers:

```bash
python src/train.py --config config.yaml --model mlp
python src/train.py --config config.yaml --model cnn1d
python src/train.py --config config.yaml --model pointnet
python src/evaluate.py --compare --models mlp cnn1d pointnet
```

Current result summary from `results/model_comparison.csv`:

- `pointnet`: 72.00% test accuracy, 0.9600 ROC-AUC
- `cnn1d`: 34.80% test accuracy, 0.7851 ROC-AUC
- `mlp`: 10.85% test accuracy, 0.5296 ROC-AUC

## Autoencoder Comparison
Train and compare the implemented autoencoders:

```bash
python src/train_ae.py --config config.yaml --model mlp_ae
python src/train_ae.py --config config.yaml --model pointnet_ae
python src/evaluate_ae.py --config config.yaml --compare
```

Current result summary from `results/ae_comparison.md`:

- `pointnet_ae`: CD mean 0.004430
- `mlp_ae`: CD mean 0.015800

## Centering Ablation
The main ablation in this repository tests whether centering helps each classifier:

```bash
bash run_experiments.sh
python analyze_results.py
```

The experiment toggles:

- `normalize_center = true`
- `normalize_center = false`

Applied to:

- `mlp`
- `cnn1d`
- `pointnet`

Key finding from `results/experiments/summary_report.txt`:

- PointNet improves by 17.18 percentage points with centering.
- CNN1D changes only slightly.
- MLP performs slightly better without centering in this setup.

## Practical Advice
- Rebuild the processed dataset if you change `num_points`, centering, or scaling.
- Use grouped splits to avoid leakage across augmented samples from the same mesh.
- Keep `results/` tables as the source of truth when updating report or presentation materials.

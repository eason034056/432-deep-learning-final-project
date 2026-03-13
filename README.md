# Point Cloud Human Identification Platform

PyTorch-based coursework project for two related tasks on 3D human point clouds:

1. Classification: identify one of 10 subjects from a point cloud.
2. Autoencoding: compress and reconstruct point clouds with Chamfer Distance.

The repository now includes both a command-line workflow and a Flask + browser GUI for preprocessing, training, monitoring, evaluation, and artifact download.

Course: MLDS 432 Deep Learning
Institution: Northwestern University

## Overview
This project uses the FAUST body-mesh dataset as a proxy point-cloud benchmark for privacy-preserving human sensing. The motivating application is mmWave-style sparse 3D sensing, but the actual training data in this repository is FAUST mesh data converted into point clouds.

That distinction matters:

- The repository does not contain real mmWave radar captures.
- The repository does contain a full end-to-end point-cloud ML pipeline that can be reused for real radar-style sparse 3D data later.
- All metrics reported below come from the current FAUST-based experiments in `results/`.

## Current Scope
Implemented classification models:

- `mlp`
- `cnn1d`
- `pointnet`

Implemented autoencoder models:

- `mlp_ae`
- `pointnet_ae`

Platform deliverables:

- CLI training and evaluation scripts
- GUI for upload, preprocessing, training, and downloads
- Docker / Docker Compose workflow
- Experiment runner for centering ablation
- Result summaries under `results/`

## Repository Highlights
Core ML pipeline:

- `src/dataset.py`: FAUST loading, processed-cache handling, grouped splitting, dataloaders
- `src/train.py`: classification training
- `src/evaluate.py`: classification evaluation and model comparison
- `src/train_ae.py`: autoencoder training
- `src/evaluate_ae.py`: autoencoder evaluation and comparison
- `src/models/`: MLP, CNN1D, PointNet Tiny, and autoencoder implementations

Application layer:

- `backend/app.py`: Flask API
- `backend/training_manager.py`: training jobs, persisted run state, and report generation
- `backend/train_integration.py`: bridge from GUI to training scripts
- `frontend/index.html`: browser UI
- `frontend/app.js`: client-side workflow and chart updates

Supporting material:

- `report/FINAL_REPORT.md`
- `presentation/presentation.md`
- `docs/GUI_GUIDE.md`
- `docs/EXPERIMENT_GUIDE.md`
- `docs/MODEL_OPS.md`

## Quick Start
### Option 1: GUI
Requirements:

- Docker
- Docker Compose
- FAUST mesh files placed in `data/raw/`

Run:

```bash
bash start.sh
```

Then open [http://localhost:8080](http://localhost:8080).

The GUI supports:

- uploading mesh files
- preprocessing into sampled point clouds
- switching between classification and autoencoder tasks
- real-time training monitoring
- evaluation report generation
- checkpoint and report download

### Option 2: Command Line
Create an environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Train classification models:

```bash
python src/train.py --config config.yaml --model mlp
python src/train.py --config config.yaml --model cnn1d
python src/train.py --config config.yaml --model pointnet
```

Train autoencoders:

```bash
python src/train_ae.py --config config.yaml --model mlp_ae
python src/train_ae.py --config config.yaml --model pointnet_ae
```

Evaluate:

```bash
python src/evaluate.py --compare --models mlp cnn1d pointnet
python src/evaluate_ae.py --config config.yaml --compare
```

Run centering ablation:

```bash
bash run_experiments.sh
python analyze_results.py
```

TensorBoard:

```bash
tensorboard --logdir results/tensorboard
```

## Dataset
This repository uses the FAUST training split:

- 100 meshes total
- 10 subjects x 10 poses
- each mesh sampled into point clouds
- default processed cache at `data/processed/faust_pc.npz`

Current preprocessing settings from `config.yaml`:

```yaml
data:
  num_points: 500
  samples_per_mesh: 100
  normalize_center: true
  normalize_scale: false
  raw_dir: data/raw
  processed_dir: data/processed

augmentation:
  normalize: false
  rotation_range: 360
  translation_range: 0.0
```

The pipeline performs:

1. mesh loading from `data/raw/`
2. farthest point sampling
3. augmentation across `samples_per_mesh`
4. optional centering / scaling based on config
5. grouped train/val/test split to reduce leakage across augmented samples from the same source mesh

## Models
### Classification
`mlp`

- flatten-based baseline
- fastest to train
- order-dependent

`cnn1d`

- Conv1D classifier over ordered points
- stronger than MLP baseline
- still sensitive to representation choices

`pointnet`

- PointNet-style model with T-Net
- permutation-aware design for point-cloud input
- strongest classifier in the current experiments

### Autoencoding
`mlp_ae`

- flatten-based encoder/decoder
- simple baseline for reconstruction

`pointnet_ae`

- PointNet-style encoder with point-cloud-aware reconstruction path
- strongest reconstruction model in the current experiments

## Configuration
Main training config:

```yaml
model:
  type: pointnet
  num_classes: 10
  dropout: 0.5
  dropout_sa: 0.5
  cnn1d_kernel_size: 3

training:
  batch_size: 64
  num_epochs: 120
  learning_rate: 0.001
  weight_decay: 0.001
  early_stopping_patience: 20

split:
  seed: 1
  train_ratio: 0.7
  val_ratio: 0.1
  test_ratio: 0.2

device: cuda
```

Autoencoder config:

```yaml
autoencoder:
  common:
    num_channels: 3
  mlp:
    latent_dim: 128
    hidden_dims: [256, 128]
    dropout: 0.1
  pointnet:
    latent_dim: 256
    channel_dims: [64, 128, 512]
    decoder_dims: [512, 256, 128]
    dropout: 0.1
    use_tnet: false
  train:
    batch_size: 64
    learning_rate: 0.001
    weight_decay: 0.00001
    num_epochs: 120
    augment: false
```

If you are running on Apple Silicon or CPU-only hardware, update `device` in `config.yaml` before training.

## Current Results
### Classification
Measured from `results/model_comparison.csv`:

| Model | Test Accuracy (%) | F1 (%) | ROC-AUC | Precision (%) | Recall (%) | Parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MLP | 10.85 | 10.94 | 0.5296 | 11.33 | 10.85 | 419,210 |
| CNN1D | 34.80 | 33.90 | 0.7851 | 37.54 | 34.80 | 158,986 |
| PointNet | 72.00 | 72.04 | 0.9600 | 78.23 | 72.00 | 1,606,419 |

Current takeaway:

- `pointnet` is the best classification model by a large margin.
- `cnn1d` is noticeably stronger than `mlp`, but still far behind `pointnet`.
- `mlp` mainly serves as a sanity-check baseline.

### Autoencoder
Measured from `results/ae_comparison.md`:

| Model | CD Mean | CD Median | CD Std | CD Max | Parameters | Inference (s) | ms/sample |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mlp_ae | 0.015800 | 0.014414 | 0.005920 | 0.049512 | 870,236 | 0.3628 | 0.18 |
| pointnet_ae | 0.004430 | 0.002753 | 0.003628 | 0.016544 | 506,115 | 0.4559 | 0.23 |

Current takeaway:

- `pointnet_ae` reconstructs point clouds much more accurately than `mlp_ae`.
- `pointnet_ae` also uses fewer parameters than the MLP autoencoder in the current implementation.

### Ablation: Centering
Measured from `results/experiments/summary_report.txt`:

| Model | With Centering (%) | No Centering (%) | Difference |
| --- | ---: | ---: | ---: |
| MLP | 20.18 | 30.27 | -10.09 |
| CNN1D | 60.91 | 59.09 | +1.82 |
| PointNet | 71.73 | 54.55 | +17.18 |

Main finding:

- Centering is critical for `pointnet`.
- Centering has very little effect on `cnn1d`.
- In this experiment, `mlp` did slightly better without centering.

## Expected Outputs
After training and evaluation, the repository produces artifacts such as:

- `results/checkpoints/<model>/model_best.pth`
- `results/model_comparison.csv`
- `results/ae_comparison.csv`
- `results/ae_comparison.md`
- `results/reports/*.json`
- `results/experiments/summary_report.txt`
- TensorBoard logs under `results/tensorboard/`

## Documentation Map
- `docs/QUICKSTART.md`: CLI setup and essential commands
- `docs/GUI_GUIDE.md`: GUI workflow
- `docs/EXPERIMENT_GUIDE.md`: ablation and comparison workflow
- `docs/MODEL_OPS.md`: deployment and maintenance notes
- `report/FINAL_REPORT.md`: final written report
- `presentation/presentation.md`: slide deck content

## Known Limitations
- FAUST is a mesh dataset, not a real mmWave radar dataset.
- The current project focuses on static point-cloud inputs rather than temporal sequences.
- Results may shift if you regenerate the processed dataset under different preprocessing settings.
- The GUI report export is JSON-based; it does not currently generate PDF slides or polished PDF reports automatically.

## Troubleshooting
If no data is found:

```bash
ls data/raw
```

If CUDA is unavailable:

- set `device: cpu` or `device: mps` in `config.yaml`
- reduce `batch_size`
- reduce `num_points`

If Docker does not start:

```bash
docker-compose logs
docker-compose down
docker-compose up --build
```

## References
- PointNet: [https://arxiv.org/abs/1612.00593](https://arxiv.org/abs/1612.00593)
- PointNet++: [https://arxiv.org/abs/1706.02413](https://arxiv.org/abs/1706.02413)
- FAUST dataset: [http://faust.is.tue.mpg.de/](http://faust.is.tue.mpg.de/)
- PyTorch docs: [https://pytorch.org/docs/](https://pytorch.org/docs/)

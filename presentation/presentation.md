---
title: "Point Cloud Human Identification and Compression"
subtitle: "Deep Learning Final Project - MLDS 432"
author: "[Add Team Names]"
date: "Northwestern University"
---

# Slide 1 - Title

## Point Cloud Human Identification and Compression

- Two tasks: 10-class subject classification and point-cloud reconstruction
- End-to-end deliverable: models + experiments + GUI + Docker workflow
- Dataset used in this repository: FAUST point clouds

Speaker note:
Frame the project as both a modeling study and a usable mini-platform.

---

# Slide 2 - Motivation

## Why this problem matters

- Privacy-preserving sensing is attractive compared with camera-based identification
- Sparse 3D geometry is a natural fit for radar-inspired applications
- We want to know which architectures work best on point-cloud inputs

## Important scope clarification

- Motivation comes from mmWave-style sensing
- Current experiments use FAUST meshes as a proxy dataset
- This project is a proof-of-concept for point-cloud learning, not a direct radar benchmark

---

# Slide 3 - Project Scope

## What we built

- Classification pipeline for `mlp`, `cnn1d`, and `pointnet`
- Autoencoder pipeline for `mlp_ae` and `pointnet_ae`
- Flask backend + browser GUI + Dockerized workflow
- Result comparison tables and preprocessing ablation study

## Deliverables

- CLI scripts for reproducible runs
- GUI for upload, preprocessing, training, monitoring, and download
- Written report and experiment summaries

---

# Slide 4 - Dataset and Pipeline

## Dataset

- FAUST training set
- 100 meshes = 10 subjects x 10 poses
- Current config samples each mesh into 500-point clouds
- `samples_per_mesh = 100` produces the processed cache used for training

## Processing pipeline

1. Raw mesh files from `data/raw/`
2. Farthest point sampling
3. Augmentation and preprocessing
4. Grouped train / val / test split
5. Training and evaluation

Suggested visual:
A pipeline figure from raw mesh to processed point cloud to model output.

---

# Slide 5 - Preprocessing Choices

## Current config used for the refreshed materials

- `num_points: 500`
- `normalize_center: true`
- `normalize_scale: false`
- `rotation_range: 360`
- `translation_range: 0.0`

## Why this matters

- Preprocessing is part of the model pipeline, not just cleanup
- Different architectures respond differently to centering
- We later show that PointNet is especially sensitive to this choice

---

# Slide 6 - Classification Models

## Compared architectures

| Model | Main idea | Strength | Weakness |
| --- | --- | --- | --- |
| MLP | Flatten point cloud | Simple baseline | Order-dependent |
| CNN1D | Conv1D over ordered points | Learns local patterns | Still representation-sensitive |
| PointNet | Shared MLP + max pooling + T-Net | Best fit for unordered point sets | Higher complexity |

## Expected narrative

- MLP is a sanity-check baseline
- CNN1D should improve over MLP
- PointNet should win because it is designed for point sets

---

# Slide 7 - Classification Results

## Measured performance

| Model | Test Accuracy (%) | F1 (%) | ROC-AUC |
| --- | ---: | ---: | ---: |
| MLP | 10.85 | 10.94 | 0.5296 |
| CNN1D | 34.80 | 33.90 | 0.7851 |
| PointNet | 72.00 | 72.04 | 0.9600 |

## Key takeaway

- PointNet clearly outperforms both baselines
- CNN1D is better than MLP but still far behind PointNet
- Flatten-based modeling is not enough for this task

Suggested visual:
Bar chart of test accuracy or ROC-AUC.

---

# Slide 8 - Autoencoder Models

## Compared architectures

| Model | Main idea | Strength | Weakness |
| --- | --- | --- | --- |
| MLP AE | Flatten, compress, reconstruct | Simple baseline | Order-dependent reconstruction |
| PointNet AE | Point-cloud-aware encoder | Better geometry preservation | Slightly slower inference |

## Evaluation metric

- Chamfer Distance
- Lower is better
- Measures how close two point sets are in 3D space

---

# Slide 9 - Autoencoder Results

## Measured performance

| Model | CD Mean | CD Median | Parameters |
| --- | ---: | ---: | ---: |
| mlp_ae | 0.015800 | 0.014414 | 870,236 |
| pointnet_ae | 0.004430 | 0.002753 | 506,115 |

## Key takeaway

- PointNet AE reconstructs geometry much better than the MLP autoencoder
- It also uses fewer parameters in the current implementation
- Point-cloud-aware design helps for both classification and reconstruction

Suggested visual:
Original vs reconstructed point clouds for both autoencoders.

---

# Slide 10 - Ablation Insight: Centering

## Centering study

| Model | With Centering (%) | No Centering (%) | Difference |
| --- | ---: | ---: | ---: |
| MLP | 20.18 | 30.27 | -10.09 |
| CNN1D | 60.91 | 59.09 | +1.82 |
| PointNet | 71.73 | 54.55 | +17.18 |

## Main finding

- PointNet strongly benefits from centered inputs
- CNN1D is mostly unchanged
- MLP slightly prefers no-centering in this setup

## Why this slide matters

- This is the clearest research-style insight in the project
- It shows preprocessing and architecture interact in nontrivial ways

---

# Slide 11 - Platform and Deployment

## What makes this more than a notebook project

- Flask backend for training jobs and report generation
- Browser GUI for upload, preprocessing, training, and monitoring
- Docker / Docker Compose for reproducible startup
- Shared `config.yaml` for pipeline control

## Demo flow

1. Place or upload FAUST meshes
2. Preprocess data
3. Select task and model
4. Train and monitor curves
5. Download checkpoint and report

Suggested visual:
Screenshot of the GUI or a small system diagram.

---

# Slide 12 - Limitations and Future Work

## Limitations

- FAUST is not real mmWave radar data
- Current experiments use static point clouds rather than temporal sequences
- Dataset scale is still limited
- GUI exports JSON reports rather than presentation-ready PDFs

## Future work

- Collect or integrate real sparse sensing data
- Add temporal models for sequential identity cues
- Extend ablations for scaling, augmentation, and split strategy
- Improve deployment polish and richer visualization exports

---

# Slide 13 - Conclusion

## Final takeaways

- PointNet is the strongest classifier in the current repo: 72.00% accuracy
- PointNet AE is the strongest autoencoder: 0.004430 mean Chamfer Distance
- Centering is a major factor for PointNet performance
- The repository now supports both experimentation and demonstration through CLI + GUI

## Questions

- Code: repository root
- Report: `report/FINAL_REPORT.md`
- Operations notes: `docs/MODEL_OPS.md`

---
title: "mmWave Radar Human Identification & Point Cloud Compression"
subtitle: "Deep Learning Final Project - MLDS 432"
author: "[Add Team Names]"
date: "Northwestern University"
---

# Overview (5 min)

## Project Goals

- **Problem 1**: Human identification from sparse 3D point clouds (10-class)
- **Problem 2**: Point cloud compression/reconstruction (Autoencoder)
- **Dataset**: FAUST (10 subjects × 10 poses, 200 points per cloud)

## Why mmWave Radar?

- Privacy-preserving (no facial imagery)
- Works in darkness
- Sparse data (~200 points per frame)

---

# Exploratory Data Analysis (5 min)

## Data Summary

- 100 meshes → 10,000 augmented samples
- Schema: (N, 200, 3) — N samples, 200 points, xyz
- No missing values; normalized to unit sphere

## Visualizations

- Class distribution (bar chart)
- 3D point cloud samples
- Histograms, violin plots, heatmaps
- Q-Q plots, outlier analysis

---

# Model Training - Classification (5 min)

## Architectures

| Model | Accuracy | Notes |
|-------|----------|-------|
| MLP | ~20-40% | Baseline, order-dependent |
| 1D-CNN | ~65-70% | Local patterns |
| **PointNet Tiny** | **~70-85%** | Champion, permutation-invariant |

## Evaluation Metrics

- Accuracy, F1, Precision, Recall
- Confusion Matrix
- **ROC-AUC** (one-vs-rest)

---

# Model Training - Autoencoder (5 min)

## Architectures

| Model | Chamfer Distance | Notes |
|-------|------------------|-------|
| MLP AE | Higher | Challenger |
| **PointNet AE** | **Lower** | Champion, permutation-invariant |

## Loss: Chamfer Distance

- CD = mean min-dist(pred→target) + mean min-dist(target→pred)

---

# Model Operations & Conclusion

## Deployment

- Docker + Flask API + Web GUI
- `bash start.sh` → http://localhost:8080

## Maintenance

- Retraining triggers: performance drop, new data
- Config versioning, A/B testing, rollback plan

## Conclusion

- Two cognitive problems solved
- PointNet Tiny best for classification; PointNet AE best for compression
- Full deployment and maintenance documentation

---

# Thank You

## Questions?

- **Code**: Full source in repository
- **Report**: `report/FINAL_REPORT.pdf`
- **Model Ops**: `docs/MODEL_OPS.md`

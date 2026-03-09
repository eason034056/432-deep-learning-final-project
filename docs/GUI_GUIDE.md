# GUI Guide

## Quick Start

1. Place FAUST mesh files in `data/raw/`
2. Run `bash start.sh`
3. Open http://localhost:8080 in your browser

## Features

- **Data Upload**: Drag & drop mesh files
- **Preprocessing**: Configure sampling and augmentation
- **Task Type**: Choose Classification or Autoencoder
- **Training**: 
  - Classification: MLP, 1D-CNN, PointNet Tiny
  - Autoencoder: MLP AE, PointNet AE
- **Monitoring**: Real-time training curves (loss, accuracy for classification; loss only for AE)
- **Evaluation**: Generate performance reports
- **Download**: Save trained models and results

## Requirements

- Docker and Docker Compose
- 8GB+ RAM recommended

# Quick Start (Command Line)

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Download Data

1. Get FAUST from http://faust.is.tue.mpg.de/
2. Place `.ply` files in `data/raw/`

## Train Classification Models

```bash
python src/train.py --model pointnet
python src/train.py --model cnn1d
python src/train.py --model mlp
```

## Train Autoencoder

```bash
python src/train_ae.py --model mlp_ae
python src/train_ae.py --model mlp_ae
```

## Evaluate

```bash
python src/evaluate.py --model pointnet --checkpoint results/checkpoints/pointnet/model_best.pth
python src/evaluate.py --compare
python src/evaluate_ae.py --compare
```

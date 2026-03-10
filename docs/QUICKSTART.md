# Quick Start (Command Line)

## Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Prepare Data
1. Download FAUST from http://faust.is.tue.mpg.de/
2. Place the mesh files in `data/raw/`

## Train Classification Models
```bash
python src/train.py --config config.yaml --model mlp
python src/train.py --config config.yaml --model cnn1d
python src/train.py --config config.yaml --model pointnet
```

## Train Autoencoders
```bash
python src/train_ae.py --config config.yaml --model mlp_ae
python src/train_ae.py --config config.yaml --model pointnet_ae
```

## Compare Models
```bash
python src/evaluate.py --compare --models mlp cnn1d pointnet
python src/evaluate_ae.py --config config.yaml --compare
```

## Run Ablation
```bash
bash run_experiments.sh
python analyze_results.py
```

## Notes
- The current repository supports `mlp`, `cnn1d`, and `pointnet` for classification.
- The current repository supports `mlp_ae` and `pointnet_ae` for reconstruction.
- If you change point count or preprocessing settings, regenerate the processed dataset before comparing results.

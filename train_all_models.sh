#!/bin/bash
# Script to train all three models sequentially
# Usage: bash train_all_models.sh

echo "======================================================================"
echo "Training All Models for mmWave Human Identification POC"
echo "======================================================================"

# Check if config.yaml exists
if [ ! -f "config.yaml" ]; then
    echo "Error: config.yaml not found!"
    exit 1
fi

# Check if data directory exists
if [ ! -d "data/raw" ]; then
    echo "Error: data/raw directory not found!"
    echo "Please create data/raw and add FAUST mesh files"
    exit 1
fi

echo ""
echo "Starting training pipeline..."
echo ""

# Train MLP Baseline
echo "======================================================================"
echo "[1/3] Training MLP Baseline Model"
echo "======================================================================"
python src/train.py --config config.yaml --model mlp

if [ $? -ne 0 ]; then
    echo "Error: MLP training failed!"
    exit 1
fi

echo ""
echo "MLP training completed successfully!"
echo ""

# Train 1D-CNN Model  
echo "======================================================================"
echo "[2/3] Training 1D-CNN Model"
echo "======================================================================"
python src/train.py --config config.yaml --model cnn1d

if [ $? -ne 0 ]; then
    echo "Error: CNN1D training failed!"
    exit 1
fi

echo ""
echo "1D-CNN training completed successfully!"
echo ""

# Train Tiny PointNet
echo "======================================================================"
echo "[3/3] Training Tiny PointNet Model"
echo "======================================================================"
python src/train.py --config config.yaml --model pointnet

if [ $? -ne 0 ]; then
    echo "Error: PointNet training failed!"
    exit 1
fi

echo ""
echo "PointNet training completed successfully!"
echo ""

# Compare all models
echo "======================================================================"
echo "Comparing All Models"
echo "======================================================================"
python src/evaluate.py --compare --models mlp cnn1d pointnet

echo ""
echo "======================================================================"
echo "All models trained and evaluated successfully!"
echo "======================================================================"
echo ""
echo "Results saved to:"
echo "  - Checkpoints: results/checkpoints/"
echo "  - TensorBoard logs: results/tensorboard/"
echo "  - Comparison table: results/model_comparison.csv"
echo ""
echo "To view training curves:"
echo "  tensorboard --logdir results/tensorboard"
echo ""


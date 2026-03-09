#!/bin/bash
# Unified Experiment Runner Script
# Tests 3 models with/without centering normalization
#
# Experiment Design:
#   - Centering: with vs without
#   - Models: MLP, CNN1D (kernel_size=3), PointNet
#   - Split: Grouped (no data leakage)
#   - Total runs: 2 × 3 = 6 training runs
#
# Estimated time: ~2-3 hours

set -e  # Exit on error

echo "================================================================"
echo "Unified Experiment Runner Script"
echo "================================================================"
echo ""
echo "Experiment Configuration:"
echo "  • Centering options: With / Without"
echo "  • Models: MLP, CNN1D (k=3), PointNet"
echo "  • Split strategy: Grouped (no data leakage)"
echo "  • Total runs: 6"
echo ""
echo "Estimated time: ~2-3 hours"
echo ""
read -p "Press Enter to start the experiments..."

# ========================================
# Setup and Initialization
# ========================================

# Results directory
RESULTS_DIR="results/experiments"
mkdir -p $RESULTS_DIR

echo ""
echo "✓ Results directory created: $RESULTS_DIR"

# ========================================
# Backup and Restore Functions
# ========================================

backup_config() {
    cp config.yaml config.yaml.backup
    echo "✓ Config backed up"
}

restore_config() {
    if [ -f config.yaml.backup ]; then
        mv config.yaml.backup config.yaml
        echo "✓ Config restored"
    fi
}

# ========================================
# Configuration Update Functions
# ========================================

set_normalize_center() {
    local value=$1
    sed -i '' "s/normalize_center: .*/normalize_center: $value  # Experiment setting/" config.yaml
}

clean_processed_data() {
    echo "  Cleaning processed data..."
    rm -f data/processed/faust_pc.npz
}

# ========================================
# Training Function
# ========================================

train_model() {
    local model=$1
    local center_setting=$2
    
    echo ""
    echo "================================================================"
    echo "Training: $model"
    echo "  Centering: $center_setting"
    echo "================================================================"
    
    # Clean old checkpoints and logs
    rm -rf results/checkpoints/${model}/*
    rm -rf results/tensorboard/${model}/*
    
    # Train model
    echo "  Starting training..."
    python src/train.py --config config.yaml --model $model
    
    # Save results
    local result_dir="$RESULTS_DIR/${center_setting}_${model}"
    mkdir -p $result_dir
    cp -r results/checkpoints/${model} $result_dir/checkpoints
    cp -r results/tensorboard/${model} $result_dir/tensorboard
    
    echo "✓ Results saved: $result_dir"
}

# ========================================
# Main Experiment Flow
# ========================================

backup_config

# Ensure normalize_scale is false (keep absolute body size)
sed -i '' "s/normalize_scale: .*/normalize_scale: false  # Keep absolute body size/" config.yaml

# ========================================
# Experiment Group 1: WITH CENTERING
# ========================================
echo ""
echo "================================================================"
echo "Experiment Group 1: WITH CENTERING"
echo "normalize_center=true (center at origin, keep body size)"
echo "================================================================"

set_normalize_center "true"
clean_processed_data

train_model "mlp" "with_center"
train_model "cnn1d" "with_center"
train_model "pointnet" "with_center"

# ========================================
# Experiment Group 2: WITHOUT CENTERING
# ========================================
echo ""
echo "================================================================"
echo "Experiment Group 2: WITHOUT CENTERING"
echo "normalize_center=false (keep original coordinates and body size)"
echo "================================================================"

set_normalize_center "false"
clean_processed_data

train_model "mlp" "no_center"
train_model "cnn1d" "no_center"
train_model "pointnet" "no_center"

# ========================================
# Completion and Cleanup
# ========================================

restore_config

echo ""
echo "================================================================"
echo "✓ EXPERIMENTS COMPLETED!"
echo "================================================================"
echo ""
echo "Results Directory: $RESULTS_DIR"
echo ""
echo "Directory Structure:"
echo "  with_center_mlp/         - MLP with centering"
echo "  with_center_cnn1d/       - CNN1D with centering (kernel_size=3)"
echo "  with_center_pointnet/    - PointNet with centering"
echo "  no_center_mlp/           - MLP without centering"
echo "  no_center_cnn1d/         - CNN1D without centering (kernel_size=3)"
echo "  no_center_pointnet/      - PointNet without centering"
echo ""
echo "Processed Data:"
echo "  data/processed/faust_pc.npz - grouped split (no data leakage)"
echo ""
echo "================================================================"
echo "Next Steps:"
echo "================================================================"
echo ""
echo "1. View training curves in TensorBoard:"
echo "   tensorboard --logdir=$RESULTS_DIR"
echo ""
echo "2. Generate analysis report and plots:"
echo "   python analyze_results.py"
echo ""
echo "3. Evaluate single model:"
echo "   python src/evaluate.py --model mlp --checkpoint $RESULTS_DIR/with_center_mlp/checkpoints/model_best.pth"
echo ""



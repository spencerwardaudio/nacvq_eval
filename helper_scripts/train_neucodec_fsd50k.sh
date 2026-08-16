#!/bin/bash
#
# Train NeuCodec on FSD50K with W&B monitoring
#

set -e  # Exit on error

# Configuration
VENV_PATH=".venv"
NEUCODEC_DIR="neucodec/training"
DATA_DIR="data"

# W&B settings (can be overridden with environment variables)
export WANDB_PROJECT="${WANDB_PROJECT:-neucodec-fsd50k}"
export WANDB_NAME="${WANDB_NAME:-neucodec-fsd50k-$(date +%Y%m%d-%H%M%S)}"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please create it first: python3 -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Check/create data directory and convert CSVs to txt filelists
echo "Preparing FSD50K filelists..."
mkdir -p "$DATA_DIR"

# Convert FSD50K CSVs to txt format (one path per line)
# Assuming CSVs are in Encodec/config paths (from config_multi_dataset.yaml)
FSD50K_TRAIN_CSV="/home/jovyan/teaching_material/MSC_Project_SW/msc_proj/datasets/fsd50k_train.csv"
FSD50K_VAL_CSV="/home/jovyan/teaching_material/MSC_Project_SW/msc_proj/datasets/fsd50k_val.csv"

# Check if we're on the server or local
if [ ! -f "$FSD50K_TRAIN_CSV" ]; then
    # Try local paths
    FSD50K_TRAIN_CSV="datasets/fsd50k_train.csv"
    FSD50K_VAL_CSV="datasets/fsd50k_val.csv"
fi

if [ ! -f "$FSD50K_TRAIN_CSV" ]; then
    echo "Error: Cannot find FSD50K train CSV at $FSD50K_TRAIN_CSV"
    echo "Please ensure the dataset is available"
    exit 1
fi

# Convert CSVs to txt filelists (CSVs are headerless, one path per line)
echo "Converting CSVs to txt filelists..."
cp "$FSD50K_TRAIN_CSV" "$DATA_DIR/fsd50k_train_files.txt"
cp "$FSD50K_VAL_CSV" "$DATA_DIR/fsd50k_val_files.txt"

echo "Train files: $(wc -l < $DATA_DIR/fsd50k_train_files.txt)"
echo "Val files: $(wc -l < $DATA_DIR/fsd50k_val_files.txt)"

# Check if dependencies are installed
echo "Checking dependencies..."
python -c "import torch; import pytorch_lightning; import hydra; import wandb; import neucodec" 2>/dev/null || {
    echo "Error: Missing dependencies. Please install:"
    echo "  pip install pytorch-lightning hydra-core wandb"
    echo "  pip install -e neucodec"
    exit 1
}

# Set GPU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Change to training directory
cd "$NEUCODEC_DIR"

# Run training
echo "Starting NeuCodec training..."
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Run Name: $WANDB_NAME"
echo ""

python train.py \
    --config-name fsd50k \
    wandb.project="$WANDB_PROJECT" \
    wandb.name="$WANDB_NAME"

echo "Training completed!"

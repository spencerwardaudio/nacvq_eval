#!/bin/bash
#
# Train SpeechTokenizer on FSD50K with W&B monitoring
#

set -e  # Exit on error

# Configuration
VENV_PATH=".venv"
SPT_DIR="SpeechTokenizer"
DATA_DIR="data"

# W&B settings (can be overridden with environment variables)
export WANDB_PROJECT="${WANDB_PROJECT:-codec-fsd50k-scale}"
export WANDB_NAME="${WANDB_NAME:-spt-fsd50k-$(date +%Y%m%d-%H%M%S)}"

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    echo "Please create it first: python3 -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"

# Check/create data directory
echo "Preparing FSD50K filelists..."
mkdir -p "$DATA_DIR"

# Convert FSD50K CSVs to txt format (one path per line)
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
python -c "import torch; import accelerate; import beartype; import wandb; import speechtokenizer" 2>/dev/null || {
    echo "Error: Missing dependencies. Please install:"
    echo "  pip install accelerate beartype wandb"
    echo "  pip install -e SpeechTokenizer"
    exit 1
}

# Apply W&B patch
echo "Applying W&B patch to SpeechTokenizer trainer..."
PATCH_MARKER="$SPT_DIR/speechtokenizer/trainer/.wandb_patched"
if [ ! -f "$PATCH_MARKER" ]; then
    python patch_speechtokenizer_wandb.py "$SPT_DIR/speechtokenizer/trainer/trainer.py"
    touch "$PATCH_MARKER"
else
    echo "W&B patch already applied"
fi

# Set GPU
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Run training
echo "Starting SpeechTokenizer training..."
echo "W&B Project: $WANDB_PROJECT"
echo "W&B Run Name: $WANDB_NAME"
echo ""

cd "$SPT_DIR"
python scripts/train_example.py --config config/fsd50k_cfg.json

echo "Training completed!"

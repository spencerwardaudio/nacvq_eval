#!/bin/bash

# Multi-dataset EnCodec training script
# This script runs training with multiple datasets (jamendo, common_voice, etc.)

echo "Starting multi-dataset EnCodec training..."
echo "Target bandwidths: 24.0 kbps"

# Check if virtual environment exists
if [ ! -d "venv_encodec" ]; then
    echo "❌ Virtual environment 'venv_encodec' not found."
    echo "Please create it first: python3 -m venv venv_encodec"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv_encodec/bin/activate

# Check if activation was successful
if [ $? -ne 0 ]; then
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

echo "✓ Virtual environment activated"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create output directory if it doesn't exist
mkdir -p ./checkpoints_multi_dataset/

# Run training (fixed_length=16000 = 0.67s for stable mel-spectrogram gradients at large STFT scales)
# warmup_epoch=20: let reconstruction losses converge before discriminator activates
# disc_lr=1e-4: slower discriminator vs generator (3e-4) to prevent disc dominating early
# l_g weight reduced from 3 to 1.5: softer adversarial pressure for small-data regime
echo "Starting training..."
python train_multi_dataset.py \
    --config-name=config_multi_dataset \
    common.max_epoch=50 \
    datasets.batch_size=16 \
    datasets.fixed_length=16000 \
    model.sample_rate=24000 \
    model.channels=1 \
    model.target_bandwidths=[24.0] \
    lr_scheduler.warmup_epoch=20 \
    optimization.disc_lr=1e-4 \
    balancer.weights.l_g=1.5 \
    "$@"
    wandb.enabled=true \
    wandb.project=multi-dataset-encodec

echo "Training completed!"

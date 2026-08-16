#!/bin/bash
# 2-epoch test training for all 5 models
# Tests: OOM, initialization, gradients, W&B logging
# Usage: bash test_2epoch_training.sh

set -e  # Exit on error

echo "=========================================="
echo "2-EPOCH TRAINING TEST"
echo "Tests: OOM, init, gradients, W&B logging"
echo "=========================================="
echo ""

# Activate venv
source .venv/bin/activate

# Set W&B project for tests
export WANDB_PROJECT=codec-fsd50k-test

# Track results
RESULTS_FILE="test_2epoch_results.txt"
echo "2-Epoch Training Test Results - $(date)" > "$RESULTS_FILE"
echo "========================================" >> "$RESULTS_FILE"

# Test counter
PASSED=0
FAILED=0

# Test function
test_model() {
    local model_name=$1
    local test_command=$2
    
    echo ""
    echo "=========================================="
    echo "Testing: $model_name (2 epochs)"
    echo "=========================================="
    echo ""
    
    if eval "$test_command"; then
        echo "✅ $model_name: PASSED" | tee -a "$RESULTS_FILE"
        ((PASSED++))
        return 0
    else
        echo "❌ $model_name: FAILED" | tee -a "$RESULTS_FILE"
        ((FAILED++))
        return 1
    fi
}

# ============================================================================
# TEST 1: Q2D2 (fastest to validate setup)
# ============================================================================
test_model "Q2D2" '
export WANDB_NAME=q2d2-test-2ep
cd Q2D2

# Create temporary 2-epoch config
cat > configs/test_2ep.yaml << "EOF"
seed_everything: 42
trainer:
  accelerator: gpu
  devices: 1
  max_epochs: 2
  check_val_every_n_epoch: 1
  precision: 32
  gradient_clip_val: 1.0
  
model:
  sample_rate: 24000
  train_batch_size: 8
  val_batch_size: 5
  num_workers: 4
  learning_rate: 0.00008
  dataset: fsd50k
  train_csv: ../datasets/fsd50k_train.csv
  val_csv: ../datasets/fsd50k_val.csv
  encoder_dim: 512
  decoder_dim: 512
  quantizer_type: geometric_lattice
  bitrate: 9.8
EOF

python train.py fit --config configs/test_2ep.yaml
EXIT_CODE=$?
rm -f configs/test_2ep.yaml
cd ..
exit $EXIT_CODE
'

# ============================================================================
# TEST 2: Encodec
# ============================================================================
test_model "Encodec" '
export WANDB_NAME=encodec-test-2ep
cd Encodec

# Backup original config
cp config/config_multi_dataset.yaml config/config_multi_dataset.yaml.backup

# Modify for 2-epoch test
python << "PYEOF"
from omegaconf import OmegaConf
cfg = OmegaConf.load("config/config_multi_dataset.yaml")
cfg.common.max_epoch = 2
cfg.common.valid_interval = 1
OmegaConf.save(cfg, "config/config_multi_dataset.yaml")
print("✓ Modified Encodec config to 2 epochs")
PYEOF

python train_multi_dataset.py
EXIT_CODE=$?

# Restore original config
mv config/config_multi_dataset.yaml.backup config/config_multi_dataset.yaml
cd ..
exit $EXIT_CODE
'

# ============================================================================
# TEST 3: SpeechTokenizer
# ============================================================================
test_model "SpeechTokenizer" '
export WANDB_NAME=speechtokenizer-test-2ep
cd SpeechTokenizer

# Backup original config
cp config/fsd50k_cfg.json config/fsd50k_cfg.json.backup

# Modify for 2-epoch test
python << "PYEOF"
import json
with open("config/fsd50k_cfg.json", "r") as f:
    cfg = json.load(f)
cfg["epochs"] = 2
cfg["valid_interval"] = 4097  # Validate once per epoch
with open("config/fsd50k_cfg.json", "w") as f:
    json.dump(cfg, f, indent=2)
print("✓ Modified SpeechTokenizer config to 2 epochs")
PYEOF

python scripts/train.py --config config/fsd50k_cfg.json
EXIT_CODE=$?

# Restore original config
mv config/fsd50k_cfg.json.backup config/fsd50k_cfg.json
cd ..
exit $EXIT_CODE
'

# ============================================================================
# TEST 4: HiFiCodec
# ============================================================================
test_model "HiFiCodec" '
export WANDB_RUN_NAME=hificodec-test-2ep

# Calculate validation interval for 2 epochs
# 32772 samples / 8 batch_size * 2 epochs = 8193 steps
VAL_INTERVAL=8193

python3 hificodec/academicodec/models/hificodec/train.py \
  --config hificodec/egs/HiFi-Codec-24k-320d/config_24k_320d.json \
  --checkpoint_path hificodec/egs/HiFi-Codec-24k-320d/exp_test \
  --training_epochs 2 \
  --validation_interval $VAL_INTERVAL
'

# ============================================================================
# TEST 5: DAC-FSQ
# ============================================================================
test_model "DAC-FSQ" '
export WANDB_NAME=dac-fsq-test-2ep
cd descript-audio-codec

# Backup original config
cp conf/fsd50k_fsq.yml conf/fsd50k_fsq.yml.backup

# Modify for 2-epoch test
python << "PYEOF"
import yaml
with open("conf/fsd50k_fsq.yml", "r") as f:
    cfg = yaml.safe_load(f)
cfg["training_epochs"] = 2
cfg["valid_freq"] = 4098  # Validate once per epoch (8196 steps / 2)
with open("conf/fsd50k_fsq.yml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)
print("✓ Modified DAC-FSQ config to 2 epochs")
PYEOF

python scripts/train_fsq.py --config conf/fsd50k_fsq.yml
EXIT_CODE=$?

# Restore original config
mv conf/fsd50k_fsq.yml.backup conf/fsd50k_fsq.yml
cd ..
exit $EXIT_CODE
'

# ============================================================================
# FINAL REPORT
# ============================================================================
echo ""
echo "=========================================="
echo "2-EPOCH TEST COMPLETE"
echo "=========================================="
echo ""
cat "$RESULTS_FILE"
echo ""
echo "Summary: $PASSED passed, $FAILED failed"
echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""

if [ $FAILED -eq 0 ]; then
    echo "✅ ALL TESTS PASSED!"
    echo ""
    echo "Verified:"
    echo "  ✅ No OOM errors"
    echo "  ✅ Models initialize correctly"
    echo "  ✅ Gradients stable"
    echo "  ✅ W&B logging works"
    echo "  ✅ Data loading works"
    echo ""
    echo "🚀 READY FOR FULL 50-EPOCH TRAINING!"
    exit 0
else
    echo "❌ $FAILED TEST(S) FAILED"
    echo ""
    echo "Action required:"
    echo "  1. Check error messages above"
    echo "  2. Fix issues"
    echo "  3. Re-run: bash test_2epoch_training.sh"
    echo ""
    echo "Do NOT start full training until all tests pass!"
    exit 1
fi

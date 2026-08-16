#!/usr/bin/env bash
# train_5codecs.sh — Train all 5 neural audio codecs on FSD50K (50 epochs each)
#
# Trains: SpeechTokenizer, Encodec, DAC-FSQ, HiFiCodec, Q2D2
# Dataset: FSD50K with 8:1:1 train/val/test split (already prepared)
# Epochs: 50 with validation every 5 epochs
# W&B Project: codec-fsd50k-scale (customizable via env)
#
# Usage (RECOMMENDED - survives SSH disconnection):
#   nohup bash train_5codecs.sh [--gpu 0] > training.log 2>&1 &
#   echo $! > training.pid
#   tail -f training.log
#
# Alternative (direct run, but SSH disconnect will kill it):
#   bash train_5codecs.sh [--gpu 0]
#
# To monitor progress later:
#   tail -f training.log
#
# To check if still running:
#   ps -p $(cat training.pid)
#
# Environment variables (optional):
#   WANDB_PROJECT   W&B project name (default: codec-fsd50k-scale)
#   CUDA_VISIBLE_DEVICES  GPU to use (default: 0)
#
# Output:
#   - SpeechTokenizer: results/speechtokenizer_fsd50k/
#   - Encodec: Encodec/checkpoints_multi_dataset/
#   - DAC-FSQ: descript-audio-codec/ckpt/fsd50k_fsq/
#   - HiFiCodec: hificodec/egs/hificodec_fsd50k/
#   - Q2D2: Q2D2/Q2D2_fsd50k_9.8kbps/
#   - Training log: training.log (when using nohup)

set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJ_ROOT"

# ── Configuration ─────────────────────────────────────────────────────────────
EPOCHS=50
VAL_INTERVAL=5
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_PROJECT="${WANDB_PROJECT:-codec-fsd50k-scale}"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) export CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    *) echo "[WARN] Unknown arg: $1"; shift ;;
  esac
done

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
section() { echo ""; echo "========================================"; echo "  $*"; echo "========================================"; }

# ── Pre-flight checks ─────────────────────────────────────────────────────────
log "Starting 5-codec training pipeline"
log "  Epochs per model: ${EPOCHS}"
log "  Validation: every ${VAL_INTERVAL} epochs"
log "  GPU: ${CUDA_VISIBLE_DEVICES}"
log "  W&B project: ${WANDB_PROJECT}"
log "  Project root: ${PROJ_ROOT}"

# Check datasets exist
for dataset in datasets/fsd50k_train.csv datasets/fsd50k_val.csv; do
  if [[ ! -f "$dataset" ]]; then
    log "[ERROR] Dataset not found: $dataset"
    exit 1
  fi
done

# Activate venv if exists
if [[ -f "${PROJ_ROOT}/.venv/bin/activate" ]]; then
  source "${PROJ_ROOT}/.venv/bin/activate"
  log "Activated venv: ${PROJ_ROOT}/.venv"
fi

# ══════════════════════════════════════════════════════════════════════════════
# 1. DAC-FSQ (Descript Audio Codec with Finite Scalar Quantization)
# ══════════════════════════════════════════════════════════════════════════════
section "1/5: DAC-FSQ"
export WANDB_NAME="dac-fsq-fsd50k-50ep"

log "Training DAC-FSQ (24kHz, FSQ quantizer)"
log "  Epochs: ${EPOCHS}"

bash train_dac_fsq_fsd50k.sh --epochs ${EPOCHS} --gpus ${CUDA_VISIBLE_DEVICES}

log "✓ DAC-FSQ training complete"

# ══════════════════════════════════════════════════════════════════════════════
# 2. Q2D2 (Two-Dimensional Quantization with rhombic grid lattice)
# ══════════════════════════════════════════════════════════════════════════════
section "2/5: Q2D2"
export WANDB_NAME="q2d2-fsd50k-50ep"

log "Training Q2D2 (rhombic grid quantization, 9.8kbps)"
log "  Max epochs: ${EPOCHS}"
log "  Check val every: ${VAL_INTERVAL} epochs"

cd Q2D2
python train.py fit \
  --config configs/Q2D2_fsd50k_9.8kbps_dim512_attn_b16.yaml \
  --trainer.max_epochs=${EPOCHS} \
  --trainer.check_val_every_n_epoch=${VAL_INTERVAL}

cd "$PROJ_ROOT"
log "✓ Q2D2 training complete"

# ══════════════════════════════════════════════════════════════════════════════
# 3. ENCODEC (RVQ, 24kbps)
# ══════════════════════════════════════════════════════════════════════════════
section "3/5: Encodec"
export WANDB_NAME="encodec-fsd50k-50ep"

log "Training Encodec with multi-dataset trainer"
log "  Max epochs: ${EPOCHS}"
log "  Validation interval: ${VAL_INTERVAL}"

cd Encodec
python train_multi_dataset.py \
  common.max_epoch=${EPOCHS} \
  common.valid_interval=${VAL_INTERVAL} \
  wandb.enabled=true \
  wandb.project="${WANDB_PROJECT}" \
  wandb.name="${WANDB_NAME}"

cd "$PROJ_ROOT"
log "✓ Encodec training complete"

# ══════════════════════════════════════════════════════════════════════════════
# 4. SPEECHTOKENIZER (with semantic distillation disabled for FSD50K)
# ══════════════════════════════════════════════════════════════════════════════
section "4/5: SpeechTokenizer"
export WANDB_NAME="speechtokenizer-fsd50k-50ep"

log "Training SpeechTokenizer (distill_loss_lambda=0 for general audio)"
log "  Config: SpeechTokenizer/config/fsd50k_cfg.json"
log "  Saves every 5000 steps, validates on save"

python SpeechTokenizer/scripts/train_example.py \
  --config SpeechTokenizer/config/fsd50k_cfg.json

log "✓ SpeechTokenizer training complete"

# ══════════════════════════════════════════════════════════════════════════════
# 5. HIFICODEC (HiFi-Codec with group-residual VQ)
# ══════════════════════════════════════════════════════════════════════════════
section "5/5: HiFiCodec"
export WANDB_RUN_NAME="hificodec-fsd50k-50ep"

log "Training HiFiCodec"
log "  Epochs: ${EPOCHS}"
log "  Uses train_fsd50k.sh wrapper (handles PYTHONPATH and patching)"

bash train_fsd50k.sh --epochs ${EPOCHS} --gpus ${CUDA_VISIBLE_DEVICES}

log "✓ HiFiCodec training complete"

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section "Training Complete!"
log "All 5 neural audio codecs trained successfully on FSD50K"
log ""
log "Checkpoints saved to:"
log "  1. DAC-FSQ:         descript-audio-codec/ckpt/fsd50k_fsq/"
log "  2. Q2D2:            Q2D2/outputs/lightning_logs/"
log "  3. Encodec:         Encodec/checkpoints_multi_dataset/"
log "  4. SpeechTokenizer: results/speechtokenizer_fsd50k/"
log "  5. HiFiCodec:       hificodec/egs/hificodec_fsd50k/"
log ""
log "View training logs at: https://wandb.ai/YOUR_USERNAME/${WANDB_PROJECT}"
log ""
log "Next steps:"
log "  - Evaluate models: bash evaluate_codecs.sh"
log "  - Run analysis: python tok_analysis/analyze_all_codecs.py"
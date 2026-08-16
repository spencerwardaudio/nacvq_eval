#!/usr/bin/env bash
# train_dac_fsq_fsd50k.sh — Train DAC-FSQ (24 kHz, FSQ quantizer) on FSD50K with W&B
#
# Usage:
#   bash train_dac_fsq_fsd50k.sh [--epochs 50] [--gpus 0] [--checkpoint-path PATH]
#                               [--n-train-examples N] [--n-val-examples N]
#                               [--num-iters N]
#
# Environment:
#   WANDB_PROJECT   W&B project (default: codec-fsd50k-scale)
#   WANDB_NAME      W&B run name (auto-timestamped if unset)
#   CUDA_VISIBLE_DEVICES  GPU(s) to use (default: 0)
#
# Example:
#   export WANDB_PROJECT=codec-fsd50k-scale
#   bash train_dac_fsq_fsd50k.sh --gpus 0

set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAC_DIR="${PROJ_ROOT}/descript-audio-codec"
CHECKPOINT_PATH="${DAC_DIR}/ckpt/fsd50k_fsq"
EPOCHS=50
N_TRAIN_EXAMPLES=12000
N_VAL_EXAMPLES=1500
NUM_ITERS=""

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --epochs)              EPOCHS="$2";                      shift 2 ;;
    --gpus)                export CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
    --checkpoint-path)     CHECKPOINT_PATH="$2";             shift 2 ;;
    --n-train-examples)    N_TRAIN_EXAMPLES="$2";            shift 2 ;;
    --n-val-examples)      N_VAL_EXAMPLES="$2";              shift 2 ;;
    --num-iters)           NUM_ITERS="$2";                   shift 2 ;;
    *) echo "[WARN] Unknown arg: $1"; shift ;;
  esac
done

# ── Environment ───────────────────────────────────────────────────────────────
export WANDB_PROJECT="${WANDB_PROJECT:-codec-fsd50k-scale}"
export WANDB_NAME="${WANDB_NAME:-dac-fsq-24k-$(date +%Y%m%d-%H%M%S)}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== DAC-FSQ FSD50K Training ==="
log "Project root : ${PROJ_ROOT}"
log "Checkpoint   : ${CHECKPOINT_PATH}"
log "Epochs       : ${EPOCHS}"
if [[ -n "${NUM_ITERS}" ]]; then
  log "Num iters    : ${NUM_ITERS}"
fi
log "GPU(s)       : ${CUDA_VISIBLE_DEVICES}"
log "W&B project  : ${WANDB_PROJECT}"
log "W&B run      : ${WANDB_NAME}"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
TRAIN_CSV="${PROJ_ROOT}/datasets/fsd50k_train.csv"
VAL_CSV="${PROJ_ROOT}/datasets/fsd50k_val.csv"
CONF="${DAC_DIR}/conf/fsd50k_fsq.yml"
TRAIN_SCRIPT="${DAC_DIR}/scripts/train_fsq.py"

for f in "${TRAIN_CSV}" "${VAL_CSV}" "${CONF}" "${TRAIN_SCRIPT}"; do
  if [[ ! -f "${f}" ]]; then
    log "[ERROR] Required file not found: ${f}"
    exit 1
  fi
done

# ── Activate venv ─────────────────────────────────────────────────────────────
if [[ -f "${PROJ_ROOT}/.venv/bin/activate" ]]; then
  source "${PROJ_ROOT}/.venv/bin/activate"
  log "Activated venv: ${PROJ_ROOT}/.venv"
else
  log "[WARN] .venv not found — using system Python"
fi

# ── Ensure vector_quantize_pytorch is installed ───────────────────────────────
if ! python3 -c "import vector_quantize_pytorch" 2>/dev/null; then
  log "Installing vector-quantize-pytorch..."
  pip install vector-quantize-pytorch
fi

# ── Run from DAC dir so argbind resolves conf/base.yml relative includes ─────
cd "${DAC_DIR}"

EXTRA_ARGS=()
if [[ -n "${NUM_ITERS}" ]]; then
  EXTRA_ARGS+=(--num_iters "${NUM_ITERS}")
fi

log "Starting training..."
python3 scripts/train_fsq.py \
  --args.load conf/fsd50k_fsq.yml \
  --device cuda \
  --amp true \
  --training_epochs "${EPOCHS}" \
  --save_path "${CHECKPOINT_PATH}" \
  --"train/build_dataset.filelist" "${PROJ_ROOT}/datasets/fsd50k_train.csv" \
  --"val/build_dataset.filelist" "${PROJ_ROOT}/datasets/fsd50k_val.csv" \
  --"train/build_dataset.n_examples" "${N_TRAIN_EXAMPLES}" \
  --"val/build_dataset.n_examples" "${N_VAL_EXAMPLES}" \
  "${EXTRA_ARGS[@]}"

log "Training complete. Checkpoints: ${CHECKPOINT_PATH}"

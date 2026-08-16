#!/usr/bin/env bash
# train_fsd50k.sh — FSD50K codec training launcher
#
# Trains HiFiCodec on FSD50K for 50 epochs.
# Other codecs (DAC-FSQ, SpeechTokenizer, Encodec, Q2D2) have their own training scripts.
#
# Usage:
#   bash train_fsd50k.sh [--epochs N] [--gpus 0] [--checkpoint-path PATH]
#
# Environment:
#   WANDB_PROJECT   Set to enable wandb logging (e.g. export WANDB_PROJECT=codec-fsd50k)
#   WANDB_RUN_NAME  Optional wandb run name (default: hificodec-fsd50k-50ep)
#
# Example (server):
#   cd /home/jovyan/teaching_material/MSC_Project_SW/msc_proj
#   export WANDB_PROJECT=codec-fsd50k
#   bash train_fsd50k.sh --epochs 50 --gpus 0

set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────────────
EPOCHS=50
GPUS="0"
CHECKPOINT_PATH="${PROJ_ROOT}/hificodec/egs/hificodec_fsd50k"
HIFICODEC_EGS="${PROJ_ROOT}/hificodec/egs/HiFi-Codec-24k-320d"
HIFICODEC_TRAIN="${PROJ_ROOT}/hificodec/academicodec/models/hificodec/train.py"
CONFIG="${HIFICODEC_EGS}/config_24k_320d.json"
LST_DIR="${PROJ_ROOT}/hificodec/egs/data"
TRAIN_CSV="${PROJ_ROOT}/datasets/fsd50k_train.csv"
VAL_CSV="${PROJ_ROOT}/datasets/fsd50k_val.csv"
TRAIN_LST="${LST_DIR}/fsd50k_train.lst"
VAL_LST="${LST_DIR}/fsd50k_val.lst"

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)        EPOCHS="$2";          shift 2 ;;
        --gpus)          GPUS="$2";            shift 2 ;;
        --checkpoint-path) CHECKPOINT_PATH="$2"; shift 2 ;;
        *) echo "[WARN] Unknown arg: $1"; shift ;;
    esac
done

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Pre-flight checks ─────────────────────────────────────────────────────────
log "=== FSD50K Codec Training ==="
log "Project root : ${PROJ_ROOT}"
log "Epochs       : ${EPOCHS}"
log "GPUs         : ${GPUS}"
log "Checkpoint   : ${CHECKPOINT_PATH}"

if [[ ! -f "${TRAIN_CSV}" ]]; then
    echo "[ERROR] Train CSV not found: ${TRAIN_CSV}"
    echo "  Expected: datasets/fsd50k_train.csv"
    exit 1
fi
if [[ ! -f "${VAL_CSV}" ]]; then
    echo "[ERROR] Val CSV not found: ${VAL_CSV}"
    exit 1
fi
if [[ ! -f "${HIFICODEC_TRAIN}" ]]; then
    echo "[ERROR] HiFiCodec train.py not found: ${HIFICODEC_TRAIN}"
    echo "  Did you clone the hificodec submodule? Run: git submodule update --init"
    exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] HiFiCodec config not found: ${CONFIG}"
    exit 1
fi

# ── Other codecs (trained separately via run_pipeline.py) ───────────────────
log ""
log "=== Other codecs (trained separately) ==="
log "  Encodec  : see Encodec/checkpoints_multi_dataset/"
log "  Q2D2     : see Q2D2/outputs/"
log "  DAC-FSQ  : see descript-audio-codec/ckpt/fsd50k_fsq/"
log "  SpeechTokenizer: see results/speechtokenizer_fsd50k/"

# ── Generate .lst files from CSV ──────────────────────────────────────────────
log ""
log "=== Generating HiFiCodec .lst filelists from FSD50K CSVs ==="
mkdir -p "${LST_DIR}"

# CSV format: headerless, one absolute path per line (as written by generate_dataset_csvs.py)
python3 - <<'PYEOF'
import csv, pathlib, sys, os

proj = pathlib.Path(os.environ.get("PROJ_ROOT", "."))
train_csv = proj / "datasets" / "fsd50k_train.csv"
val_csv   = proj / "datasets" / "fsd50k_val.csv"
lst_dir   = proj / "hificodec" / "egs" / "data"
lst_dir.mkdir(parents=True, exist_ok=True)

for csv_path, out_name in [(train_csv, "fsd50k_train.lst"), (val_csv, "fsd50k_val.lst")]:
    paths = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            p = row[0].strip().strip('"')
            if p and not p.startswith("fname") and not p.startswith("filepath"):
                paths.append(p)
    out = lst_dir / out_name
    out.write_text("\n".join(paths) + "\n")
    print(f"  {out_name}: {len(paths)} files")

# Sample-rate check on first file
first = paths[0] if paths else None
if first and pathlib.Path(first).exists():
    try:
        import torchaudio
        info = torchaudio.info(first)
        sr = info.sample_rate
        print(f"  Sample rate of first FSD50K file: {sr} Hz")
        if sr != 24000:
            print(f"  [WARN] HiFiCodec expects 24kHz but FSD50K is {sr}Hz.")
            print(f"         HiFiCodec will resample on-the-fly via its dataloader.")
    except Exception as e:
        print(f"  [INFO] Could not check sample rate: {e}")
PYEOF

export PROJ_ROOT
python3 - <<'PYEOF'
import os, pathlib

proj  = pathlib.Path(os.environ["PROJ_ROOT"])
lst   = proj / "hificodec" / "egs" / "data" / "fsd50k_train.lst"
n     = sum(1 for _ in open(lst))
print(f"  fsd50k_train.lst: {n} entries")
PYEOF

log "Filelists written to ${LST_DIR}"

# ── Optional wandb patch ───────────────────────────────────────────────────────
if [[ -n "${WANDB_PROJECT:-}" ]]; then
    log ""
    log "=== wandb enabled (project: ${WANDB_PROJECT}) ==="
    export WANDB_RUN_NAME="${WANDB_RUN_NAME:-hificodec-fsd50k-${EPOCHS}ep}"
    
    # Apply W&B patch using dedicated script
    PATCH_MARKER="${PROJ_ROOT}/hificodec/academicodec/models/hificodec/.wandb_patched"
    if [[ ! -f "${PATCH_MARKER}" ]]; then
        log "Applying W&B patch to train.py..."
        python3 "${PROJ_ROOT}/patch_hificodec_wandb.py" "${HIFICODEC_TRAIN}"
        touch "${PATCH_MARKER}"
    else
        log "W&B patch already applied"
    fi
    
    export WANDB_PROJECT
    export WANDB_RUN_NAME
fi

# ── HiFiCodec training ─────────────────────────────────────────────────────────
log ""
log "=== Training HiFiCodec on FSD50K for ${EPOCHS} epochs ==="
log "Config      : ${CONFIG}"
log "Train list  : ${TRAIN_LST}"
log "Val list    : ${VAL_LST}"
log "Output      : ${CHECKPOINT_PATH}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTHONPATH="${PROJ_ROOT}/hificodec:${PYTHONPATH:-}"

# TRAIN_N_SAMPLES (passed by run_pipeline.py) caps meldataset.__len__; use
# it here so checkpoint_interval matches the actual steps-per-epoch, not the
# full-dataset count that wc -l would give.
N_TRAIN_FILE=$(wc -l < "${TRAIN_LST}")
BATCH_SIZE=8
N_TRAIN=${TRAIN_N_SAMPLES:-${N_TRAIN_FILE}}
[[ "${N_TRAIN}" -gt "${N_TRAIN_FILE}" ]] && N_TRAIN=${N_TRAIN_FILE}
STEPS_PER_EPOCH=$(( (N_TRAIN + BATCH_SIZE - 1) / BATCH_SIZE ))
[[ "${STEPS_PER_EPOCH}" -lt 1 ]] && STEPS_PER_EPOCH=1
CHECKPOINT_INTERVAL=${STEPS_PER_EPOCH}   # save once per epoch
VAL_INTERVAL=$((STEPS_PER_EPOCH * 5))    # validate every 5 epochs
log "Training files (file): ${N_TRAIN_FILE}"
log "Training files (eff) : ${N_TRAIN}  (TRAIN_N_SAMPLES env or file count)"
log "Steps per epoch      : ${STEPS_PER_EPOCH}"
log "Checkpoint interval  : ${CHECKPOINT_INTERVAL} steps (every epoch)"
log "Validation interval  : ${VAL_INTERVAL} steps (every 5 epochs)"

python3 "${HIFICODEC_TRAIN}" \
    --config "${CONFIG}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --input_training_file "${TRAIN_LST}" \
    --input_validation_file "${VAL_LST}" \
    --training_epochs "${EPOCHS}" \
    --checkpoint_interval "${CHECKPOINT_INTERVAL}" \
    --summary_interval 100 \
    --validation_interval "${VAL_INTERVAL}" \
    --stdout_interval ${STDOUT_INTERVAL:-50}

log ""
log "=== HiFiCodec training complete ==="
log "Checkpoint saved to: ${CHECKPOINT_PATH}"
log ""
log "Next steps:"
log "  1. Run encode scripts to tokenize test signals:"
log "     python tok_analysis/encode_hificodec_tokens.py --checkpoint ${CHECKPOINT_PATH}"
log "  2. Generate multi-codec combined report:"
log "     python tok_analysis/report_multi_codec_sensitivity.py --codecs encodec q2d2 hificodec dac speechtokenizer"

#!/bin/bash
# setup_codec_training.sh
#
# Sets up FSD50K dataset and configures codec training environment.
# Used for initial project setup; actual training is done via run_pipeline.py
#
# Usage:
#   bash setup_codec_training.sh
#
# Key env vars (set before running or edit the defaults below):
#   FSD50K_ROOT      Root dir of FSD50K audio. Script auto-detects from Encodec
#                    CSV if not set, then falls back to datasets/fsd50k_filelists.
#   DEVICE           cuda or cpu (default: cuda)
#   NUM_WORKERS      Dataloader workers (default: 8)
#   BATCH_SIZE       Training batch size (default: 12)
#
# Note: This script is primarily for initial setup.
# Use 'python run_pipeline.py' for actual codec training.

set -euo pipefail

# ── Parse flags ──────────────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --dac-only)      TRAIN_DAC=1; TRAIN_WAV=0 ;;
        --wav-only)      TRAIN_DAC=0; TRAIN_WAV=1 ;;
        *) echo "[WARN] Unknown argument: $arg" ;;
    esac
done

# ── Defaults ─────────────────────────────────────────────────────────────────────
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FSD50K_ROOT="${FSD50K_ROOT:-}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-12}"
TRAIN_DAC="${TRAIN_DAC:-1}"
TRAIN_WAV="${TRAIN_WAV:-1}"

CODECS_DIR="${PROJ_ROOT}/.external_codecs"
DAC_DIR="${CODECS_DIR}/descript-audio-codec"
WAV_DIR="${CODECS_DIR}/WavTokenizer"
LISTS_DIR="${PROJ_ROOT}/datasets/fsd50k_filelists"
RESULTS_DIR="${PROJ_ROOT}/results"

# ── Helpers ───────────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
err()  { echo "[ERROR] $*" >&2; exit 1; }
warn() { echo "[WARN]  $*" >&2; }
hr()   { echo "────────────────────────────────────────────────────────────"; }

require_cmd() { command -v "$1" &>/dev/null || err "Required command not found: $1 — please install it."; }

require_cmd git
require_cmd python3

mkdir -p "$CODECS_DIR" "$LISTS_DIR" "$RESULTS_DIR"

# ── Step 1: Detect FSD50K root ───────────────────────────────────────────────────
hr
log "Step 1: Locating FSD50K audio"

if [[ -z "$FSD50K_ROOT" ]]; then
    CONFIG="${PROJ_ROOT}/Encodec/config/config_multi_dataset.yaml"
    if [[ -f "$CONFIG" ]]; then
        CSV_PATH=$(grep 'fsd50k_train_csv' "$CONFIG" | awk -F"'" '{print $2}' | head -1)
        if [[ -n "$CSV_PATH" && -f "$CSV_PATH" ]]; then
            FIRST_FILE=$(head -1 "$CSV_PATH" | tr -d '\r')
            if [[ -f "$FIRST_FILE" ]]; then
                # CSV rows are bare paths; audio root is two levels up
                FSD50K_ROOT="$(dirname "$(dirname "$FIRST_FILE")")"
                log "Auto-detected FSD50K root from Encodec config: $FSD50K_ROOT"
            fi
        fi
    fi
fi

if [[ -z "$FSD50K_ROOT" || ! -d "$FSD50K_ROOT" ]]; then
    err "FSD50K_ROOT is not set or does not exist.
Set it before running:
  export FSD50K_ROOT=/path/to/fsd50k
  bash $0"
fi
log "FSD50K root: $FSD50K_ROOT"

# ── Step 2: Build filelists ───────────────────────────────────────────────────────
hr
log "Step 2: Building FSD50K filelists in $LISTS_DIR"

# Split strategy (8:1:1 using FSD50K's official dev/eval boundary):
#   dev_audio  → train  (~40,966 clips, all of dev)
#   eval_audio → randomly shuffled, first half → val, second half → test (~5,115 each)
# Fallback (no eval dir): pool all audio, randomly shuffle, split 80/10/10.

DEV_DIR=""
EVAL_DIR=""
for candidate in \
    "${FSD50K_ROOT}/FSD50K.dev_audio" \
    "${FSD50K_ROOT}/dev" \
    "${FSD50K_ROOT}/train"; do
    [[ -d "$candidate" ]] && { DEV_DIR="$candidate"; break; }
done
for candidate in \
    "${FSD50K_ROOT}/FSD50K.eval_audio" \
    "${FSD50K_ROOT}/eval" \
    "${FSD50K_ROOT}/test"; do
    [[ -d "$candidate" ]] && { EVAL_DIR="$candidate"; break; }
done

TRAIN_LIST="${LISTS_DIR}/fsd50k_train.txt"
VAL_LIST="${LISTS_DIR}/fsd50k_val.txt"
TEST_LIST="${LISTS_DIR}/fsd50k_test.txt"

if [[ -f "$TRAIN_LIST" && -f "$VAL_LIST" && -f "$TEST_LIST" ]]; then
    log "Filelists already exist — skipping generation."
else
    require_cmd shuf
    TMP=$(mktemp)

    if [[ -n "$DEV_DIR" && -n "$EVAL_DIR" ]]; then
        log "Found standard FSD50K layout — dev→train, eval→val+test (8:1:1)."

        # All dev clips → train (no clips withheld from training)
        find "$DEV_DIR" \( -name "*.wav" -o -name "*.flac" \) | sort > "$TRAIN_LIST"

        # Randomly shuffle eval, split exactly 50/50 → val / test
        find "$EVAL_DIR" \( -name "*.wav" -o -name "*.flac" \) | shuf > "$TMP"
        NEVAL=$(wc -l < "$TMP")
        NVAL=$(( NEVAL / 2 ))
        head -n "$NVAL"           "$TMP" > "$VAL_LIST"
        tail -n +"$((NVAL + 1))" "$TMP" > "$TEST_LIST"
    else
        log "No eval dir found — pooling all audio and splitting 80/10/10 randomly."
        find "$FSD50K_ROOT" \( -name "*.wav" -o -name "*.flac" \) | shuf > "$TMP"
        TOTAL=$(wc -l < "$TMP")
        NTRAIN=$(( TOTAL * 8 / 10 ))
        NVAL=$(( (TOTAL - NTRAIN) / 2 ))
        head -n "$NTRAIN"                          "$TMP" > "$TRAIN_LIST"
        tail -n +"$((NTRAIN + 1))" "$TMP" | head -n "$NVAL" > "$VAL_LIST"
        tail -n +"$((NTRAIN + NVAL + 1))"          "$TMP" > "$TEST_LIST"
    fi

    rm -f "$TMP"

    log "  train : $(wc -l < "$TRAIN_LIST") files"
    log "  val   : $(wc -l < "$VAL_LIST") files"
    log "  test  : $(wc -l < "$TEST_LIST") files"
    [[ $(wc -l < "$TRAIN_LIST") -eq 0 ]] && \
        err "No audio files found under FSD50K_ROOT=$FSD50K_ROOT"
fi

# Also build CSV versions (one path per row, no header) for Encodec-style tools
TRAIN_CSV="${LISTS_DIR}/fsd50k_train.csv"
VAL_CSV="${LISTS_DIR}/fsd50k_val.csv"
TEST_CSV="${LISTS_DIR}/fsd50k_test.csv"
[[ ! -f "$TRAIN_CSV" ]] && cp "$TRAIN_LIST" "$TRAIN_CSV"
[[ ! -f "$VAL_CSV" ]]   && cp "$VAL_LIST"   "$VAL_CSV"
[[ ! -f "$TEST_CSV" ]]  && cp "$TEST_LIST"  "$TEST_CSV"

# Audio directory used by DAC (it walks directories, not filelists)
FSD50K_TRAIN_DIR="${DEV_DIR:-$FSD50K_ROOT}"
FSD50K_VAL_DIR="${EVAL_DIR:-$FSD50K_ROOT}"

# ════════════════════════════════════════════════════════════════
# ── DAC ─────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════
if [[ "$TRAIN_DAC" == "1" ]]; then
    hr
    log "=== DAC ==="

    # 1. Clone
    if [[ ! -d "$DAC_DIR/.git" ]]; then
        log "Cloning descript-audio-codec..."
        git clone https://github.com/descriptinc/descript-audio-codec.git "$DAC_DIR"
    else
        log "DAC already cloned."
    fi

    # 2. Virtualenv + install
    DAC_VENV="${DAC_DIR}/venv_dac"
    if [[ ! -d "$DAC_VENV" ]]; then
        log "Creating DAC virtualenv..."
        python3 -m venv "$DAC_VENV"
    fi
    source "${DAC_VENV}/bin/activate"
    log "Installing DAC + dependencies (pip install -e \"[dev]\")..."
    pip install --quiet --upgrade pip
    pip install --quiet -e "${DAC_DIR}[dev]"
    deactivate

    # 3. Write FSD50K config
    DAC_SAVE="${RESULTS_DIR}/dac_fsd50k"
    mkdir -p "$DAC_SAVE"
    DAC_FSD_CONFIG="${DAC_DIR}/conf/fsd50k_24k.yml"

    log "Writing DAC config → $DAC_FSD_CONFIG"
    cat > "$DAC_FSD_CONFIG" << YAML
# DAC — FSD50K training config (24 kHz mono)
# Generated by setup_codec_training.sh

\$include:
  - conf/base.yml
  - conf/1gpu.yml

# 24 kHz to match project sample rate
DAC.sample_rate: 24000
Discriminator.sample_rate: 24000

# 8 kbps at 24 kHz: 12 codebooks × 1024-entry × log2(1024)/frame_rate
DAC.n_codebooks: 12
DAC.codebook_size: 1024
DAC.codebook_dim: 8
DAC.quantizer_dropout: 1.0

# FSD50K directories (AudioLoader walks subdirs recursively)
train/build_dataset.folders:
  fsd50k:
    - ${FSD50K_TRAIN_DIR}

val/build_dataset.folders:
  fsd50k:
    - ${FSD50K_VAL_DIR}

test/build_dataset.folders:
  fsd50k:
    - ${FSD50K_VAL_DIR}

# Training budget
batch_size: ${BATCH_SIZE}
num_workers: ${NUM_WORKERS}
num_iters: 100000
save_iters: [10000, 50000, 100000]
valid_freq: 1000
sample_freq: 5000
device: ${DEVICE}
YAML

    # 4. Launch
    log "Launching DAC training..."
    log "  config  : $DAC_FSD_CONFIG"
    log "  output  : $DAC_SAVE"

    source "${DAC_VENV}/bin/activate"
    pushd "$DAC_DIR" > /dev/null
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python scripts/train.py \
            --args.load "$DAC_FSD_CONFIG" \
            --save_path "$DAC_SAVE"
    popd > /dev/null
    deactivate

    log "✓ DAC training complete — checkpoints in $DAC_SAVE"
fi

# ════════════════════════════════════════════════════════════════
# ── WavTokenizer ────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════
if [[ "$TRAIN_WAV" == "1" ]]; then
    hr
    log "=== WavTokenizer ==="

    # 1. Clone
    if [[ ! -d "$WAV_DIR/.git" ]]; then
        log "Cloning WavTokenizer..."
        git clone https://github.com/jishengpeng/WavTokenizer.git "$WAV_DIR"
    else
        log "WavTokenizer already cloned."
    fi

    # 2. Virtualenv + install
    WAV_VENV="${WAV_DIR}/venv_wav"
    if [[ ! -d "$WAV_VENV" ]]; then
        log "Creating WavTokenizer virtualenv..."
        python3 -m venv "$WAV_VENV"
    fi
    source "${WAV_VENV}/bin/activate"
    log "Installing WavTokenizer dependencies..."
    pip install --quiet --upgrade pip
    pip install --quiet -r "${WAV_DIR}/requirements.txt"
    deactivate

    # 3. Copy filelists into WavTokenizer data dir
    mkdir -p "${WAV_DIR}/data"
    cp "$TRAIN_LIST" "${WAV_DIR}/data/fsd50k_train.txt"
    cp "$VAL_LIST"   "${WAV_DIR}/data/fsd50k_val.txt"
    log "Filelists copied to ${WAV_DIR}/data/"

    # 4. Build config from the first available base config
    WAV_SAVE="${RESULTS_DIR}/wavtokenizer_fsd50k"
    mkdir -p "$WAV_SAVE"
    WAV_FSD_CONFIG="${WAV_DIR}/configs/wavtokenizer_fsd50k_24k.yaml"

    BASE_CFG=$(find "${WAV_DIR}/configs" -name "*.yaml" | head -1)
    [[ -z "$BASE_CFG" ]] && err "No base config found in ${WAV_DIR}/configs — check the clone."
    log "Patching base config: $BASE_CFG → $WAV_FSD_CONFIG"

    source "${WAV_VENV}/bin/activate"
    python3 - <<PYEOF
import yaml, re, sys, copy

src = "${BASE_CFG}"
dst = "${WAV_FSD_CONFIG}"

with open(src) as f:
    raw = f.read()

# Strip custom YAML tags so PyYAML can parse
raw_clean = re.sub(r'!![\w./]+', '', raw)
try:
    cfg = yaml.safe_load(raw_clean) or {}
except Exception as e:
    print(f"WARN: could not parse {src} ({e}). Writing minimal config.", file=sys.stderr)
    cfg = {}

def deep_set(d, keys, val):
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = val

# Patch filelist paths — WavTokenizer expects filelist_scps nested under data
deep_set(cfg, ["data", "init_args", "filelist_scps", "train"], ["${WAV_DIR}/data/fsd50k_train.txt"])
deep_set(cfg, ["data", "init_args", "filelist_scps", "valid"], ["${WAV_DIR}/data/fsd50k_val.txt"])
deep_set(cfg, ["data", "init_args", "filelist_scps", "test"],  ["${WAV_DIR}/data/fsd50k_val.txt"])
deep_set(cfg, ["data", "init_args", "batch_size"], int("${BATCH_SIZE}"))
deep_set(cfg, ["data", "init_args", "num_workers"], int("${NUM_WORKERS}"))

# Patch trainer
deep_set(cfg, ["trainer", "default_root_dir"], "${WAV_SAVE}")
deep_set(cfg, ["trainer", "max_epochs"], 300)
deep_set(cfg, ["trainer", "accelerator"], "${DEVICE}")

with open(dst, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True)

print(f"Config written to {dst}")
PYEOF

    # 5. Launch
    log "Launching WavTokenizer training..."
    log "  config  : $WAV_FSD_CONFIG"
    log "  output  : $WAV_SAVE"

    pushd "$WAV_DIR" > /dev/null
    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python train.py fit --config "$WAV_FSD_CONFIG"
    popd > /dev/null
    deactivate

    log "✓ WavTokenizer training complete — checkpoints in $WAV_SAVE"
fi

# ════════════════════════════════════════════════════════════════
# ── Summary ─────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════
hr
log "Summary - FSD50K Dataset Setup Complete"
hr
log "  5 codecs configured for training via run_pipeline.py:"
log "    • DAC-FSQ        : Finite Scalar Quantization (24kHz)"
log "    • Q2D2           : 2D lattice quantization (24kHz)"
log "    • HiFiCodec      : Group RVQ (24kHz)"
log "    • SpeechTokenizer: RVQ adapted for general audio (16kHz)"
log "    • Encodec        : Meta RVQ (24kHz)"
log ""
log "  To train all models:"
log "    python run_pipeline.py"
log ""
log "  To train a single model:"
log "    python run_pipeline.py --model DAC-FSQ"
hr

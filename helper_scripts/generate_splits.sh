#!/bin/bash
# generate_splits.sh
#
# Generates FSD50K train/val/test splits and distributes filelists to all 5 models.
#
# Split strategy (8:1:1, FSD50K official boundary):
#   FSD50K.dev_audio  → train  (all ~40,966 clips)
#   FSD50K.eval_audio → randomly shuffled, 50/50 → val (~5,115) + test (~5,116)
# Fallback (no eval dir): pool all audio, random 80/10/10 split.
#
# Usage:
#   export FSD50K_ROOT=/path/to/fsd50k
#   bash generate_splits.sh

set -euo pipefail

PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FSD50K_ROOT="${FSD50K_ROOT:-}"

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
err()  { echo "[ERROR] $*" >&2; exit 1; }
require_cmd() { command -v "$1" &>/dev/null || err "Required command not found: $1"; }

require_cmd shuf
require_cmd find

# ── Locate FSD50K root ─────────────────────────────────────────────────────────
if [[ -z "$FSD50K_ROOT" ]]; then
    ENCODEC_CFG="${PROJ_ROOT}/Encodec/config/config_multi_dataset.yaml"
    if [[ -f "$ENCODEC_CFG" ]]; then
        CSV_PATH=$(grep 'fsd50k_train_csv' "$ENCODEC_CFG" | awk -F"'" '{print $2}' | head -1)
        if [[ -n "$CSV_PATH" && -f "$CSV_PATH" ]]; then
            FIRST_FILE=$(head -1 "$CSV_PATH" | tr -d '\r')
            [[ -f "$FIRST_FILE" ]] && FSD50K_ROOT="$(dirname "$(dirname "$FIRST_FILE")")"
        fi
    fi
fi

[[ -z "$FSD50K_ROOT" || ! -d "$FSD50K_ROOT" ]] && err \
    "FSD50K_ROOT is not set or does not exist.
  export FSD50K_ROOT=/path/to/fsd50k
  bash $0"

log "FSD50K root: $FSD50K_ROOT"

# ── Detect dev / eval dirs ─────────────────────────────────────────────────────
DEV_DIR="" EVAL_DIR=""
for c in "${FSD50K_ROOT}/FSD50K.dev_audio" "${FSD50K_ROOT}/dev" "${FSD50K_ROOT}/train"; do
    [[ -d "$c" ]] && { DEV_DIR="$c"; break; }
done
for c in "${FSD50K_ROOT}/FSD50K.eval_audio" "${FSD50K_ROOT}/eval" "${FSD50K_ROOT}/test"; do
    [[ -d "$c" ]] && { EVAL_DIR="$c"; break; }
done

# ── Generate splits ────────────────────────────────────────────────────────────
SPLITS_DIR="${PROJ_ROOT}/datasets"
mkdir -p "$SPLITS_DIR"

TRAIN_CSV="${SPLITS_DIR}/fsd50k_train.csv"
VAL_CSV="${SPLITS_DIR}/fsd50k_val.csv"
TEST_CSV="${SPLITS_DIR}/fsd50k_test.csv"

TMP=$(mktemp)
trap "rm -f '$TMP'" EXIT

if [[ -n "$DEV_DIR" && -n "$EVAL_DIR" ]]; then
    log "Standard FSD50K layout — dev→train, eval→val+test (8:1:1)"
    # Shuffle train list at creation time so all downstream trainers sample
    # from a randomized source ordering, even when they cap to first-N rows.
    find "$DEV_DIR"  \( -name "*.wav" -o -name "*.flac" \) | shuf > "$TRAIN_CSV"
    find "$EVAL_DIR" \( -name "*.wav" -o -name "*.flac" \) | shuf > "$TMP"
    NEVAL=$(wc -l < "$TMP")
    NVAL=$(( NEVAL / 2 ))
    head -n "$NVAL"           "$TMP" > "$VAL_CSV"
    tail -n +"$((NVAL + 1))" "$TMP" > "$TEST_CSV"
else
    log "No eval dir found — pooling all audio, random 80/10/10 split"
    find "$FSD50K_ROOT" \( -name "*.wav" -o -name "*.flac" \) | shuf > "$TMP"
    TOTAL=$(wc -l < "$TMP")
    NTRAIN=$(( TOTAL * 8 / 10 ))
    NVAL=$(( (TOTAL - NTRAIN) / 2 ))
    head -n "$NTRAIN"                            "$TMP" > "$TRAIN_CSV"
    tail -n +"$((NTRAIN + 1))" "$TMP" | head -n "$NVAL" > "$VAL_CSV"
    tail -n +"$((NTRAIN + NVAL + 1))"            "$TMP" > "$TEST_CSV"
fi

[[ $(wc -l < "$TRAIN_CSV") -eq 0 ]] && err "No audio files found — check FSD50K_ROOT"

log "  train : $(wc -l < "$TRAIN_CSV") files"
log "  val   : $(wc -l < "$VAL_CSV") files"
log "  test  : $(wc -l < "$TEST_CSV") files"

# ── Distribute to all 5 models ─────────────────────────────────────────────────
echo "────────────────────────────────────────────────────────────"
log "Distributing filelists to models..."

# DAC-FSQ: reads datasets/fsd50k_{train,val}.csv relative to project root — already in place
log "DAC-FSQ       ✓  datasets/fsd50k_train.csv (no copy needed)"

# HiFiCodec: train_fsd50k.sh converts CSV→LST at training time
log "HiFiCodec     ✓  train_fsd50k.sh converts CSV→LST at runtime (no copy needed)"

# Q2D2: expects txt filelists in Q2D2/data/
Q2D2_DATA="${PROJ_ROOT}/Q2D2/data"
mkdir -p "$Q2D2_DATA"
cp "$TRAIN_CSV" "${Q2D2_DATA}/fsd50k_train_files.txt"
cp "$VAL_CSV"   "${Q2D2_DATA}/fsd50k_val_files.txt"
log "Q2D2          ✓  ${Q2D2_DATA}/fsd50k_{train,val}_files.txt"

# SpeechTokenizer: expects txt filelists in {proj_root}/data/
ST_DATA="${PROJ_ROOT}/data"
mkdir -p "$ST_DATA"
cp "$TRAIN_CSV" "${ST_DATA}/fsd50k_train_files.txt"
cp "$VAL_CSV"   "${ST_DATA}/fsd50k_val_files.txt"
log "SpeechTokenizer ✓  ${ST_DATA}/fsd50k_{train,val}_files.txt"

# Encodec: config has hardcoded absolute paths — patch to current machine
ENCODEC_CFG="${PROJ_ROOT}/Encodec/config/config_multi_dataset.yaml"
if [[ -f "$ENCODEC_CFG" ]]; then
    sed -i \
        -e "s|fsd50k_train_csv:.*|fsd50k_train_csv: '${TRAIN_CSV}'|" \
        -e "s|fsd50k_valid_csv:.*|fsd50k_valid_csv: '${VAL_CSV}'|" \
        -e "s|fsd50k_test_csv:.*|fsd50k_test_csv: '${TEST_CSV}'|" \
        "$ENCODEC_CFG"
    log "Encodec       ✓  patched $ENCODEC_CFG"
else
    log "Encodec       ⚠  config not found at $ENCODEC_CFG — skipping"
fi

echo "────────────────────────────────────────────────────────────"
log "Done. Run: python run_pipeline.py --scale-test --models Encodec"

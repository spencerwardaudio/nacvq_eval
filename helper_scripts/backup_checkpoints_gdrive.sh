#!/bin/bash
# Backs up checkpoints + analysis data needed to re-run analysis elsewhere.
# One-time setup: rclone config  (name remote "gdrive", type: drive,
#   root_folder_id: 12UAvu0vJyNouhTX_1xbuold3WOZ_4iT2)

set -euo pipefail

REMOTE="gdrive"
FOLDER_ID="12UAvu0vJyNouhTX_1xbuold3WOZ_4iT2"
DEST="${REMOTE}:msc_proj_backup"
SRC="${1:-$(pwd)}"
LOG="rclone_backup_$(date +%Y%m%d_%H%M%S).log"

if ! command -v rclone &>/dev/null; then
    echo "ERROR: rclone not found. Install: curl https://rclone.org/install.sh | sudo bash"
    exit 1
fi

if ! rclone listremotes | grep -q "^${REMOTE}:"; then
    echo "ERROR: remote '${REMOTE}' not configured. Run: rclone config"
    exit 1
fi

echo "========================================"
echo "  msc_proj backup → Google Drive"
echo "========================================"
echo "  Source : ${SRC}"
echo "  Dest   : ${DEST}"
echo "  Log    : ${LOG}"
echo "========================================"
echo ""

RCLONE_OPTS=(
    --drive-root-folder-id "${FOLDER_ID}"
    --progress
    --transfers 6
    --checkers 12
    --drive-chunk-size 128M
    --fast-list
    --stats 15s
    --log-file "${LOG}"
    --log-level INFO
)

echo "==> Encodec checkpoints"
rclone copy "${SRC}/Encodec/outputs" "${DEST}/Encodec/outputs" \
    --include "*.pt" "${RCLONE_OPTS[@]}"

echo "==> Q2D2 checkpoints"
rclone copy "${SRC}/Q2D2/outputs" "${DEST}/Q2D2/outputs" \
    --include "*.ckpt" "${RCLONE_OPTS[@]}"

echo "==> HiFiCodec checkpoints + config"
rclone copy "${SRC}/hificodec/egs" "${DEST}/hificodec/egs" \
    --include "g_*" --include "*.json" "${RCLONE_OPTS[@]}"

echo "==> SpeechTokenizer checkpoints"
rclone copy "${SRC}/results" "${DEST}/results" \
    --include "SpeechTokenizer*.pt" "${RCLONE_OPTS[@]}"

echo "==> DAC-FSQ checkpoint"
rclone copy "${SRC}/descript-audio-codec/ckpt/fsd50k_fsq" \
    "${DEST}/descript-audio-codec/ckpt/fsd50k_fsq" "${RCLONE_OPTS[@]}"

echo "==> FSD50K dataset CSVs"
rclone copy "${SRC}/datasets" "${DEST}/datasets" \
    --include "fsd50k_*.csv" "${RCLONE_OPTS[@]}"

echo "==> Per-codec dataset filelists"
rclone copy "${SRC}/Q2D2/data" "${DEST}/Q2D2/data" \
    --include "fsd50k_*.txt" "${RCLONE_OPTS[@]}"
rclone copy "${SRC}/data" "${DEST}/data" \
    --include "fsd50k_*.txt" "${RCLONE_OPTS[@]}"
rclone copy "${SRC}/hificodec/egs/data" "${DEST}/hificodec/egs/data" \
    --include "fsd50k_*.lst" "${RCLONE_OPTS[@]}"

echo "==> EGFx audio pairs"
rclone copy "${SRC}/datasets/egfx" "${DEST}/datasets/egfx" "${RCLONE_OPTS[@]}"

echo "==> Encoded tokens (egfx_adaptive)"
rclone copy "${SRC}/datasets/audio_tokens/egfx_adaptive" \
    "${DEST}/datasets/audio_tokens/egfx_adaptive" "${RCLONE_OPTS[@]}"

echo "==> Analysis outputs"
rclone copy "${SRC}/datasets/analysis" "${DEST}/datasets/analysis" "${RCLONE_OPTS[@]}"

echo "==> DSP test recordings"
rclone copy "${SRC}/datasets/dsp_test_recordings" \
    "${DEST}/datasets/dsp_test_recordings" "${RCLONE_OPTS[@]}"

echo ""
echo "========================================"
echo "  Backup complete. Log: ${LOG}"
echo "========================================"

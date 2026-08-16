#!/bin/bash
# download_fsd50k.sh
#
# Downloads and extracts FSD50K from Zenodo (record 4060432).
# Fetches dev_audio, eval_audio, and ground_truth only.
#
# Usage:
#   bash download_fsd50k.sh                   # extracts to ./fsd50k/
#   bash download_fsd50k.sh /data/fsd50k      # custom path
#   FSD50K_DIR=/data/fsd50k bash download_fsd50k.sh

set -euo pipefail

ZENODO_RECORD="4060432"
ZENODO_API="https://zenodo.org/api/records/${ZENODO_RECORD}"
FSD50K_DIR="${1:-${FSD50K_DIR:-$(pwd)/fsd50k}}"

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
err()  { echo "[ERROR] $*" >&2; exit 1; }
require_cmd() { command -v "$1" &>/dev/null || err "Required: $1 — install it first"; }

require_cmd wget
require_cmd python3
require_cmd unzip

mkdir -p "$FSD50K_DIR"
DOWNLOAD_DIR="${FSD50K_DIR}/.downloads"
mkdir -p "$DOWNLOAD_DIR"

# ── Fetch file manifest from Zenodo API ───────────────────────────────────────
log "Fetching file list from Zenodo record ${ZENODO_RECORD}..."
MANIFEST="${DOWNLOAD_DIR}/zenodo_manifest.json"
wget -q -O "$MANIFEST" "${ZENODO_API}" || err "Cannot reach Zenodo API — check internet connection"

# Parse filenames + URLs for audio and ground-truth partitions only
mapfile -t FILE_ENTRIES < <(python3 - "$MANIFEST" <<'PYEOF'
import json, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
for entry in data.get("files", []):
    key  = entry.get("key", "")
    link = entry.get("links", {}).get("self", "")
    if any(k in key for k in ["dev_audio", "eval_audio", "ground_truth"]):
        print(f"{key}\t{link}")
PYEOF
)

[[ ${#FILE_ENTRIES[@]} -eq 0 ]] && err "No audio files found in manifest — Zenodo structure may have changed"

log "Files to download:"
for entry in "${FILE_ENTRIES[@]}"; do
    fname=$(echo "$entry" | cut -f1)
    log "  $fname"
done

# ── Download each file (resume-capable with -c) ────────────────────────────────
for entry in "${FILE_ENTRIES[@]}"; do
    fname=$(echo "$entry" | cut -f1)
    url=$(echo "$entry"   | cut -f2)
    dest="${DOWNLOAD_DIR}/${fname}"
    if [[ -f "$dest" ]]; then
        log "  Already downloaded: $fname — skipping"
    else
        log "Downloading $fname ..."
        wget -c --show-progress -O "$dest" "$url" || err "Download failed: $fname"
    fi
done

# ── Extract archives ───────────────────────────────────────────────────────────
# unzip handles split archives automatically when all parts (.z01/.z02/...) are co-located
for name in eval_audio dev_audio ground_truth; do
    zip_path="${DOWNLOAD_DIR}/FSD50K.${name}.zip"
    out_dir="${FSD50K_DIR}/FSD50K.${name}"
    if [[ -d "$out_dir" ]]; then
        log "  Already extracted: FSD50K.${name} — skipping"
    elif [[ -f "$zip_path" ]]; then
        log "Extracting FSD50K.${name} ..."
        unzip -q -o "$zip_path" -d "$FSD50K_DIR" && log "  ✓ ${name} extracted"
    else
        log "  ⚠ ${name} zip not found in downloads — skipping"
    fi
done

# ── Verify result ──────────────────────────────────────────────────────────────
echo "────────────────────────────────────────────────────────────"
ALL_OK=1
for expected in "FSD50K.dev_audio" "FSD50K.eval_audio"; do
    dir="${FSD50K_DIR}/${expected}"
    if [[ -d "$dir" ]]; then
        count=$(find "$dir" \( -name "*.wav" -o -name "*.flac" \) | wc -l)
        log "  ${expected}: ${count} audio files"
    else
        log "  ⚠ NOT FOUND: ${dir}"
        ALL_OK=0
    fi
done

if [[ "$ALL_OK" -eq 0 ]]; then
    err "One or more expected directories are missing. Check extraction errors above."
fi

# Remove zip files only after extraction is verified — saves ~25 GB
log "Removing downloaded zip files..."
rm -rf "$DOWNLOAD_DIR"
log "  ✓ $DOWNLOAD_DIR removed"

echo "────────────────────────────────────────────────────────────"
log "Download complete. FSD50K root: $FSD50K_DIR"
log ""
log "Next: generate train/val/test splits for all 5 models:"
log "  export FSD50K_ROOT=${FSD50K_DIR}"
log "  bash generate_splits.sh"

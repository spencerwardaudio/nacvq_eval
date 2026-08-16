#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

ENABLE_FREE_SPACE_GUARD="${ENABLE_FREE_SPACE_GUARD:-1}"
FREE_SPACE_GUARD_GB="${FREE_SPACE_GUARD_GB:-10}"

ensure_free_space() {
  local stage="$1"
  if [[ "$ENABLE_FREE_SPACE_GUARD" != "1" ]]; then
    return
  fi

  local avail_kb
  avail_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
  local min_kb=$((FREE_SPACE_GUARD_GB * 1024 * 1024))
  if [[ -z "$avail_kb" ]]; then
    echo "Failed to read free space for guard check at stage: $stage" >&2
    exit 1
  fi
  if (( avail_kb < min_kb )); then
    local avail_gb
    avail_gb="$(awk -v kb="$avail_kb" 'BEGIN { printf "%.2f", kb / 1024 / 1024 }')"
    echo "Free-space guard tripped at stage: $stage" >&2
    echo "Available: ${avail_gb} GB; required minimum: ${FREE_SPACE_GUARD_GB} GB" >&2
    echo "Aborting to prevent disk exhaustion." >&2
    exit 1
  fi
}

ensure_free_space "startup"

python3 -m pip install --upgrade pip
python3 -m pip install -r "$ROOT/requirements.txt"
ensure_free_space "post_dependency_install"

# Dataset acquisition strategy:
# 1. Try HEAR preprocessed tasks (recommended, standardized)
# 2. Fall back to direct downloads if HEAR unavailable

if [[ "${USE_HEAR_TASKS:-true}" == "true" ]]; then
  HEAR_TASK_STAGE="${HEAR_TASK_STAGE:-A}"
  case "$HEAR_TASK_STAGE" in
    A|a)
      HEAR_TASKS="${HEAR_TASKS:-esc50 vocal_imitation crema_d}"
      ;;
    B|b)
      HEAR_TASKS="${HEAR_TASKS:-esc50 vocal_imitation crema_d speech_commands}"
      ;;
    C|c)
      HEAR_TASKS="${HEAR_TASKS:-esc50 vocal_imitation crema_d speech_commands libricount}"
      ;;
    D|d)
      HEAR_TASKS="${HEAR_TASKS:-esc50 vocal_imitation crema_d speech_commands libricount nsynth_pitch}"
      ;;
    *)
      echo "Invalid HEAR_TASK_STAGE=$HEAR_TASK_STAGE (expected A/B/C/D)" >&2
      exit 1
      ;;
  esac

  echo "Downloading HEAR benchmark tasks (standardized, 48kHz)..."
  ensure_free_space "before_hear_download"
  python3 "$ROOT/tok_analysis/download_hear_tasks.py" \
    --output-dir "$ROOT/datasets/hear_tasks" \
    --tasks ${HEAR_TASKS}
  
  # Create symlink for backward compatibility
  if [[ ! -e "$ROOT/datasets/audio_files" ]] || [[ -L "$ROOT/datasets/audio_files" ]]; then
    ln -sfn hear_tasks "$ROOT/datasets/audio_files"
    echo "Created symlink: datasets/audio_files → hear_tasks"
  fi
else
  echo "Downloading evaluation dataset samples (fallback mode, up to 20 per dataset)..."
  ensure_free_space "before_fallback_download"
  python3 "$ROOT/tok_analysis/download_eval_datasets.py" \
    --output-dir "$ROOT/datasets/audio_files" \
    --n-samples 20
fi
ensure_free_space "post_dataset_download"

WAVTOKENIZER_REPO="${WAVTOKENIZER_REPO:-$ROOT/.external/WavTokenizer}"
mkdir -p "$(dirname "$WAVTOKENIZER_REPO")"

# Install git-lfs via conda if not present
if ! command -v git-lfs >/dev/null 2>&1; then
  if command -v conda >/dev/null 2>&1; then
    echo "git-lfs not found; installing via conda..."
    conda install -y -c conda-forge git-lfs
  fi
fi

if [[ ! -d "$WAVTOKENIZER_REPO/.git" ]]; then
  if command -v git-lfs >/dev/null 2>&1; then
    ensure_free_space "before_wavtokenizer_clone"
    git lfs install
    git clone https://huggingface.co/novateur/WavTokenizer "$WAVTOKENIZER_REPO"
  else
    echo "git-lfs not found; skipping WavTokenizer clone. Set WAVTOKENIZER_REPO manually to an existing checkout." >&2
  fi
fi

# Install opus-tools — try apt, then conda
if ! command -v opusenc >/dev/null 2>&1 || ! command -v opusdec >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    sudo apt-get update -q && sudo apt-get install -y opus-tools
  elif command -v conda >/dev/null 2>&1; then
    echo "apt unavailable; installing opus-tools via conda..."
    conda install -y -c conda-forge opus-tools
  else
    echo "opus-tools not installed and no package manager available; opus codec will be skipped." >&2
  fi
fi

echo "Setup complete."
echo "WavTokenizer repo: $WAVTOKENIZER_REPO"
if command -v opusenc >/dev/null 2>&1; then
  echo "opusenc: $(command -v opusenc)"
fi
if command -v opusdec >/dev/null 2>&1; then
  echo "opusdec: $(command -v opusdec)"
fi

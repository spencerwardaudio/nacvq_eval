#!/bin/bash
# launch_pipeline.sh
#
# Full unattended pipeline: download FSD50K → split → train (n=200) → EGFx → analysis.
# Run this after setup_env.sh completes.
#
# Default mode: tmux (recommended on RunPod — reattach anytime to see live output).
# Fallback mode: nohup (no tmux required — output written to pipeline.log only).
#
# Usage:
#   bash launch_pipeline.sh [/path/to/fsd50k]           # tmux mode (default)
#   bash launch_pipeline.sh [/path/to/fsd50k] --nohup   # nohup fallback
#
# tmux workflow:
#   Detach   : Ctrl+B, D
#   Reattach : tmux attach -t pipeline
#   Stop     : tmux kill-session -t pipeline
#
# nohup workflow:
#   Monitor  : tail -f pipeline.log
#   Stop     : kill $(cat pipeline.pid)
#
# FSD50K destination defaults to ./fsd50k if not given.

# Any unhandled error exits immediately rather than silently continuing
set -euo pipefail

# Resolve the repo root regardless of where the script is called from
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# FSD50K path: first positional arg → $FSD50K_DIR env var → default ./fsd50k
FSD50K_DIR="${1:-${FSD50K_DIR:-${PROJ_ROOT}/fsd50k}}"
USE_NOHUP=0

# Parse flags — accept --nohup anywhere in the argument list
for arg in "$@"; do
    [[ "$arg" == "--nohup" ]] && USE_NOHUP=1
done

# The full pipeline is stored as a string so both tmux and nohup can run the same steps.
# Sequence:
#   1. download_fsd50k.sh   — fetch FSD50K audio archive (~21 GB) to FSD50K_DIR
#   2. generate_splits.sh   — create train/val/test CSV splits from the downloaded files
#   3. run_pipeline.py      — sanity-test (2 epochs) then full 200-step/epoch training for all 5 codecs
#   4. run_egfx_adaptive.sh — encode EGFx guitar-effect pairs and score with geometric metrics
#   5. run_analysis.py      — encode sinusoid test signals, compute Jacobians, generate sensitivity PDFs and CSVs
PIPELINE_SCRIPT="
set -euo pipefail
cd '$PROJ_ROOT'
source .venv/bin/activate
bash helper_scripts/download_fsd50k.sh '$FSD50K_DIR' &&
export FSD50K_ROOT='$FSD50K_DIR' &&
bash helper_scripts/generate_splits.sh &&
python helper_scripts/run_pipeline.py --scale-test --stages 200 \
  --models HiFiCodec SpeechTokenizer Encodec Q2D2 DAC-FSQ &&
bash helper_scripts/run_egfx_adaptive.sh &&
python helper_scripts/run_analysis.py --yes --skip-egfx &&
echo \"[\$(date '+%H:%M:%S')] Pipeline complete.\"
"

if [[ "$USE_NOHUP" -eq 1 ]]; then
    LOG="${PROJ_ROOT}/pipeline.log"
    PID="${PROJ_ROOT}/pipeline.pid"
    echo "[$(date '+%H:%M:%S')] Launching pipeline via nohup — log: $LOG"
    # Detach from terminal; stdout+stderr go to pipeline.log; PID saved for stop/status checks
    nohup bash -c "$PIPELINE_SCRIPT" > "$LOG" 2>&1 &
    echo $! > "$PID"
    echo "PID $(cat "$PID") saved to pipeline.pid"
    echo ""
    echo "Monitor  : tail -f pipeline.log"
    echo "Status   : ps -p \$(cat pipeline.pid)"
    echo "Stop     : kill \$(cat pipeline.pid)"
else
    if ! command -v tmux &>/dev/null; then
        echo "[WARN] tmux not found — falling back to nohup. Install tmux or pass --nohup to suppress this warning."
        # Re-invoke self with --nohup flag rather than duplicating the launch logic
        exec bash "$0" "$FSD50K_DIR" --nohup
    fi
    SESSION="pipeline"
    # Kill any leftover session with the same name so this is safe to re-run
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    echo "[$(date '+%H:%M:%S')] Launching pipeline in tmux session '$SESSION'"
    # Detach (-d) so the terminal is returned immediately; session stays alive after SSH disconnect
    tmux new-session -d -s "$SESSION" bash -c "$PIPELINE_SCRIPT"
    echo ""
    echo "Reattach : tmux attach -t $SESSION"
    echo "Stop     : tmux kill-session -t $SESSION"
fi

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

INPUT_PATH="${BENCHMARK_INPUT:-$ROOT/datasets/audio_files}"
OUTPUT_DIR="${BENCHMARK_OUTPUT_DIR:-$ROOT/datasets/analysis/benchmark_runs}"
OUTPUT_CSV="${BENCHMARK_OUTPUT_CSV:-$ROOT/datasets/analysis/benchmark_metrics.csv}"
ANALYSIS_ROOT="${BENCHMARK_ANALYSIS_ROOT:-$ROOT/datasets/analysis}"
FINAL_PLOTS_DIR="${BENCHMARK_FINAL_PLOTS_DIR:-$ROOT/datasets/analysis/final_plots}"
DEVICE="${BENCHMARK_DEVICE:-cuda}"
BITRATES="${BENCHMARK_BITRATES:-24 12 6 3 1.5}"
CODECS="${BENCHMARK_CODECS:-encodec dac semanticodec opus wavtokenizer}"
ENCODEC_MODEL_NAME="${ENCODEC_MODEL_NAME:-multi_dataset_encodec}"
ENCODEC_CHECKPOINT="${ENCODEC_CHECKPOINT:-$ROOT/Encodec/checkpoints_multi_dataset/bs16_cut24000_length32000_epoch335_lr0.0003.pt}"
WAVTOKENIZER_REPO="${WAVTOKENIZER_REPO:-$ROOT/.external/WavTokenizer}"
WAVTOKENIZER_CONFIG="${WAVTOKENIZER_CONFIG:-}"
WAVTOKENIZER_CHECKPOINT="${WAVTOKENIZER_CHECKPOINT:-}"
WAVTOKENIZER_BANDWIDTH_ID="${WAVTOKENIZER_BANDWIDTH_ID:-0}"
WAVTOKENIZER_SAMPLE_RATE="${WAVTOKENIZER_SAMPLE_RATE:-24000}"
WAVTOKENIZER_VOCAB_SIZE="${WAVTOKENIZER_VOCAB_SIZE:-4096}"
ASR_COMMAND="${BENCHMARK_ASR_COMMAND:-}"
CLASSIFIER_COMMAND="${BENCHMARK_CLASSIFIER_COMMAND:-}"
MANIFEST_CSV="${BENCHMARK_MANIFEST_CSV:-}"
GENERATE_MANIFEST="${BENCHMARK_GENERATE_MANIFEST:-1}"
TRANSCRIPTS_CSV="${BENCHMARK_TRANSCRIPTS_CSV:-}"
BENCHMARK_STORAGE_SAFE="${BENCHMARK_STORAGE_SAFE:-1}"
DELETE_COMPLETED_TASK_DATA="${DELETE_COMPLETED_TASK_DATA:-1}"
CLEANUP_TASK_OUTPUTS="${CLEANUP_TASK_OUTPUTS:-1}"
BENCHMARK_TASKS="${BENCHMARK_TASKS:-}"
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

mkdir -p "$ANALYSIS_ROOT" "$FINAL_PLOTS_DIR" "$OUTPUT_DIR"
ensure_free_space "startup"

if [[ -z "$WAVTOKENIZER_CONFIG" && -d "$WAVTOKENIZER_REPO" ]]; then
  WAVTOKENIZER_CONFIG="$(find "$WAVTOKENIZER_REPO" -type f \( -name '*.yaml' -o -name '*.yml' \) | head -n 1 || true)"
fi
if [[ -z "$WAVTOKENIZER_CHECKPOINT" && -d "$WAVTOKENIZER_REPO" ]]; then
  WAVTOKENIZER_CHECKPOINT="$(find "$WAVTOKENIZER_REPO" -type f -name '*.ckpt' | head -n 1 || true)"
fi

run_benchmark_once() {
  local manifest_path="$1"
  local input_path="$2"
  local run_output_dir="$3"
  local run_output_csv="$4"

  cmd=(python3 "$ROOT/tok_analysis/benchmark_codecs.py")
  if [[ -n "$manifest_path" ]]; then
    cmd+=(--manifest-csv "$manifest_path")
  else
    cmd+=(--input "$input_path")
  fi
  cmd+=(
    --codecs ${CODECS}
    --bitrates ${BITRATES}
    --device "$DEVICE"
    --encodec-model-name "$ENCODEC_MODEL_NAME"
    --encodec-checkpoint "$ENCODEC_CHECKPOINT"
    --skip-missing-codecs
    --output-dir "$run_output_dir"
    --output-csv "$run_output_csv"
  )
  if [[ -n "$WAVTOKENIZER_CONFIG" ]]; then
    cmd+=(--wavtokenizer-repo "$WAVTOKENIZER_REPO" --wavtokenizer-config "$WAVTOKENIZER_CONFIG")
  fi
  if [[ -n "$WAVTOKENIZER_CHECKPOINT" ]]; then
    cmd+=(--wavtokenizer-checkpoint "$WAVTOKENIZER_CHECKPOINT")
  fi
  cmd+=(--wavtokenizer-bandwidth-id "$WAVTOKENIZER_BANDWIDTH_ID" --wavtokenizer-sample-rate "$WAVTOKENIZER_SAMPLE_RATE" --wavtokenizer-vocab-size "$WAVTOKENIZER_VOCAB_SIZE")
  if [[ -n "$ASR_COMMAND" ]]; then
    cmd+=(--asr-command "$ASR_COMMAND")
  fi
  if [[ -n "$CLASSIFIER_COMMAND" ]]; then
    cmd+=(--classifier-command "$CLASSIFIER_COMMAND")
  fi

  "${cmd[@]}"
}

if [[ "$BENCHMARK_STORAGE_SAFE" == "1" && -z "$MANIFEST_CSV" ]]; then
  rm -f "$OUTPUT_CSV"

  if [[ -n "$BENCHMARK_TASKS" ]]; then
    read -r -a TASK_LIST <<< "$BENCHMARK_TASKS"
  else
    mapfile -t TASK_LIST < <(find "$INPUT_PATH" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" | sort)
  fi

  if [[ ${#TASK_LIST[@]} -eq 0 ]]; then
    echo "No task directories found under $INPUT_PATH" >&2
    exit 1
  fi

  for task in "${TASK_LIST[@]}"; do
    ensure_free_space "before_task_${task}"
    task_input="$INPUT_PATH/$task"
    if [[ ! -d "$task_input" ]]; then
      echo "Skipping missing task dir: $task_input"
      continue
    fi

    task_manifest="$ANALYSIS_ROOT/benchmark_manifest_${task}.csv"
    task_csv="$ANALYSIS_ROOT/benchmark_metrics_${task}.csv"
    task_output_dir="$OUTPUT_DIR/$task"

    manifest_cmd=(python3 "$ROOT/tok_analysis/generate_benchmark_manifest.py" --input "$task_input" --output-csv "$task_manifest")
    if [[ -n "$TRANSCRIPTS_CSV" ]]; then
      manifest_cmd+=(--transcripts-csv "$TRANSCRIPTS_CSV")
    fi
    "${manifest_cmd[@]}"

    if [[ ! -s "$task_manifest" ]]; then
      echo "Skipping empty task manifest for $task" >&2
      continue
    fi

    echo "Running benchmark for task: $task"
    run_benchmark_once "$task_manifest" "" "$task_output_dir" "$task_csv"

    if [[ ! -s "$task_csv" ]]; then
      echo "Task CSV missing or empty: $task_csv" >&2
      exit 1
    fi

    if [[ ! -f "$OUTPUT_CSV" ]]; then
      cp "$task_csv" "$OUTPUT_CSV"
    else
      tail -n +2 "$task_csv" >> "$OUTPUT_CSV"
    fi

    rm -f "$task_manifest" "$task_csv"

    if [[ "$CLEANUP_TASK_OUTPUTS" == "1" ]]; then
      rm -rf "$task_output_dir"
    fi

    if [[ "$DELETE_COMPLETED_TASK_DATA" == "1" ]]; then
      rm -rf "$task_input"
      echo "Deleted completed task input: $task_input"
    fi

    ensure_free_space "after_task_${task}"
  done
else
  ensure_free_space "before_full_run"
  if [[ -z "$MANIFEST_CSV" && "$GENERATE_MANIFEST" == "1" ]]; then
    MANIFEST_CSV="$ROOT/datasets/analysis/benchmark_manifest.csv"
    manifest_cmd=(python3 "$ROOT/tok_analysis/generate_benchmark_manifest.py" --input "$INPUT_PATH" --output-csv "$MANIFEST_CSV")
    if [[ -n "$TRANSCRIPTS_CSV" ]]; then
      manifest_cmd+=(--transcripts-csv "$TRANSCRIPTS_CSV")
    fi
    "${manifest_cmd[@]}"
  fi

  if [[ -n "$MANIFEST_CSV" && ! -s "$MANIFEST_CSV" ]]; then
    echo "Manifest missing or empty: $MANIFEST_CSV" >&2
    exit 1
  fi

  run_benchmark_once "$MANIFEST_CSV" "$INPUT_PATH" "$OUTPUT_DIR" "$OUTPUT_CSV"
fi

if [[ ! -s "$OUTPUT_CSV" ]]; then
  echo "Benchmark CSV missing or empty: $OUTPUT_CSV" >&2
  exit 1
fi

ensure_free_space "before_plots"
python3 "$ROOT/tok_analysis/final_benchmark_plots.py" \
  --analysis-root "$ANALYSIS_ROOT" \
  --metrics-csv "$OUTPUT_CSV" \
  --output-dir "$FINAL_PLOTS_DIR"

required_plots=(
  "$FINAL_PLOTS_DIR/reconstruction_quality_vs_bitrate.png"
  "$FINAL_PLOTS_DIR/semantic_robustness_vs_bitrate.png"
  "$FINAL_PLOTS_DIR/stability_heatmap_phase_selfampfreq.png"
  "$FINAL_PLOTS_DIR/lowfreq_diagnostic_lt200hz.png"
)

for plot_path in "${required_plots[@]}"; do
  if [[ ! -s "$plot_path" ]]; then
    echo "Expected final plot missing or empty: $plot_path" >&2
    exit 1
  fi
done

echo "Benchmark complete. Final plots: $FINAL_PLOTS_DIR"

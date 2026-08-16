#!/bin/bash

# Adaptive EGFX pipeline - runs with whatever codec checkpoints are available

# Auto-discovers best checkpoints and skips missing codecs

# Usage: bash run_egfx_adaptive.sh [MAX_PER_CATEGORY]

  

set -e

  

MAX_PER_CATEGORY="${1:-25}" # Default 75 samples (25x3 categories)

SEED=42

DEVICE="cuda"

  

echo "========================================"

echo "EGFX Adaptive Test"

echo "========================================"

echo "Samples per category: $MAX_PER_CATEGORY"

echo ""

  

# Configuration

EGFX_DIR="datasets/egfx"

PAIRS_JSON="datasets/egfx/effect_pairs.json"

SAMPLED_PAIRS="datasets/egfx/effect_pairs_adaptive.json"

TOKENS_DIR="datasets/audio_tokens/egfx_adaptive"

METRICS_JSON="datasets/analysis/egfx_metrics_adaptive.json"

REPORT_PDF="datasets/analysis/egfx_report_adaptive.pdf"

  

# ============================================================================

# Auto-discover available codec checkpoints

# ============================================================================

  

AVAILABLE_CODECS=()

CODEC_ARGS=()

  

# --- Encodec ---

echo -n "Checking Encodec... "

ENCODEC_CKPT=$(find Encodec/outputs -type f -name '*.pt' 2>/dev/null | \
               grep -E 'epoch[0-9]+.*\.pt$' | sort -V | tail -1)

if [[ -n "$ENCODEC_CKPT" && -f "$ENCODEC_CKPT" ]]; then

    AVAILABLE_CODECS+=("encodec")

    CODEC_ARGS+=("--encodec-checkpoint" "$ENCODEC_CKPT")

    echo "[OK] Found: $ENCODEC_CKPT"

else

    echo "[MISSING] Not found"

fi

  

# --- Q2D2 ---

echo -n "Checking Q2D2... "

Q2D2_CKPT=$(find Q2D2/outputs -type f -name '*.ckpt' 2>/dev/null | sort -V | tail -1)

if [[ -n "$Q2D2_CKPT" && -f "$Q2D2_CKPT" ]]; then

    AVAILABLE_CODECS+=("q2d2")

    CODEC_ARGS+=("--q2d2-checkpoint" "$Q2D2_CKPT")

    echo "[OK] Found: $Q2D2_CKPT"

else

    echo "[MISSING] Not found"

fi

  

# --- HiFiCodec ---

echo -n "Checking HiFiCodec... "

HIFICODEC_CKPT=$(find hificodec/egs -type f -name 'g_*' 2>/dev/null | sort -V | tail -1)

if [[ -n "$HIFICODEC_CKPT" && -f "$HIFICODEC_CKPT" ]]; then

    # Get directory containing the checkpoint

    HIFICODEC_DIR=$(dirname "$HIFICODEC_CKPT")

    AVAILABLE_CODECS+=("hificodec")

    CODEC_ARGS+=("--hificodec-checkpoint" "$HIFICODEC_DIR")

    echo "[OK] Found: $HIFICODEC_CKPT"

else

    echo "[MISSING] Not found"

fi

  

# --- SpeechTokenizer ---

echo -n "Checking SpeechTokenizer... "

SPEECHTOKENIZER_CKPT=$(find results -type f -name 'SpeechTokenizer*.pt' 2>/dev/null | \
                        grep -v 'optimizer' | sort -V | tail -1)

if [[ -n "$SPEECHTOKENIZER_CKPT" && -f "$SPEECHTOKENIZER_CKPT" ]]; then

    AVAILABLE_CODECS+=("speechtokenizer")

    CODEC_ARGS+=("--speechtokenizer-checkpoint" "$SPEECHTOKENIZER_CKPT")

    echo "[OK] Found: $SPEECHTOKENIZER_CKPT"

else

    echo "[MISSING] Not found"

fi

  

# --- DAC-FSQ ---

echo -n "Checking DAC-FSQ... "

DAC_FSQ_CKPT="descript-audio-codec/ckpt/fsd50k_fsq"

if [[ -d "$DAC_FSQ_CKPT" ]] && [[ -d "$DAC_FSQ_CKPT/latest" || -f "$DAC_FSQ_CKPT/weights.pth" ]]; then

    AVAILABLE_CODECS+=("dac_fsq")

    CODEC_ARGS+=("--dac-fsq-checkpoint" "$DAC_FSQ_CKPT")

    echo "[OK] Found: $DAC_FSQ_CKPT"

else

    echo "[MISSING] Not found"

fi

  

echo ""

echo "========================================"

echo "Available codecs: ${#AVAILABLE_CODECS[@]}"

echo "  ${AVAILABLE_CODECS[*]}"

echo "========================================"

  

if [[ ${#AVAILABLE_CODECS[@]} -eq 0 ]]; then

    echo ""

    echo "ERROR: No codec checkpoints found!"

    echo ""

    echo "Searched in:"

    echo "  - Encodec/outputs/**/*.pt"

    echo "  - Q2D2/outputs/**/*.ckpt"

    echo "  - hificodec/egs/**/g_*"

    echo "  - results/**/SpeechTokenizer*.pt"

    echo "  - descript-audio-codec/ckpt/fsd50k_fsq/"

    echo ""

    exit 1

fi

  

echo ""

  

# ============================================================================

# Download/prepare EGFX dataset

# ============================================================================

  

if [[ ! -d "$EGFX_DIR" ]] || [[ ! -f "$PAIRS_JSON" ]]; then

    echo "Step 1/4: Downloading EGFx dataset..."

    python tok_analysis/egfx_download.py --output-dir "$EGFX_DIR"

    echo ""

    echo "Step 2/4: Preparing effect pairs..."

    python tok_analysis/egfx_prepare_pairs.py \
        --egfx-dir "$EGFX_DIR" \
        --output "$PAIRS_JSON"

else

    echo "Step 1-2/4: Dataset already prepared"

fi

  

# ============================================================================

# Encode through available codecs

# ============================================================================

  

echo ""

echo "Step 3/4: Encoding through ${#AVAILABLE_CODECS[@]} codec(s)..."

echo ""

  

# Build codec list for --codecs argument

CODEC_LIST=$(IFS=' '; echo "${AVAILABLE_CODECS[*]}")

  

python tok_analysis/egfx_encode.py \
    --pairs "$PAIRS_JSON" \
    --categories distortion modulation time_based \
    --max-per-category "$MAX_PER_CATEGORY" \
    --seed "$SEED" \
    --write-sampled-pairs "$SAMPLED_PAIRS" \
    --codecs $CODEC_LIST \
    "${CODEC_ARGS[@]}" \
    --output-dir "$TOKENS_DIR" \
    --device "$DEVICE"

  

# ============================================================================

# Compute metrics and generate report

# ============================================================================

  

echo ""

echo "Step 4/4: Computing geometric metrics..."

python tok_analysis/egfx_metrics.py \
    --tokens-dir "$TOKENS_DIR" \
    --output "$METRICS_JSON"

  

echo ""

echo "Step 5/5: Generating analysis report..."

python tok_analysis/egfx_analyze.py \
    --metrics "$METRICS_JSON" \
    --sampled-pairs "$SAMPLED_PAIRS" \
    --output "$REPORT_PDF"

  

# ============================================================================

# Summary

# ============================================================================

  

echo ""

echo "========================================"

echo "[OK] Pipeline complete!"

echo "========================================"

echo ""

echo "Codecs tested: ${AVAILABLE_CODECS[*]}"

echo "Total samples: ~$((MAX_PER_CATEGORY * 3))"

echo ""

echo "Results:"

echo "  - Sampled pairs: $SAMPLED_PAIRS"

echo "  - Token outputs: $TOKENS_DIR"

echo "  - Metrics JSON:  $METRICS_JSON"

echo "  - Report PDF:    $REPORT_PDF"

echo ""

echo "Open the PDF to view comparative analysis."

echo ""
#!/bin/bash
# Run EGFX pipeline with all 5 codecs on 100-sample subset
# Usage: bash run_egfx_100samples.sh

set -e  # Exit on error

echo "========================================"
echo "EGFX 100-Sample Test (All 5 Codecs)"
echo "========================================"

# Configuration
EGFX_DIR="datasets/egfx"
PAIRS_JSON="datasets/egfx/effect_pairs.json"
SAMPLED_PAIRS="datasets/egfx/effect_pairs_100samples.json"
TOKENS_DIR="datasets/audio_tokens/egfx_100samples"
METRICS_JSON="datasets/analysis/egfx_metrics_100samples.json"
REPORT_PDF="datasets/analysis/egfx_report_100samples.pdf"

# Checkpoint paths (verified 2026-07-29)
ENCODEC_CKPT="Encodec/outputs/2026-07-28/17-51-40/checkpoints_multi_dataset/bs12_cut72000_length12000_epoch50_disc_lr0.0003.pt"
Q2D2_CKPT="Q2D2/outputs/lightning_logs/version_6/checkpoints/last.ckpt"
HIFICODEC_CKPT="${HIFICODEC_CKPT:-hificodec/egs/hificodec_fsd50k}"
SPEECHTOKENIZER_CKPT="results/speechtokenizer_fsd50k/SpeechTokenizer_best_dev.pt"
DAC_FSQ_CKPT="descript-audio-codec/ckpt/fsd50k_fsq"

# Sample size
MAX_PER_CATEGORY=25  # 4 categories × 25 = 100 samples
SEED=42

# Device
DEVICE="cuda"

echo ""
echo "Checkpoint verification:"
echo "  Encodec:         $ENCODEC_CKPT"
echo "  Q2D2:            $Q2D2_CKPT"
echo "  HiFiCodec:       $HIFICODEC_CKPT"
echo "  SpeechTokenizer: $SPEECHTOKENIZER_CKPT"
echo "  DAC-FSQ:         $DAC_FSQ_CKPT"
echo ""

# Check if checkpoints exist
for ckpt in "$ENCODEC_CKPT" "$Q2D2_CKPT" "$SPEECHTOKENIZER_CKPT"; do
    if [[ ! -f "$ckpt" ]]; then
        echo "ERROR: Checkpoint not found: $ckpt"
        exit 1
    fi
done

for ckpt_dir in "$HIFICODEC_CKPT" "$DAC_FSQ_CKPT"; do
    if [[ ! -d "$ckpt_dir" ]]; then
        echo "ERROR: Checkpoint directory not found: $ckpt_dir"
        exit 1
    fi
done

# HiFiCodec sometimes writes g_ checkpoints under a nested logs/ directory.
if [[ -d "$HIFICODEC_CKPT/logs" ]] && find "$HIFICODEC_CKPT/logs" -maxdepth 1 -type f -name 'g_*' | grep -q .; then
    HIFICODEC_CKPT="$HIFICODEC_CKPT/logs"
    echo "✓ HiFiCodec checkpoint auto-detected in nested logs/ directory"
fi

if ! find "$HIFICODEC_CKPT" -maxdepth 2 -type f -name 'g_*' | grep -q .; then
    echo "ERROR: No HiFiCodec generator checkpoint (g_*) found under: $HIFICODEC_CKPT"
    echo "Hint: run: find hificodec -type f -name 'g_*' | sort | tail -20"
    exit 1
fi

echo "✓ All checkpoints verified"
echo ""

# Step 1: Download EGFx dataset if needed
if [[ ! -d "$EGFX_DIR" ]] || [[ ! -f "$PAIRS_JSON" ]]; then
    echo "Step 1/4: Downloading EGFx dataset..."
    python tok_analysis/egfx_download.py --output-dir "$EGFX_DIR"
    
    echo ""
    echo "Step 2/4: Preparing effect pairs..."
    python tok_analysis/egfx_prepare_pairs.py \
        --egfx-dir "$EGFX_DIR" \
        --output "$PAIRS_JSON"
else
    echo "Step 1-2/4: Dataset already prepared, skipping download"
fi

echo ""
echo "Step 3/4: Encoding through all 5 codecs (100 samples)..."
echo "  Categories: distortion, modulation, time_based, dynamics"
echo "  Samples per category: $MAX_PER_CATEGORY"
echo "  Total samples: ~100"
echo ""

python tok_analysis/egfx_encode.py \
    --pairs "$PAIRS_JSON" \
    --categories distortion modulation time_based dynamics \
    --max-per-category "$MAX_PER_CATEGORY" \
    --seed "$SEED" \
    --write-sampled-pairs "$SAMPLED_PAIRS" \
    --codecs encodec q2d2 hificodec speechtokenizer dac_fsq \
    --encodec-checkpoint "$ENCODEC_CKPT" \
    --q2d2-checkpoint "$Q2D2_CKPT" \
    --hificodec-checkpoint "$HIFICODEC_CKPT" \
    --speechtokenizer-checkpoint "$SPEECHTOKENIZER_CKPT" \
    --dac-fsq-checkpoint "$DAC_FSQ_CKPT" \
    --output-dir "$TOKENS_DIR" \
    --device "$DEVICE"

echo ""
echo "Step 4/4: Computing geometric metrics..."
python tok_analysis/egfx_metrics.py \
    --tokens-dir "$TOKENS_DIR" \
    --output "$METRICS_JSON"

echo ""
echo "Step 5/5: Generating analysis report..."
python tok_analysis/egfx_analyze.py \
    --metrics "$METRICS_JSON" \
    --output "$REPORT_PDF"

echo ""
echo "========================================"
echo "✓ Pipeline complete!"
echo "========================================"
echo ""
echo "Results:"
echo "  - Sampled pairs: $SAMPLED_PAIRS"
echo "  - Token outputs: $TOKENS_DIR"
echo "  - Metrics JSON:  $METRICS_JSON"
echo "  - Report PDF:    $REPORT_PDF"
echo ""
echo "Open the PDF to view comparative analysis across all 5 codecs."
echo ""

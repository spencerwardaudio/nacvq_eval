#!/bin/bash
# Run complete EGFx multi-codec analysis pipeline
# Usage: bash tok_analysis/run_egfx_pipeline.sh

set -e  # Exit on error

echo "=================================="
echo "EGFx Multi-Codec Analysis Pipeline"
echo "=================================="

# Configuration
EGFX_DIR="datasets/egfx"
PAIRS_JSON="datasets/egfx/effect_pairs.json"
TOKENS_DIR="datasets/audio_tokens/egfx"
METRICS_JSON="datasets/analysis/egfx_metrics.json"
REPORT_PDF="datasets/analysis/egfx_report.pdf"

# Checkpoint paths (update these!)
ENCODEC_CKPT="checkpoints_multi_dataset/encodec_fsd50k_24kbps.pt"
Q2D2_CKPT="Q2D2/checkpoints/q2d2_fsd50k_9.8kbps.ckpt"
HIFICODEC_CKPT="hificodec/checkpoints/"
SPEECHTOKENIZER_CKPT="SpeechTokenizer/checkpoints/speechtokenizer_fsd50k.pt"
DAC_FSQ_CKPT="descript-audio-codec/ckpt/fsd50k_fsq"

# Device
DEVICE="cuda"

echo ""
echo "Step 1/5: Downloading EGFx dataset..."
python tok_analysis/egfx_download.py --output-dir "$EGFX_DIR"

echo ""
echo "Step 2/5: Preparing effect pairs..."
python tok_analysis/egfx_prepare_pairs.py \
    --egfx-dir "$EGFX_DIR" \
    --output "$PAIRS_JSON"

echo ""
echo "Step 3/5: Encoding through all codecs..."
echo "  (This may take several hours depending on dataset size)"
python tok_analysis/egfx_encode.py \
    --pairs "$PAIRS_JSON" \
    --codecs encodec q2d2 hificodec speechtokenizer dac_fsq \
    --encodec-checkpoint "$ENCODEC_CKPT" \
    --q2d2-checkpoint "$Q2D2_CKPT" \
    --hificodec-checkpoint "$HIFICODEC_CKPT" \
    --speechtokenizer-checkpoint "$SPEECHTOKENIZER_CKPT" \
    --dac-fsq-checkpoint "$DAC_FSQ_CKPT" \
    --output-dir "$TOKENS_DIR" \
    --device "$DEVICE"

echo ""
echo "Step 4/5: Computing geometric metrics..."
python tok_analysis/egfx_metrics.py \
    --tokens-dir "$TOKENS_DIR" \
    --output "$METRICS_JSON"

echo ""
echo "Step 5/5: Generating analysis report..."
python tok_analysis/egfx_analyze.py \
    --metrics "$METRICS_JSON" \
    --output "$REPORT_PDF"

echo ""
echo "=================================="
echo "✓ Pipeline complete!"
echo "=================================="
echo ""
echo "Results:"
echo "  - Metrics: $METRICS_JSON"
echo "  - Report:  $REPORT_PDF"
echo ""
echo "Open the PDF to view comparative analysis across codecs and effect types."

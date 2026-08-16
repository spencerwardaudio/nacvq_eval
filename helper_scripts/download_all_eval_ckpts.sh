#!/bin/bash
# Run from msc_proj/ — mirrors discover_checkpoints() in run_analysis.py exactly

set -euo pipefail

# Encodec: newest .pt by mtime, excluding discriminator files
ENC=$(find Encodec/outputs -name '*.pt' | grep -v '_disc_' \
      | xargs ls -t 2>/dev/null | head -1)

# Q2D2: newest last.ckpt by mtime across all lightning versions
Q2D=$(find Q2D2/outputs -path '*/checkpoints/last.ckpt' \
      | xargs ls -t 2>/dev/null | head -1)

# HiFiCodec: whole dir (discovery just checks it exists + has g_* files)
HFC="hificodec/egs/hificodec_fsd50k"

# SpeechTokenizer: fixed path
ST="results/speechtokenizer_fsd50k/SpeechTokenizer_best_dev.pt"

# DAC-FSQ: whole dir (discovery checks for latest/ subdir)
DAC="descript-audio-codec/ckpt/fsd50k_fsq"

echo "Checkpoints to zip:"
echo "  Encodec:         $ENC"
echo "  Q2D2:            $Q2D"
echo "  HiFiCodec:       $HFC/"
echo "  SpeechTokenizer: $ST"
echo "  DAC-FSQ:         $DAC/"
echo ""

du -sh "$ENC" "$Q2D" "$HFC" "$ST" "$DAC" 2>/dev/null
echo ""

tar -czf best_checkpoints_$(date +%Y%m%d).tar.gz \
    "$ENC" \
    "$Q2D" \
    "$HFC" \
    "$ST" \
    "$DAC"

echo "Done: best_checkpoints_$(date +%Y%m%d).zip"
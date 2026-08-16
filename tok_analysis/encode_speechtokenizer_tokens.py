"""Encode sensitivity-test signals with SpeechTokenizer and save per-RVQ-codebook token arrays.

Produces .npy files of shape [n_q, T] — same convention as the Encodec
tokenization scripts — so the existing flip-rate and perplexity analysis
functions work unchanged.

SpeechTokenizer uses 8-level RVQ at 16 kHz.

Output path convention (mirrors Encodec/HiFiCodec):
  datasets/audio_tokens/speechtokenizer/{test_name}/{signal}/
      {signal}_baseline_bwST_tokens.npy
      {signal}_var_{tag}_bwST_tokens.npy

Usage:
    python encode_speechtokenizer_tokens.py \\
        --checkpoint results/speechtokenizer_fsd50k/SpeechTokenizer_best_dev.pt \\
        [--config    SpeechTokenizer/config/fsd50k_cfg.json] \\
        [--amp-wav   datasets/dsp_self_amp_recordings] \\
        [--phase-wav datasets/dsp_self_phase_recordings] \\
        [--temporal-wav datasets/time_sine_recordings] \\
        [--device cuda] \\
        [--bw-tag ST]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from codec_interface import get_codec_encoder

_DS = _PROJ_ROOT / "datasets"


# ---------------------------------------------------------------------------
# Per-file tokenisation
# ---------------------------------------------------------------------------

def _encode_wav(wav_path: Path, encoder, bw_tag: str) -> np.ndarray:
    """Return [n_q, T] int32 token array for one wav file."""
    result = encoder.encode(str(wav_path))
    return result["tokens"]  # [n_q, T] int32


# ---------------------------------------------------------------------------
# Walk a test-signal directory tree
# ---------------------------------------------------------------------------

def _process_test_dir(
    wav_root: Path,
    out_root: Path,
    encoder,
    bw_tag: str,
    verbose: bool = True,
) -> None:
    wav_files = sorted(wav_root.rglob("*.wav"))
    if not wav_files:
        print(f"  [WARN] No .wav files found under {wav_root}")
        return

    print(f"  Found {len(wav_files)} wav files in {wav_root.name}")
    for wf in wav_files:
        rel     = wf.relative_to(wav_root)
        sub_dir = rel.parent
        stem    = wf.stem

        out_dir = out_root / sub_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        out_name = f"{stem}_bw{bw_tag}_tokens.npy"
        out_path = out_dir / out_name

        if out_path.exists():
            if verbose:
                print(f"    skip (exists): {out_name}")
            continue

        try:
            tokens = _encode_wav(wf, encoder, bw_tag)
            np.save(str(out_path), tokens)
            if verbose:
                print(f"    {stem} → {tokens.shape}  saved")
        except Exception as exc:
            print(f"    [FAIL] {wf.name}: {exc}")

    # Sanity check
    written = list(out_root.rglob("*_tokens.npy"))
    assert len(written) > 0, f"No token files written to {out_root}"  # catches silent failures where every wav raised an exception


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint",
                    default=str(_PROJ_ROOT / "results" / "speechtokenizer_fsd50k" / "SpeechTokenizer_best_dev.pt"),
                    help="Path to SpeechTokenizer checkpoint (.pt)")
    ap.add_argument("--config",
                    default=str(_PROJ_ROOT / "SpeechTokenizer" / "config" / "fsd50k_cfg.json"),
                    help="SpeechTokenizer JSON config")
    ap.add_argument("--amp-wav",
                    default=str(_DS / "dsp_self_amp_recordings"),
                    help="Root of amplitude-test wav recordings")
    ap.add_argument("--phase-wav",
                    default=str(_DS / "dsp_self_phase_recordings"),
                    help="Root of phase-test wav recordings")
    ap.add_argument("--temporal-wav",
                    default=str(_DS / "time_sine_recordings"),
                    help="Root of temporal-test wav recordings")
    ap.add_argument("--out-root",
                    default=str(_DS / "audio_tokens" / "speechtokenizer"),
                    help="Root directory for output token .npy files")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bw-tag", default="ST",
                    help="Bandwidth tag used in output filenames (default: ST)")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint)
    out_root   = Path(args.out_root)

    print(f"\n[SpeechTokenizer Tokenizer]")
    print(f"  Checkpoint : {checkpoint}")
    print(f"  Config     : {args.config}")
    print(f"  Output     : {out_root}")
    print(f"  Device     : {args.device}")
    print(f"  BW tag     : {args.bw_tag}")

    encoder = get_codec_encoder(
        "speechtokenizer",
        checkpoint,
        config_path=Path(args.config),
        device=args.device,
    )
    print(f"  RVQ codebooks: {encoder.n_layers}")

    test_dirs = [
        (Path(args.amp_wav),      out_root / "dsp_self_amp",   "Amplitude"),
        (Path(args.phase_wav),    out_root / "dsp_self_phase", "Phase"),
        (Path(args.temporal_wav), out_root / "time_sine",      "Temporal"),
    ]

    for wav_root, out_dir, label in test_dirs:
        if not wav_root.exists():
            print(f"\n[SKIP] {label}: {wav_root} does not exist")
            continue
        print(f"\n[{label}] {wav_root.name} → {out_dir.name}/")
        _process_test_dir(wav_root, out_dir, encoder, args.bw_tag, args.verbose)

    print(f"\nDone. Tokens written to {out_root}")
    print(f"Run the combined report:")
    print(f"  python report_multi_codec_sensitivity.py \\")
    print(f"      --codecs encodec q2d2 hificodec speechtokenizer \\")
    print(f"      --speechtokenizer-bw-tag {args.bw_tag}")


if __name__ == "__main__":
    main()

"""Encode sensitivity-test signals with pre-trained NeuCodec and save token arrays.

NeuCodec uses Finite Scalar Quantisation (FSQ) — a single 1D quantizer
producing one token stream.  Output shape: [1, T_50hz] where T_50hz is the
number of 50Hz frames (50 tokens per second at 16kHz input, upsampled to
24kHz output).

Because FSQ is a single-stream quantizer (no separate codebooks), the
"codebook" dimension is 1.  The existing flip-rate and perplexity analysis
functions handle this correctly — n_codebooks == 1 means a single row of
subplots.

The model is loaded from HuggingFace Hub (auto-downloads ~1 GB on first run):
  NeuCodec.from_pretrained("neuphonic/neucodec")

Input audio is resampled to 16kHz before encoding (NeuCodec requirement).

Output path convention:
  datasets/audio_tokens/neucodec/{test_name}/{signal}/
      {signal}_baseline_bwNEU_tokens.npy
      {signal}_var_{tag}_bwNEU_tokens.npy

Usage:
    python encode_neucodec_tokens.py \\
        [--amp-wav   datasets/dsp_self_amp_recordings] \\
        [--phase-wav datasets/dsp_self_phase_recordings] \\
        [--temporal-wav datasets/time_sine_recordings] \\
        [--device cuda] \\
        [--bw-tag NEU]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torchaudio import transforms as T

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
_NEUCODEC_DIR = _PROJ_ROOT / "neucodec"

if str(_NEUCODEC_DIR) not in sys.path:
    sys.path.insert(0, str(_NEUCODEC_DIR))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_neucodec(device: str):
    """Load NeuCodec pre-trained model from HuggingFace Hub."""
    from neucodec import NeuCodec

    print("  Downloading / loading NeuCodec from HuggingFace Hub …")
    model = NeuCodec.from_pretrained("neuphonic/neucodec")
    model.eval().to(device)
    print(f"  NeuCodec loaded  (sample_rate={model.sample_rate}, hop={model.hop_length})")
    return model


# ---------------------------------------------------------------------------
# Per-file tokenisation
# ---------------------------------------------------------------------------

_TARGET_SR = 16_000  # NeuCodec was trained at 16 kHz; all inputs must be resampled to this rate


@torch.no_grad()
def _encode_wav(wav_path: Path, model, device: str) -> np.ndarray:
    """Return [1, T_frames] int32 FSQ token array for one wav file."""
    wav, sr = torchaudio.load(str(wav_path))
    if sr != _TARGET_SR:
        wav = T.Resample(sr, _TARGET_SR)(wav)
    if wav.shape[0] > 1:         # stereo → mono by averaging channels
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.unsqueeze(0).to(device)   # [1, 1, T_16k]

    fsq_codes = model.encode_code(wav)  # [B, 1, T_50hz]
    tokens = fsq_codes[0].cpu().numpy()  # [1, T_50hz]
    return tokens.astype(np.int32)


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------

def _process_test_dir(
    wav_root: Path,
    out_root: Path,
    model,
    device: str,
    bw_tag: str,
    verbose: bool = True,
) -> None:
    wav_files = sorted(wav_root.rglob("*.wav"))
    if not wav_files:
        print(f"  [WARN] No .wav files under {wav_root}")
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
            tokens = _encode_wav(wf, model, device)
            np.save(str(out_path), tokens)
            if verbose:
                print(f"    {stem} → {tokens.shape}  saved")
        except Exception as exc:
            print(f"    [FAIL] {wf.name}: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_DS = _PROJ_ROOT / "datasets"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
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
                    default=str(_DS / "audio_tokens" / "neucodec"),
                    help="Root directory for output token .npy files")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bw-tag", default="NEU",
                    help="Bandwidth tag used in output filenames (default: NEU)")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    print(f"\n[NeuCodec Tokenizer — pre-trained FSQ]")
    print(f"  Output  : {out_root}")
    print(f"  Device  : {args.device}")
    print(f"  BW tag  : {args.bw_tag}")
    print(f"  Note    : NeuCodec uses FSQ (single token stream, n_codebooks=1)")

    model = _load_neucodec(args.device)

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
        _process_test_dir(wav_root, out_dir, model, args.device,
                          args.bw_tag, args.verbose)

    print(f"\nDone. Tokens written to {out_root}")


if __name__ == "__main__":
    main()

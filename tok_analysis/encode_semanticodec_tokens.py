"""Encode sensitivity-test signals with pre-trained SemantiCodec and save token arrays.

SemantiCodec produces two token streams:
  • Semantic tokens  — from a frozen AudioMAE encoder (coarse, content-level)
  • Acoustic tokens  — from a trained acoustic VQ (fine, reconstruction-level)

Output shape: [2, T_frames] where row 0 is semantic and row 1 is acoustic.
For token-rate=25/50/100 the T_frames dimension scales accordingly.

This matches the [n_codebooks, T] convention used by Encodec/HiFiCodec so
the existing flip-rate and perplexity analysis functions work unchanged.
'Codebook 1' in the report corresponds to the semantic token stream,
'Codebook 2' corresponds to the acoustic token stream.

Output path convention:
  datasets/audio_tokens/semanticodec/{test_name}/{signal}/
      {signal}_baseline_bwSEM_tokens.npy
      {signal}_var_{tag}_bwSEM_tokens.npy

Usage:
    python encode_semanticodec_tokens.py \\
        [--token-rate  50]   (25 / 50 / 100 tokens/sec) \\
        [--vocab-size  32768] \\
        [--amp-wav    datasets/dsp_self_amp_recordings] \\
        [--phase-wav  datasets/dsp_self_phase_recordings] \\
        [--temporal-wav datasets/time_sine_recordings] \\
        [--device cuda] \\
        [--bw-tag SEM]
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
_SEMANTICODEC_DIR = _PROJ_ROOT / "SemantiCodec-inference"

if str(_SEMANTICODEC_DIR) not in sys.path:
    sys.path.insert(0, str(_SEMANTICODEC_DIR))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_semanticodec(token_rate: int, vocab_size: int, device: str):
    """Load SemantiCodec; weights auto-downloaded from HuggingFace on first use."""
    from semanticodec import SemantiCodec

    print(f"  Loading SemantiCodec (token_rate={token_rate}, vocab={vocab_size}) …")
    model = SemantiCodec(token_rate=token_rate, semantic_vocab_size=vocab_size)
    print(f"  SemantiCodec loaded")
    return model


# ---------------------------------------------------------------------------
# Per-file tokenisation
# ---------------------------------------------------------------------------

_TARGET_SR = 16_000  # SemantiCodec’s AudioMAE encoder was trained at 16 kHz; higher rates must be downsampled


def _encode_wav(wav_path: Path, model, device: str) -> np.ndarray:
    """Return [2, T_frames] int32 token array: row0=semantic, row1=acoustic."""
    # SemantiCodec expects a file path or 16kHz tensor
    try:
        # Preferred: pass path directly (SemantiCodec handles loading)
        tokens = model.encode(str(wav_path))
    except Exception:
        # Fallback: load and resample manually, then encode
        wav, sr = torchaudio.load(str(wav_path))
        if sr != _TARGET_SR:
            wav = T.Resample(sr, _TARGET_SR)(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        import tempfile, soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        sf.write(tmp_path, wav.squeeze().numpy(), _TARGET_SR)
        tokens = model.encode(tmp_path)
        Path(tmp_path).unlink(missing_ok=True)

    # tokens may be numpy or torch
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().numpy()
    tokens = np.asarray(tokens, dtype=np.int32)

    # SemantiCodec returns shape [1, T, 2] or [T, 2] or [2, T] depending on version
    if tokens.ndim == 3:
        tokens = tokens[0]          # [T, 2]  — strip batch dim if present
    if tokens.ndim == 2 and tokens.shape[-1] == 2:
        tokens = tokens.T           # [2, T]  ← semantic row, acoustic row  — normalise to [n_streams, T]
    elif tokens.ndim == 2 and tokens.shape[0] == 2:
        pass                        # already [2, T]
    elif tokens.ndim == 1:
        tokens = tokens[np.newaxis, :]  # [1, T]  single-stream fallback
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
    ap.add_argument("--token-rate", type=int, default=50,
                    choices=[25, 50, 100],
                    help="SemantiCodec token rate in tokens/sec")
    ap.add_argument("--vocab-size", type=int, default=32768,
                    choices=[4096, 8192, 16384, 32768])
    ap.add_argument("--amp-wav",
                    default=str(_DS / "dsp_self_amp_recordings"))
    ap.add_argument("--phase-wav",
                    default=str(_DS / "dsp_self_phase_recordings"))
    ap.add_argument("--temporal-wav",
                    default=str(_DS / "time_sine_recordings"))
    ap.add_argument("--out-root",
                    default=str(_DS / "audio_tokens" / "semanticodec"))
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bw-tag", default="SEM",
                    help="Bandwidth tag in output filenames (default: SEM)")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    print(f"\n[SemantiCodec Tokenizer — pre-trained semantic+acoustic VQ]")
    print(f"  Token rate  : {args.token_rate} tokens/sec")
    print(f"  Vocab size  : {args.vocab_size}")
    print(f"  Output      : {out_root}")
    print(f"  Device      : {args.device}")
    print(f"  BW tag      : {args.bw_tag}")
    print(f"  Token layout: [2, T] — row 0 = semantic, row 1 = acoustic")

    model = _load_semanticodec(args.token_rate, args.vocab_size, args.device)

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

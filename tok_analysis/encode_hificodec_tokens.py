"""Encode sensitivity-test signals with HiFiCodec and save per-group token arrays.

Produces .npy files of shape [n_codebooks, T] — same convention as the
Encodec tokenization scripts — so the existing flip-rate and perplexity
analysis functions work unchanged.

HiFiCodec uses Group-Residual VQ (GRVQ) with n_code_groups codebooks
(default 4 in config_24k_320d.json).  Each group's token index is stored
as a separate row in the output array.

Output path convention (mirrors Encodec):
  datasets/audio_tokens/hificodec/{test_name}/{signal}/
      {signal}_baseline_bwHFC_tokens.npy
      {signal}_var_{tag}_bwHFC_tokens.npy

where "HFC" is the bandwidth tag (fixed string for HiFiCodec since it does
not have variable bitrate in the same sense as Encodec).

Usage:
    python encode_hificodec_tokens.py \\
        --checkpoint PATH/TO/hificodec_fsd50k \\
        [--config    PATH/TO/config_24k_320d.json] \\
        [--amp-wav   datasets/dsp_self_amp_recordings] \\
        [--phase-wav datasets/dsp_self_phase_recordings] \\
        [--temporal-wav datasets/time_sine_recordings] \\
        [--device cuda] \\
        [--bw-tag HFC]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torchaudio import transforms as T

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
_HIFICODEC_DIR = _PROJ_ROOT / "hificodec"

# Make hificodec importable
if str(_HIFICODEC_DIR) not in sys.path:
    sys.path.insert(0, str(_HIFICODEC_DIR))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_hificodec(checkpoint_dir: Path, config_path: Path, device: str):
    """Load HiFiCodec encoder + quantizer from a training checkpoint dir."""
    from academicodec.models.hificodec.env import AttrDict
    from academicodec.models.hificodec.models import Encoder, Quantizer
    from academicodec.utils import scan_checkpoint, load_checkpoint

    with open(config_path) as f:
        h = AttrDict(json.load(f))

    encoder   = Encoder(h).to(device)
    quantizer = Quantizer(h).to(device)

    # Use g_best (best val checkpoint)
    g_best = Path(checkpoint_dir) / "g_best"  # HiFiCodec trainer symlinks the best generator checkpoint here
    if not g_best.exists():
        raise FileNotFoundError(
            f"No g_best checkpoint found in {checkpoint_dir}. "
            "Train HiFiCodec first with train_fsd50k.sh"
        )
    cp_g = str(g_best)

    state = load_checkpoint(cp_g, device)
    encoder.load_state_dict(state["encoder"])
    quantizer.load_state_dict(state["quantizer"])

    encoder.eval()
    quantizer.eval()
    print(f"  Loaded HiFiCodec from: {cp_g}")
    print(f"  Codebook groups: {h.n_code_groups}  vocab size: {h.n_codes}")
    return encoder, quantizer, h


# ---------------------------------------------------------------------------
# Per-file tokenisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _encode_wav(wav_path: Path, encoder, quantizer, h, device: str) -> np.ndarray:
    """Return [n_code_groups, T_frames] int32 token array for one wav file."""
    wav, sr = torchaudio.load(str(wav_path))
    if sr != h.sampling_rate:
        wav = T.Resample(sr, h.sampling_rate)(wav)
    if wav.shape[0] > 1:          # stereo → mono by averaging channels to avoid doubling energy
        wav = wav.mean(dim=0, keepdim=True)
    wav = wav.unsqueeze(0).to(device)  # [1, 1, T]

    z = encoder(wav)               # [1, quantized_vector_size, T_frames]
    _, _, tokens = quantizer(z)    # returns (quantized_out, vq_loss, per_group_indices)

    tokens_np = np.stack([t.squeeze(0).cpu().numpy() for t in tokens], axis=0)  # [G, T]
    return tokens_np.astype(np.int32)


# ---------------------------------------------------------------------------
# Walk a test-signal directory tree (same layout as Encodec tokens)
# ---------------------------------------------------------------------------

def _process_test_dir(
    wav_root: Path,
    out_root: Path,
    encoder,
    quantizer,
    h,
    device: str,
    bw_tag: str,
    verbose: bool = True,
) -> None:
    """Encode all wav files under wav_root, mirroring directory structure."""
    wav_files = sorted(wav_root.rglob("*.wav"))
    if not wav_files:
        print(f"  [WARN] No .wav files found under {wav_root}")
        return

    print(f"  Found {len(wav_files)} wav files in {wav_root.name}")
    for wf in wav_files:
        # Compute relative subdir structure
        rel      = wf.relative_to(wav_root)
        sub_dir  = rel.parent          # e.g. self_amp_440hz/ — preserves test-signal hierarchy in output
        stem     = wf.stem             # e.g. self_amp_440hz_baseline

        out_dir  = out_root / sub_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # Map wav stem to token filename using same convention as Encodec
        out_name = f"{stem}_bw{bw_tag}_tokens.npy"  # bw_tag='HFC' so filenames match the _bwHFC_ regex in analyzers
        out_path = out_dir / out_name

        if out_path.exists():
            if verbose:
                print(f"    skip (exists): {out_name}")
            continue

        try:
            tokens = _encode_wav(wf, encoder, quantizer, h, device)
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
    ap.add_argument("--checkpoint",
                    default=str(_PROJ_ROOT / "hificodec" / "egs" / "hificodec_fsd50k"),
                    help="Path to HiFiCodec training checkpoint directory")
    ap.add_argument("--config",
                    default=str(_PROJ_ROOT / "hificodec" / "egs" / "HiFi-Codec-24k-320d" / "config_24k_320d.json"),
                    help="HiFiCodec JSON config")
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
                    default=str(_DS / "audio_tokens" / "hificodec"),
                    help="Root directory for output token .npy files")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--bw-tag", default="HFC",
                    help="Bandwidth tag used in output filenames (default: HFC)")
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    checkpoint_dir = Path(args.checkpoint)
    config_path    = Path(args.config)
    out_root       = Path(args.out_root)

    print(f"\n[HiFiCodec Tokenizer]")
    print(f"  Checkpoint : {checkpoint_dir}")
    print(f"  Config     : {config_path}")
    print(f"  Output     : {out_root}")
    print(f"  Device     : {args.device}")
    print(f"  BW tag     : {args.bw_tag}")

    encoder, quantizer, h = _load_hificodec(checkpoint_dir, config_path, args.device)

    test_dirs = [
        (Path(args.amp_wav),      out_root / "dsp_self_amp",    "Amplitude"),
        (Path(args.phase_wav),    out_root / "dsp_self_phase",  "Phase"),
        (Path(args.temporal_wav), out_root / "time_sine",       "Temporal"),
    ]

    for wav_root, out_dir, label in test_dirs:
        if not wav_root.exists():
            print(f"\n[SKIP] {label}: {wav_root} does not exist")
            continue
        print(f"\n[{label}] {wav_root.name} → {out_dir.name}/")
        _process_test_dir(wav_root, out_dir, encoder, quantizer, h,
                          args.device, args.bw_tag, args.verbose)

    print(f"\nDone. Tokens written to {out_root}")
    print(f"Run the combined report:")
    print(f"  python report_multi_codec_sensitivity.py \\")
    print(f"      --codecs encodec q2d2 hificodec \\")
    print(f"      --hificodec-bw-tag {args.bw_tag}")


if __name__ == "__main__":
    main()

"""Export EnCodec token indices from .ecdc files into .npy arrays.

This script reads EnCodec bitstreams (.ecdc), decodes the latent token indices
without reconstructing audio, and saves one NumPy file per input with shape
[n_codebooks, n_frames].
"""

from __future__ import annotations

import argparse
import io
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any

# batch_encode_24kbps embeds the bw tag in the .ecdc filename; strip it before re-appending
_BW_SUFFIX_RE = re.compile(r"_bw[\d.]+$")  # matches _bw24.0 etc. at the end of a stem

# Ensure Encodec modules (binary, compress, quantization) are importable
_HERE = Path(__file__).resolve().parent
_ENCODEC_DIR = _HERE.parent / "Encodec"
if str(_ENCODEC_DIR) not in sys.path:
    sys.path.insert(0, str(_ENCODEC_DIR))

import numpy as np
import torch

import binary
from compress import MODELS


def _load_model(model_name: str, checkpoint: str | None, device: str):
    """Load and return an EnCodec model (call once, reuse across files)."""
    if model_name == "multi_dataset_encodec":
        if not checkpoint:
            raise ValueError("--checkpoint is required for multi_dataset_encodec")
        model = MODELS[model_name](checkpoint)
    else:
        model = MODELS[model_name]()
    model = model.to(device)
    model.eval()
    return model


def _extract_tokens_from_ecdc(
    ecdc_path: Path,
    model_name: str,
    checkpoint: str | None,
    device: str,
    _model=None,
) -> torch.Tensor:
    """Return token indices from one .ecdc file as [n_codebooks, n_frames]."""
    if _model is not None:
        model = _model
    elif model_name == "multi_dataset_encodec":
        if not checkpoint:
            raise ValueError("--checkpoint is required for multi_dataset_encodec")
        model = MODELS[model_name](checkpoint)
        model = model.to(device)
        model.eval()
    else:
        model = MODELS[model_name]()
        model = model.to(device)
        model.eval()

    with ecdc_path.open("rb") as f:
        ecdc_bytes = f.read()

    fo = io.BytesIO(ecdc_bytes)
    metadata: dict[str, Any] = binary.read_ecdc_header(fo)
    audio_length = metadata["al"]
    num_codebooks = metadata["nc"]
    use_lm = metadata["lm"]

    segment_length = getattr(model, "segment_length", None) or audio_length
    segment_stride = getattr(model, "segment_stride", None) or audio_length

    frames: list[torch.Tensor] = []
    for offset in range(0, audio_length, segment_stride):
        this_segment_length = min(audio_length - offset, segment_length)
        frame_length = int(math.ceil(this_segment_length * model.frame_rate / model.sample_rate))  # token frames = audio samples × (frame_rate/sample_rate)

        if getattr(model, "normalize", False):
            scale_fmt = "!f"
            binary._read_exactly(fo, struct.calcsize(scale_fmt))

        if use_lm:
            from quantization.ac import ArithmeticDecoder, build_stable_quantized_cdf

            lm = model.get_lm_model()
            decoder = ArithmeticDecoder(fo)
            states = None
            lm_offset = 0
            input_ = torch.zeros(1, num_codebooks, 1, dtype=torch.long, device=device)
        else:
            unpacker = binary.BitUnpacker(model.bits_per_codebook, fo)

        frame = torch.zeros(1, num_codebooks, frame_length, dtype=torch.long, device=device)
        for t in range(frame_length):
            if use_lm:
                with torch.no_grad():
                    probas, states, lm_offset = lm(input_, states, lm_offset)

            code_list: list[int] = []
            for k in range(num_codebooks):
                if use_lm:
                    q_cdf = build_stable_quantized_cdf(
                        probas[0, :, k, 0], decoder.total_range_bits, check=False
                    )
                    code = decoder.pull(q_cdf)
                else:
                    code = unpacker.pull()  # read exactly bits_per_codebook bits per symbol per frame

                if code is None:
                    raise EOFError(f"Unexpected end-of-stream while reading {ecdc_path}")
                code_list.append(code)

            frame[0, :, t] = torch.tensor(code_list, dtype=torch.long, device=device)
            if use_lm:
                input_ = 1 + frame[:, :, t : t + 1]  # shift by 1 so index 0 is reserved as BOS padding

        frames.append(frame)

    tokens = torch.cat([f[0] for f in frames], dim=1)  # concat along time; segments arise when audio > segment_length
    return tokens


def _extract_pairs_from_wav(
    wav_path: Path,
    model,
    device: str,
) -> torch.Tensor:
    """Re-encode a wav file and return Q2D2 per-pair indices as ``[n_pairs, n_frames]``.

    This bypasses the .ecdc binary format and works directly with the model's
    encoder + ``quantizer.encode_pairs()``.  Only valid when the model was built
    with ``use_q2d2=True``.
    """
    import soundfile as sf
    from utils import convert_audio

    wav, sr = sf.read(str(wav_path))
    wav_t = torch.from_numpy(wav).float()
    if wav_t.ndim == 1:
        wav_t = wav_t.unsqueeze(0)
    else:
        wav_t = wav_t.transpose(0, 1)
    wav_t = convert_audio(wav_t, sr, model.sample_rate, model.channels)
    wav_t = wav_t.unsqueeze(0).to(device)   # [1, C, T]

    with torch.no_grad():
        z = model.encoder(wav_t)             # [1, D, T_enc]
        pair_ind = model.quantizer.encode_pairs(z)  # [n_pairs, 1, T_enc]

    return pair_ind[:, 0, :]                 # [n_pairs, T_enc]


def _iter_inputs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".ecdc":
            raise ValueError(f"Input file must be .ecdc, got: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    return sorted([p for p in input_path.glob("*.ecdc") if p.is_file()])


def _parse_args() -> argparse.Namespace:
    _proj_root = _HERE.parent
    parser = argparse.ArgumentParser(
        description="Export token index arrays from .ecdc files as *_tokens.npy"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_proj_root / "datasets" / "ecdc",
        help="Input .ecdc file or directory containing .ecdc files (sub-dirs are processed recursively)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_proj_root / "datasets" / "audio_tokens",
        help="Directory where *_tokens.npy files will be written (mirrors input sub-dir structure)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="encodec_48khz",
        choices=sorted(MODELS.keys()),
        help="Model to use for bitstream token decoding",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path (required for multi_dataset_encodec)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device used for LM-assisted entropy decoding",
    )
    parser.add_argument(
        "--q2d2",
        action="store_true",
        help="Q2D2 mode: re-encode .wav files to extract per-pair indices "
             "(bypasses .ecdc; requires --checkpoint and --wav-input).",
    )
    parser.add_argument(
        "--wav-input",
        type=Path,
        default=None,
        help="Q2D2 mode: directory of .wav files (or a single .wav file) to encode.",
    )
    parser.add_argument(
        "--bw-tag",
        type=str,
        default="24.0",
        help="Bandwidth tag appended to output filenames: {stem}_bw{tag}_tokens.npy",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    # ------------------------------------------------------------------
    # Q2D2 mode: re-encode .wav files → per-pair [n_pairs, n_frames] npy
    # ------------------------------------------------------------------
    if args.q2d2:
        wav_input = args.wav_input
        if wav_input is None:
            raise ValueError("--wav-input is required in --q2d2 mode")

        model = _load_model(args.model_name, args.checkpoint, args.device)

        wav_files: list[Path]
        if wav_input.is_file():
            wav_files = [wav_input]
        elif wav_input.is_dir():
            wav_files = sorted(wav_input.glob("*.wav"))
            if not wav_files:
                raise FileNotFoundError(f"No .wav files found in {wav_input}")
        else:
            raise FileNotFoundError(f"--wav-input path not found: {wav_input}")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Q2D2 pair-extraction mode: {len(wav_files)} file(s) → {args.output_dir}")

        success = 0
        for wav_file in wav_files:
            try:
                pair_ind = _extract_pairs_from_wav(wav_file, model, args.device)
                out_name = f"{wav_file.stem}_bw24.0_tokens.npy"
                out_path = args.output_dir / out_name
                np.save(out_path, pair_ind.cpu().numpy().astype(np.int64, copy=False))
                print(f"[OK] {wav_file.name} -> {out_name} shape={tuple(pair_ind.shape)}")
                success += 1
            except Exception as exc:
                print(f"[FAIL] {wav_file.name}: {exc}")

        print(f"Done. Converted {success}/{len(wav_files)} file(s).")
        return

    # ------------------------------------------------------------------
    # Standard RVQ mode: read .ecdc bitstreams
    # ------------------------------------------------------------------
    input_path = args.input
    if input_path.is_dir():
        subdirs = [p for p in input_path.iterdir() if p.is_dir()]
        if subdirs:
            # Mirror sub-directory structure into output_dir
            total_success = 0
            for subdir in sorted(subdirs):
                out_subdir = args.output_dir / subdir.name
                out_subdir.mkdir(parents=True, exist_ok=True)
                inputs = _iter_inputs(subdir)
                if not inputs:
                    print(f"[SKIP] No .ecdc files in {subdir}")
                    continue
                print(f"\n--- {subdir.name}: {len(inputs)} file(s) → {out_subdir} ---")
                loaded_model = _load_model(args.model_name, args.checkpoint, args.device)
                for ecdc_file in inputs:
                    try:
                        tokens = _extract_tokens_from_ecdc(
                            ecdc_path=ecdc_file,
                            model_name=args.model_name,
                            checkpoint=args.checkpoint,
                            device=args.device,
                            _model=loaded_model,
                        )
                        stem = _BW_SUFFIX_RE.sub("", ecdc_file.stem)
                        out_name = f"{stem}_bw{args.bw_tag}_tokens.npy"
                        assert "_bw" not in out_name.replace(f"_bw{args.bw_tag}", ""), \
                            f"Double bw-tag detected in output name: {out_name}"
                        out_path = out_subdir / out_name
                        np.save(out_path, tokens.cpu().numpy().astype(np.int64, copy=False))
                        print(f"  [OK] {ecdc_file.name} -> {out_name} shape={tuple(tokens.shape)}")
                        total_success += 1
                    except Exception as exc:
                        print(f"  [FAIL] {ecdc_file.name}: {exc}")
            print(f"\nDone. Converted {total_success} file(s) total.")
            return

    inputs = _iter_inputs(input_path)
    if not inputs:
        raise FileNotFoundError(f"No .ecdc files found in {input_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(inputs)} .ecdc file(s) from: {input_path}")
    print(f"Saving token arrays to: {args.output_dir}")

    success = 0
    for ecdc_file in inputs:
        try:
            tokens = _extract_tokens_from_ecdc(
                ecdc_path=ecdc_file,
                model_name=args.model_name,
                checkpoint=args.checkpoint,
                device=args.device,
            )
            stem = _BW_SUFFIX_RE.sub("", ecdc_file.stem)
            out_name = f"{stem}_bw{args.bw_tag}_tokens.npy"
            assert "_bw" not in out_name.replace(f"_bw{args.bw_tag}", ""), \
                f"Double bw-tag detected in output name: {out_name}"
            out_path = args.output_dir / out_name
            np.save(out_path, tokens.cpu().numpy().astype(np.int64, copy=False))
            print(f"[OK] {ecdc_file.name} -> {out_name} shape={tuple(tokens.shape)}")
            success += 1
        except Exception as exc:
            print(f"[FAIL] {ecdc_file.name}: {exc}")

    print(f"Done. Converted {success}/{len(inputs)} file(s).")


if __name__ == "__main__":
    main()

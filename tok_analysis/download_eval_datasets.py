#!/usr/bin/env python3
"""Download small representative subsets of the 6 HEAR benchmark evaluation datasets.

RECOMMENDED APPROACH:
For standardized, reproducible dataset acquisition, use the official HEAR preprocessing
toolkit instead of this fallback script:

    pip install heareval
    # Download preprocessed tasks from https://doi.org/10.5281/zenodo.5885750
    # Or use hear-preprocess to generate tasks from source

This script provides a lightweight fallback for quick evaluation when the full HEAR
infrastructure is not needed. It downloads small subsets directly from public archives.

Datasets:
  1. ESC-50           - environmental sound classification (GitHub zip)
  2. NSynth Pitch     - musical note pitch recognition (HuggingFace streaming)
  3. Speech Commands  - short spoken word commands (HuggingFace streaming)
  4. LibriCount       - speaker count estimation (Zenodo record 1216072)
  5. CREMA-D          - speech emotion recognition (HuggingFace streaming)
  6. Vocal Imitations - vocal imitation classification (Zenodo record 1340763)

Each dataset is placed in its own named subdirectory of --output-dir so that the
manifest generator can identify the dataset from the folder name.
All downloads are skipped if the target directory already has enough files.
All failures are non-fatal: a warning is printed and the next dataset continues.
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _save_audio(array, sr: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(array, "cpu"):
        array = array.cpu().numpy()
    array = np.asarray(array, dtype=np.float32)
    if array.ndim > 1:
        array = array[0]
    sf.write(str(out_path), array, sr)


def _already_done(out_dir: Path, n: int) -> bool:
    existing = (
        list(out_dir.glob("*.wav"))
        + list(out_dir.glob("*.ogg"))
        + list(out_dir.glob("*.flac"))
    )
    if len(existing) >= n:
        print(f"  [{out_dir.name}] {len(existing)} files already present — skipping")  # idempotency check
        return True
    return False


# ---------------------------------------------------------------------------
# 1. ESC-50  (karolpiczak/ESC-50, GitHub zip)
# ---------------------------------------------------------------------------

def download_esc50(out_dir: Path, n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if _already_done(out_dir, n):
        return
    print("  [esc50] downloading from GitHub (~600 MB)...")
    tmp = out_dir.parent / "_esc50_tmp.zip"
    try:
        urllib.request.urlretrieve(
            "https://github.com/karolpiczak/ESC-50/archive/master.zip", tmp
        )
        with zipfile.ZipFile(tmp) as zf:
            entries = sorted(
                e for e in zf.namelist()
                if e.startswith("ESC-50-master/audio/") and e.endswith(".wav")
            )
            saved = 0
            for entry in entries[:n]:
                (out_dir / Path(entry).name).write_bytes(zf.read(entry))
                saved += 1
        print(f"  [esc50] saved {saved} files to {out_dir}")
    except Exception as exc:
        print(f"  [esc50] FAILED — {exc}")
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# HuggingFace streaming helper (used by nsynth, speech_commands, crema_d)
# ---------------------------------------------------------------------------

def _download_hf(
    hf_id: str,
    config: str | None,
    split: str,
    audio_col: str,
    out_dir: Path,
    n: int,
    prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if _already_done(out_dir, n):
        return
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print(f"  [{prefix}] 'datasets' library not installed — run: pip install datasets")
        return
    print(f"  [{prefix}] streaming first {n} samples from HuggingFace ({hf_id})...")
    try:
        ds = load_dataset(hf_id, config, split=split, streaming=True, trust_remote_code=True)
        saved = 0
        for sample in ds:
            if saved >= n:
                break
            audio = sample.get(audio_col)
            if audio is None:
                continue
            arr = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
            _save_audio(arr, sr, out_dir / f"{prefix}_{saved:04d}.wav")
            saved += 1
        print(f"  [{prefix}] saved {saved} files to {out_dir}")
    except Exception as exc:
        print(f"  [{prefix}] FAILED — {exc}")


# ---------------------------------------------------------------------------
# 2. NSynth Pitch  (google/nsynth, HuggingFace)
# ---------------------------------------------------------------------------

def download_nsynth(out_dir: Path, n: int) -> None:
    _download_hf("google/nsynth", "gansynth_subset", "test", "audio", out_dir, n, "nsynth")


# ---------------------------------------------------------------------------
# 3. Speech Commands  (speech_commands v0.02, HuggingFace)
# ---------------------------------------------------------------------------

def download_speech_commands(out_dir: Path, n: int) -> None:
    _download_hf("speech_commands", "v0.02", "test", "audio", out_dir, n, "speech_commands")


# ---------------------------------------------------------------------------
# 4. LibriCount  (Zenodo record 1216072, zip)
# ---------------------------------------------------------------------------

def download_libricount(out_dir: Path, n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if _already_done(out_dir, n):
        return
    print("  [libricount] downloading from Zenodo (~800 MB)...")
    tmp = out_dir.parent / "_lc_tmp.zip"
    extract_root = out_dir.parent / "_lc_extract"
    try:
        urllib.request.urlretrieve(
            "https://zenodo.org/record/1216072/files/LibriCount10-0dB.zip", tmp
        )
        with zipfile.ZipFile(tmp) as zf:
            wav_entries = [e for e in zf.namelist() if e.endswith(".wav")][:n]
            for entry in wav_entries:
                zf.extract(entry, extract_root)
        moved = 0
        for wav in extract_root.rglob("*.wav"):
            shutil.move(str(wav), str(out_dir / f"libricount_{moved:04d}.wav"))
            moved += 1
            if moved >= n:
                break
        print(f"  [libricount] saved {moved} files to {out_dir}")
    except Exception as exc:
        print(f"  [libricount] FAILED — {exc}")
    finally:
        tmp.unlink(missing_ok=True)
        shutil.rmtree(extract_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# 5. CREMA-D  (HuggingFace, tries two known repo IDs)
# ---------------------------------------------------------------------------

def download_crema_d(out_dir: Path, n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if _already_done(out_dir, n):
        return
    for hf_id in ("WillHeld/crema_d", "AbstractTTS/CREMA-D"):
        try:
            from datasets import load_dataset  # type: ignore
            print(f"  [crema_d] trying {hf_id}...")
            ds = load_dataset(hf_id, split="train", streaming=True, trust_remote_code=True)
            saved = 0
            for sample in ds:
                if saved >= n:
                    break
                audio = sample.get("audio") or sample.get("file")
                if audio is None:
                    continue
                if isinstance(audio, dict):
                    arr = np.asarray(audio["array"], dtype=np.float32)
                    sr = int(audio["sampling_rate"])
                else:
                    arr, sr = sf.read(audio)
                    arr = arr.astype(np.float32)
                _save_audio(arr, sr, out_dir / f"crema_d_{saved:04d}.wav")
                saved += 1
            print(f"  [crema_d] saved {saved} files to {out_dir}")
            return
        except Exception as exc:
            print(f"  [crema_d] {hf_id} — {exc}")
    print("  [crema_d] all sources failed")


# ---------------------------------------------------------------------------
# 6. Vocal Imitations  (Zenodo record 1340763, zip)
# ---------------------------------------------------------------------------

def download_vocal_imitations(out_dir: Path, n: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if _already_done(out_dir, n):
        return
    print("  [vocal_imitations] downloading from Zenodo (~225 MB)...")
    tmp = out_dir.parent / "_vocalim_tmp.zip"
    try:
        urllib.request.urlretrieve(
            "https://zenodo.org/record/1340763/files/VocalImitationSet_v1.1.3.zip", tmp
        )
        with zipfile.ZipFile(tmp) as zf:
            entries = sorted(e for e in zf.namelist() if e.lower().endswith(".wav"))
            saved = 0
            for entry in entries[:n]:
                (out_dir / f"vocal_imitations_{saved:04d}.wav").write_bytes(zf.read(entry))
                saved += 1
        print(f"  [vocal_imitations] saved {saved} files to {out_dir}")
    except Exception as exc:
        print(f"  [vocal_imitations] FAILED — {exc}")
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALL_DATASETS = [
    "esc50", "nsynth", "speech_commands", "libricount", "crema_d", "vocal_imitations"
]

_DISPATCH = {
    "esc50": download_esc50,
    "nsynth": download_nsynth,
    "speech_commands": download_speech_commands,
    "libricount": download_libricount,
    "crema_d": download_crema_d,
    "vocal_imitations": download_vocal_imitations,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download HEAR benchmark evaluation dataset samples"
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/audio_files",
        help="Root directory for audio output (default: datasets/audio_files)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=20,
        help="Number of samples per dataset (default: 20)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
        help="Which datasets to download (default: all 6)",
    )
    args = parser.parse_args()

    root = Path(args.output_dir)
    n = args.n_samples
    print(f"Downloading up to {n} samples per dataset into {root}/")

    for name in args.datasets:
        _DISPATCH[name](root / name, n)

    print("Dataset download complete.")


if __name__ == "__main__":
    main()

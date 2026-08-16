"""Download and prepare a GTZAN subset for EnCodec token sensitivity analysis.

Downloads directly from the GTZAN mirror, then resamples to 48 kHz mono
and saves full 30-second clips into datasets/audio_files/.

Output naming: {genre}_{idx:02d}.wav  (e.g., blues_00.wav, classical_09.wav)

Usage:
    python download_gtzan.py [--per-genre 10] [--output-dir ../datasets/audio_files]

Requirements:
    pip install soundfile scipy numpy
"""

from __future__ import annotations

import argparse
import io
import tarfile
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import soundfile as sf
from scipy.signal import resample as sp_resample

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
_DEFAULT_OUTPUT = _PROJ_ROOT / "datasets" / "audio_files"

GTZAN_GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

TARGET_SR = 48000  # resample all GTZAN files to 48 kHz to match EnCodec input requirements
GTZAN_URL = "http://marsyas.info/mirrors/genres.tar.gz"


def download_and_prepare(
    output_dir: Path,
    per_genre: int = 10,
) -> list[Path]:
    """Download GTZAN via mirror archive and save a subset as 48 kHz mono WAVs."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading GTZAN from {GTZAN_URL} ...")
    print(f"Output: {output_dir}")
    print(f"Per genre: {per_genre} files, {len(GTZAN_GENRES)} genres\n")

    try:
        response = urlopen(GTZAN_URL, timeout=90)  # 90 s timeout for the ~1 GB archive download
        tar_data = response.read()
    except Exception as exc:
        print(f"ERROR: Could not download from {GTZAN_URL}")
        print(f"  {exc}")
        print("\nAlternative: run this script on a machine with internet access,")
        print("then copy the prepared WAV files to datasets/audio_files/.")
        return []

    saved: list[Path] = []
    counts: Counter[str] = Counter()

    print("Extracting and processing audio files...\n")
    with tarfile.open(fileobj=io.BytesIO(tar_data), mode="r:gz") as tar:
        for member in tar:
            if not member.isfile() or not member.name.endswith(".au"):
                continue

            # Expected path: genres/<genre>/<genre>.<index>.au
            parts = member.name.split("/")
            if len(parts) < 3:
                continue

            genre = parts[1]
            if genre not in GTZAN_GENRES:
                continue
            if counts[genre] >= per_genre:
                continue

            try:
                handle = tar.extractfile(member)
                if handle is None:
                    continue

                audio, sr = sf.read(io.BytesIO(handle.read()))

                if audio.ndim > 1:
                    audio = audio.mean(axis=1)

                if sr != TARGET_SR:
                    n_out = int(len(audio) * TARGET_SR / sr)
                    audio = sp_resample(audio, n_out).astype(np.float32)

                peak = np.abs(audio).max()
                if peak > 0:
                    audio = audio / (peak + 1e-10) * 0.95

                idx = counts[genre]
                out_path = output_dir / f"{genre}_{idx:02d}.wav"
                sf.write(str(out_path), audio, TARGET_SR)
                saved.append(out_path)
                counts[genre] += 1
            except Exception as exc:
                # GTZAN has at least one known corrupt sample (e.g., jazz.00054)
                print(f"  [SKIP] {member.name}: {exc}")
                continue

    # Report
    for genre in GTZAN_GENRES:
        n = counts[genre]
        if n > 0:
            print(f"  [{genre}] {n} files saved")
        else:
            print(f"  [{genre}] No files found — skipping")

    total = sum(counts.values())
    print(f"\nDone: {total} files saved to {output_dir}")
    return saved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download GTZAN subset for EnCodec sensitivity analysis",
    )
    parser.add_argument("--per-genre", type=int, default=10,
                        help="Number of files per genre (default: 10)")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT,
                        help="Output directory for prepared WAVs")
    args = parser.parse_args()

    download_and_prepare(args.output_dir, args.per_genre)


if __name__ == "__main__":
    main()

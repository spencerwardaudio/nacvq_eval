"""Generate time-offset sine wave recordings for temporal sensitivity analysis.

For each of 20 log-spaced frequencies (10 Hz – 20 kHz), generates a 2-second
file containing a 1-second pure sine wave at 0 dBFS placed at the given offset
from the start, with silence everywhere else.  The signal is never truncated so
RMS is identical across all offsets.

  Offset schedule:
  • 1 ms – 20 ms  (every 1 ms, 20 offsets)
  • 40, 60, 80, 100 ms  (4 offsets)
  Total: 25 offsets + 1 baseline = 26 files per frequency.

Using pure sinusoids (rather than complex musical signals) lets temporal
sensitivity be measured against a single-frequency stimulus, free of
frequency-domain confounds.

Output structure under <rec_root>/:
  <freq>hz/
    baseline_0ms.wav
    offset_001ms.wav
    ...
    offset_020ms.wav
    offset_040ms.wav
    offset_060ms.wav
    offset_080ms.wav
    offset_100ms.wav

Usage:
    python generate_sine_timeoffsets.py [rec_root]

    No args → outputs to datasets/time_sine_recordings/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
_DEFAULT_OUT = _PROJ_ROOT / "datasets" / "time_sine_recordings"

# ── signal constants ──────────────────────────────────────────────────────────
SR = 24_000          # 24 kHz — matches multi_dataset_encodec checkpoint
SIG_DURATION = 1.0   # length of the sine burst (seconds)
FILE_DURATION = 2.0  # total file length — gives headroom for all offsets up to 100 ms + 1 s burst
SIG_SAMPLES = int(SR * SIG_DURATION)
FILE_SAMPLES = int(SR * FILE_DURATION)
T = np.linspace(0, SIG_DURATION, SIG_SAMPLES, endpoint=False)
PEAK = 0.5           # 0 dBFS reference (matches all other generators)

# 20 log-spaced frequencies 10 Hz → 20 kHz (same grid as all other tests)
FREQ_TAGS: list[int] = sorted({int(round(f)) for f in np.geomspace(10, 20_000, 20)})

# Offset schedule: 1–20 ms linear + 40/60/80/100 ms log
_OFFSETS_MS: list[int] = list(range(1, 21)) + [40, 60, 80, 100]  # dense near 0 to detect the codec’s temporal resolution limit


def _write(path: Path, audio: np.ndarray) -> None:
    assert len(audio) >= FILE_SAMPLES, (
        f"{path.name}: file length {len(audio)} samples < expected {FILE_SAMPLES} "
        f"({FILE_DURATION} s @ {SR} Hz); signal burst must be {SIG_SAMPLES} samples "
        f"({SIG_DURATION} s) within the file"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, SR, subtype="PCM_16")


def _make_file(sine_burst: np.ndarray, offset_samples: int) -> np.ndarray:
    """Place sine_burst into a zero-filled FILE_SAMPLES buffer at offset_samples.

    The burst is never truncated — FILE_DURATION is long enough that even the
    largest offset (100 ms = 2400 samples) leaves the full 1-second burst intact.
    """
    buf = np.zeros(FILE_SAMPLES, dtype=np.float64)
    end = offset_samples + len(sine_burst)
    buf[offset_samples:end] = sine_burst
    return buf


def generate_all(rec_root: Path) -> None:
    rec_root.mkdir(parents=True, exist_ok=True)
    total_files = 0

    for freq in FREQ_TAGS:
        out_dir = rec_root / f"{freq}hz"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1-second sine burst (same for all offsets of this frequency)
        sine_burst = PEAK * np.sin(2 * np.pi * freq * T)

        # Baseline: burst starts at sample 0
        _write(out_dir / "baseline_0ms.wav", _make_file(sine_burst, 0))

        for offset_ms in _OFFSETS_MS:
            offset_samples = int(round(offset_ms / 1000.0 * SR))
            fname = f"offset_{offset_ms:03d}ms.wav"
            _write(out_dir / fname, _make_file(sine_burst, offset_samples))

        n_per = 1 + len(_OFFSETS_MS)
        total_files += n_per
        print(f"  [{freq:>6} Hz]  {n_per} files → {out_dir}")

    print(f"\n  {len(FREQ_TAGS)} subdirs / {total_files} WAV files → {rec_root}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "rec_root",
        nargs="?",
        default=str(_DEFAULT_OUT),
        help="Root directory for recordings",
    )
    args = ap.parse_args()
    generate_all(Path(args.rec_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())

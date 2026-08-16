"""Generate self-phase modulation test signals.

For each of 20 log-spaced frequencies at TWO amplitude levels (0 dBFS and
−70 dBFS), the BASELINE is that frequency at 0° phase.  Variants are the SAME
frequency + amplitude at 24 phase offsets: 15°, 30°, 45°, …, 360°.

This isolates phase: the only thing changing between baseline and each variant
is the initial phase of the sinusoid.  Running at both 0 dBFS and −70 dBFS
reveals whether phase sensitivity is amplitude-dependent.

Output structure under <rec_root>/:
  self_phase_0dB_{freq}hz/
    self_phase_0dB_{freq}hz_baseline.wav      # 0° phase, 0 dBFS
    self_phase_0dB_{freq}hz_repeat_1.wav      # identical (determinism)
    self_phase_0dB_{freq}hz_repeat_2.wav
    self_phase_0dB_{freq}hz_var_15.wav        # 15° phase offset
    ...
    self_phase_0dB_{freq}hz_var_360.wav       # 360° phase offset (full cycle)
  self_phase_70dB_{freq}hz/
    ...                                        # same structure at −70 dBFS
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# ── signal constants ──────────────────────────────────────────────────────────
SR = 24_000  # 24 kHz — matches multi_dataset_encodec checkpoint
DURATION = 5.0
N_SAMPLES = int(SR * DURATION)
T = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)

PEAK_0DB = 0.5  # 0 dBFS reference

# Two amplitude levels: (label, peak)
AMP_LEVELS: list[tuple[str, float]] = [
    ("0dB",  PEAK_0DB),
    ("70dB", PEAK_0DB * 10 ** (-70.0 / 20.0)),
]

# 20 log-spaced frequencies 10 Hz → 20 kHz (same grid as frequency sweep test)
FREQ_TAGS: list[int] = sorted({int(round(f)) for f in np.geomspace(10, 20_000, 20)})

# Phase offsets (degrees), stored as integer tags in filenames
# Every 15° from 15° to 360° = 24 steps
PHASE_TAGS: list[int] = list(range(15, 361, 15))  # 360° is a full cycle — identical to 0°; included as a sanity anchor


def _write(path: Path, signal: np.ndarray) -> None:
    assert len(signal) >= N_SAMPLES, (
        f"{path.name}: signal length {len(signal)} samples < expected {N_SAMPLES} "
        f"({DURATION} s @ {SR} Hz)"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), signal, SR, subtype="PCM_16")


def generate_all(rec_root: Path) -> None:
    rec_root.mkdir(parents=True, exist_ok=True)
    for amp_label, peak in AMP_LEVELS:
        for freq in FREQ_TAGS:
            sig_name = f"self_phase_{amp_label}_{freq}hz"
            out_dir = rec_root / sig_name
            out_dir.mkdir(parents=True, exist_ok=True)

            # Baseline: 0° phase
            baseline = peak * np.sin(2 * np.pi * freq * T)
            _write(out_dir / f"{sig_name}_baseline.wav", baseline)
            _write(out_dir / f"{sig_name}_repeat_1.wav", baseline)
            _write(out_dir / f"{sig_name}_repeat_2.wav", baseline)

            # Phase variants
            for phase_deg in PHASE_TAGS:
                phase_rad = np.deg2rad(phase_deg)
                variant = peak * np.sin(2 * np.pi * freq * T + phase_rad)
                _write(out_dir / f"{sig_name}_var_{phase_deg}.wav", variant)

            print(f"  [{sig_name}]  phases={PHASE_TAGS}°")

    n_subdirs = len(AMP_LEVELS) * len(FREQ_TAGS)
    n_files = n_subdirs * (3 + len(PHASE_TAGS))
    print(f"\n  {n_subdirs} subdirs / {n_files} WAV files → {rec_root}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "rec_root",
        nargs="?",
        default="datasets/dsp_self_phase_recordings",
        help="Root directory for recordings",
    )
    args = ap.parse_args()
    generate_all(Path(args.rec_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())

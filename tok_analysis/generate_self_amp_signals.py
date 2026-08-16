"""Generate self-amplitude modulation test signals.

For each of 20 log-spaced frequencies, the BASELINE is that frequency at
0 dBFS.  Variants are the SAME frequency at −20 / −40 / −60 / −80 / −100
/ −120 / −140 dBFS.

This isolates amplitude: the only thing changing between baseline and each
variant is the signal level — frequency and waveform shape are fixed.

Output structure under <rec_root>/:
  self_amp_{freq}hz/
    self_amp_{freq}hz_baseline.wav       # 0 dBFS
    self_amp_{freq}hz_repeat_1.wav       # identical to baseline (determinism)
    self_amp_{freq}hz_repeat_2.wav
    self_amp_{freq}hz_var_20.wav         # −20 dBFS absolute
    self_amp_{freq}hz_var_40.wav         # −40 dBFS
    self_amp_{freq}hz_var_60.wav         # −60 dBFS
    self_amp_{freq}hz_var_80.wav         # −80 dBFS
    self_amp_{freq}hz_var_100.wav        # −100 dBFS
    self_amp_{freq}hz_var_120.wav        # −120 dBFS
    self_amp_{freq}hz_var_140.wav        # −140 dBFS
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# ── signal constants ──────────────────────────────────────────────────────────
SR = 24_000  # 24 kHz — matches multi_dataset_encodec checkpoint sample rate
DURATION = 5.0
N_SAMPLES = int(SR * DURATION)
T = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)

PEAK_0DB = 0.5  # reference for 0 dBFS (matches all other test generators); 0.5 keeps headroom below clipping

# Baseline amplitude: 0 dBFS
BASELINE_ATTEN_DB: int = 0
BASELINE_PEAK: float = PEAK_0DB

# Variant amplitude levels: absolute attenuation below 0 dBFS
# Tag = attenuation magnitude (positive integer), stored in filename as _var_<tag>_
AMP_ATTENUATIONS_DB: list[int] = [20, 40, 60, 80, 100, 120, 140]

# 20 log-spaced frequencies 10 Hz → 20 kHz (same grid as frequency sweep test)
FREQ_TAGS: list[int] = sorted({int(round(f)) for f in np.geomspace(10, 20_000, 20)})  # set() deduplicates after rounding


def _peak_for_atten(atten_db: float) -> float:
    return PEAK_0DB * 10 ** (-atten_db / 20.0)


def _write(path: Path, signal: np.ndarray) -> None:
    assert len(signal) >= N_SAMPLES, (
        f"{path.name}: signal length {len(signal)} samples < expected {N_SAMPLES} "
        f"({DURATION} s @ {SR} Hz)"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), signal, SR, subtype="PCM_16")


def generate_all(rec_root: Path) -> None:
    rec_root.mkdir(parents=True, exist_ok=True)
    for freq in FREQ_TAGS:
        sig_name = f"self_amp_{freq}hz"
        out_dir = rec_root / sig_name
        out_dir.mkdir(parents=True, exist_ok=True)

        baseline = BASELINE_PEAK * np.sin(2 * np.pi * freq * T)
        _write(out_dir / f"{sig_name}_baseline.wav", baseline)
        _write(out_dir / f"{sig_name}_repeat_1.wav", baseline)
        _write(out_dir / f"{sig_name}_repeat_2.wav", baseline)

        for atten in AMP_ATTENUATIONS_DB:
            variant = _peak_for_atten(atten) * np.sin(2 * np.pi * freq * T)
            _write(out_dir / f"{sig_name}_var_{atten}.wav", variant)

        print(
            f"  [{sig_name}]  baseline=0 dBFS  "
            f"variants={[f'−{a} dBFS' for a in AMP_ATTENUATIONS_DB]}"
        )

    n_files = len(FREQ_TAGS) * (3 + len(AMP_ATTENUATIONS_DB))
    print(f"\n  {len(FREQ_TAGS)} subdirs / {n_files} WAV files → {rec_root}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "rec_root",
        nargs="?",
        default="datasets/dsp_self_amp_recordings",
        help="Root directory for recordings",
    )
    args = ap.parse_args()
    generate_all(Path(args.rec_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Generate waveform-shape (timbre) test signals for EnCodec token sensitivity.

All four waveforms share the same 1 kHz fundamental and are compared at 3
amplitude levels. Baseline per subdir = sine wave (the simplest waveform).

Waveforms (variant tag → shape):
  0 = sine       pure tone, no harmonics
  1 = sawtooth   all harmonics (H_k = 1/k)
  2 = triangle   odd harmonics only (H_k = 1/k²) — gentler roll-off than square
  3 = square     odd harmonics only (H_k = 1/k) — bright, harsh

All shapes are peak-normalised before amplitude scaling so each variant has the
same peak level as the baseline sine.

Amplitude levels (attenuation below 0 dBFS): [0, 70, 140] dB
  → peaks: [0.5, ~1.6e-4, ~5.0e-8]

File tags = 0, 1, 2, 3 (floatable by standard _var_<tag>_bw regex).
The analyzer maps these back to shape names for axis labels.

Output layout under <output_base>/:
    timbre_0dB/
        timbre_0dB_baseline.wav      (sine, 0 dBFS)
        timbre_0dB_repeat_1.wav
        timbre_0dB_repeat_2.wav
        timbre_0dB_var_0.wav         (sine  — sanity: flip rate should be 0)
        timbre_0dB_var_1.wav         (sawtooth)
        timbre_0dB_var_2.wav         (triangle)
        timbre_0dB_var_3.wav         (square)
    timbre_70dB/   (same shapes, -70 dBFS)
    timbre_140dB/  (same shapes, -140 dBFS ≈ silence)

Usage:
    python generate_timbre_test_signals.py [output_base]

Default output_base: <project_root>/datasets/dsp_timbre_recordings/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import sawtooth, square

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
_DEFAULT_OUT = _PROJ_ROOT / "datasets" / "dsp_timbre_recordings"

SAMPLE_RATE = 48000
DURATION_S = 5.0
PEAK_0DB = 0.5
FUNDAMENTAL_HZ = 1000.0

# Variant index → (display name, generator)
# Each generator takes a phase array and returns a waveform in [-1, 1]
SHAPES: dict[int, tuple[str, object]] = {
    0: ("sine",     lambda ph: np.sin(ph)),
    1: ("saw",      lambda ph: sawtooth(ph, width=1.0)),
    2: ("triangle", lambda ph: sawtooth(ph, width=0.5)),  # sawtooth with width=0.5 gives triangle wave
    3: ("square",   lambda ph: square(ph)),
}

# Three amplitude levels: no attenuation, -70 dB, -140 dB (≈ silence)
AMP_ATTENUATIONS_DB = [0, 70, 140]


def _peak_for_atten(atten_db: float) -> float:
    return PEAK_0DB * 10.0 ** (-atten_db / 20.0)


def _waveform(shape_idx: int, n: int, peak: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    phase = 2.0 * np.pi * FUNDAMENTAL_HZ * t
    _, gen = SHAPES[shape_idx]
    x = np.asarray(gen(phase), dtype=np.float64)
    max_abs = np.abs(x).max()
    if max_abs > 0:
        x = x * (peak / max_abs)  # peak-normalise before amplitude scaling so all shapes have equal peak
    return x.astype(np.float32)


def _write(path: Path, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), signal, SAMPLE_RATE)


def main(output_base: Path = _DEFAULT_OUT) -> None:
    n = int(SAMPLE_RATE * DURATION_S)

    for atten_db in AMP_ATTENUATIONS_DB:
        peak = _peak_for_atten(atten_db)
        sig_name = f"timbre_{atten_db}dB"
        out_dir = output_base / sig_name

        # Baseline = sine at this amplitude
        baseline = _waveform(0, n, peak)
        _write(out_dir / f"{sig_name}_baseline.wav", baseline)
        _write(out_dir / f"{sig_name}_repeat_1.wav", baseline)
        _write(out_dir / f"{sig_name}_repeat_2.wav", baseline)
        print(f"  Wrote baseline + repeats  ({sig_name}, peak={peak:.3e})")

        for idx, (shape_name, _) in SHAPES.items():
            x = _waveform(idx, n, peak)
            _write(out_dir / f"{sig_name}_var_{idx}.wav", x)
            print(f"    var_{idx} = {shape_name:<10}  peak={np.abs(x).max():.3e}")

    print(f"\n  Done. Files written to: {output_base}")
    print(f"  Total WAVs: {(3 + len(SHAPES)) * len(AMP_ATTENUATIONS_DB)}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_OUT
    main(out)

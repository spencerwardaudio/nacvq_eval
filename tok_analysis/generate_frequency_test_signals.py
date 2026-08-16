"""Generate frequency sweep test signals for EnCodec token sensitivity experiments.

Sine waves at 20 log-spaced frequencies (10 Hz – 20 kHz) × 3 amplitude levels.
One output subdir per amplitude level; baseline per subdir = 1000 Hz sine.

Amplitude levels (attenuation below 0 dBFS): [0, 70, 140] dB
  → peaks: [0.5, ~1.6e-4, ~5.0e-8]

Frequencies: np.geomspace(10, 20_000, 20) rounded to integer Hz:
  [10, 15, 22, 32, 46, 68, 100, 147, 215, 316, 464, 681,
   1000, 1468, 2154, 3162, 4642, 6813, 10000, 20000]

File tags = integer Hz (floatable by standard _var_<tag>_bw regex).
Analyzer sets log x-axis, labelled "frequency (Hz)".

Output layout under <output_base>/:
    sine_freq_0dB/
        sine_freq_0dB_baseline.wav     (1000 Hz, 0 dBFS)
        sine_freq_0dB_repeat_1.wav
        sine_freq_0dB_repeat_2.wav
        sine_freq_0dB_var_10.wav       (10 Hz)
        sine_freq_0dB_var_15.wav       (15 Hz)
        ...
        sine_freq_0dB_var_20000.wav    (20 000 Hz)
    sine_freq_70dB/   (same frequencies, -70 dBFS)
    sine_freq_140dB/  (same frequencies, -140 dBFS ≈ silence)

Usage:
    python generate_frequency_test_signals.py [output_base]

Default output_base: <project_root>/datasets/dsp_frequency_recordings/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
_DEFAULT_OUT = _PROJ_ROOT / "datasets" / "dsp_frequency_recordings"

SAMPLE_RATE = 48000
DURATION_S = 5.0
PEAK_0DB = 0.5

BASELINE_FREQ_HZ = 1000.0
N_FREQS = 20
FREQ_MIN_HZ = 10.0
FREQ_MAX_HZ = 20_000.0

# Integer Hz tags — unique when rounded from geomspace(10, 20000, 20)
FREQ_TAGS: list[int] = sorted(set(
    int(round(f)) for f in np.geomspace(FREQ_MIN_HZ, FREQ_MAX_HZ, N_FREQS)  # rounding can merge very close values at low end
))

# Three amplitude levels: no attenuation, -70 dB, -140 dB (≈ silence)
AMP_ATTENUATIONS_DB = [0, 70, 140]


def _peak_for_atten(atten_db: float) -> float:
    return PEAK_0DB * 10.0 ** (-atten_db / 20.0)


def _sine(freq_hz: float, n: int, peak: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE
    return (peak * np.sin(2.0 * np.pi * freq_hz * t)).astype(np.float32)


def _write(path: Path, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), signal, SAMPLE_RATE)


def main(output_base: Path = _DEFAULT_OUT) -> None:
    n = int(SAMPLE_RATE * DURATION_S)

    for atten_db in AMP_ATTENUATIONS_DB:
        peak = _peak_for_atten(atten_db)
        sig_name = f"sine_freq_{atten_db}dB"
        out_dir = output_base / sig_name

        baseline = _sine(BASELINE_FREQ_HZ, n, peak)
        _write(out_dir / f"{sig_name}_baseline.wav", baseline)
        _write(out_dir / f"{sig_name}_repeat_1.wav", baseline)
        _write(out_dir / f"{sig_name}_repeat_2.wav", baseline)
        print(f"  Wrote baseline + repeats  ({sig_name}, peak={peak:.3e})")

        for freq_hz in FREQ_TAGS:
            x = _sine(float(freq_hz), n, peak)
            _write(out_dir / f"{sig_name}_var_{freq_hz}.wav", x)

        print(f"  Wrote {len(FREQ_TAGS)} frequency variants  "
              f"({FREQ_TAGS[0]} Hz – {FREQ_TAGS[-1]} Hz)")

    print(f"\n  Done. Files written to: {output_base}")
    print(f"  Total WAVs per amplitude: {3 + len(FREQ_TAGS)}"
          f"  ×  {len(AMP_ATTENUATIONS_DB)} levels"
          f"  =  {(3 + len(FREQ_TAGS)) * len(AMP_ATTENUATIONS_DB)}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_OUT
    main(out)

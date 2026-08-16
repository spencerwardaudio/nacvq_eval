"""Generate amplitude sweep test signals for EnCodec token sensitivity experiments.

White noise scaled to 5 amplitude levels, evenly spaced in dB from 0 to -140 dBFS.
Peak reference: 0 dBFS = peak 0.5 (matching other test generators).

Amplitude levels (attenuation below 0 dBFS): [0, 35, 70, 105, 140] dB
  → linear peaks: 0.5 * 10^(-atten/20) = [0.5, 8.9e-3, 1.6e-4, 2.8e-6, 5.0e-8]
  → -140 dBFS ≈ float32 numerical floor (effectively silence)

File tags encode the attenuation magnitude (positive integer), so the standard
_var_<tag>_bw regex can parse them as floats. The analyzer maps tag 35 → "-35 dB".

Output layout under <output_base>/white_noise_amp/:
    white_noise_amp_baseline.wav         (0 dBFS, peak 0.5)
    white_noise_amp_repeat_1.wav         (identical to baseline — determinism check)
    white_noise_amp_repeat_2.wav         (identical to baseline — determinism check)
    white_noise_amp_var_0.wav            (0 dBFS  — sanity: flip rate should be 0)
    white_noise_amp_var_35.wav           (-35 dBFS)
    white_noise_amp_var_70.wav           (-70 dBFS)
    white_noise_amp_var_105.wav          (-105 dBFS)
    white_noise_amp_var_140.wav          (-140 dBFS ≈ silence)

Usage:
    python generate_amplitude_test_signals.py [output_base]

Default output_base: <project_root>/datasets/dsp_amplitude_recordings/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
_DEFAULT_OUT = _PROJ_ROOT / "datasets" / "dsp_amplitude_recordings"

SAMPLE_RATE = 48000
DURATION_S = 5.0
PEAK_0DB = 0.5           # linear peak that corresponds to 0 dBFS in these tests; 0.5 leaves headroom below clipping

# Attenuation magnitudes in dB (0 = no attenuation, 140 ≈ silence)
AMP_ATTENUATIONS_DB = [0, 35, 70, 105, 140]


def _peak_for_atten(atten_db: float) -> float:
    """0 dB atten → PEAK_0DB;  140 dB atten → ~5e-8."""
    return PEAK_0DB * 10.0 ** (-atten_db / 20.0)


def _white_noise(n: int, seed: int = 0) -> np.ndarray:
    """Unit-peak white noise (float32)."""
    rng = np.random.default_rng(seed)  # seeded for exact reproducibility across runs
    x = rng.standard_normal(n).astype(np.float32)
    return x / np.abs(x).max()


def _write(path: Path, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), signal, SAMPLE_RATE)


def main(output_base: Path = _DEFAULT_OUT) -> None:
    n = int(SAMPLE_RATE * DURATION_S)
    sig_name = "white_noise_amp"
    out_dir = output_base / sig_name

    # Unit-peak waveform (shape only, amplitude applied below)
    unit_noise = _white_noise(n)

    # Baseline = 0 dBFS
    baseline = (unit_noise * _peak_for_atten(0)).astype(np.float32)
    _write(out_dir / f"{sig_name}_baseline.wav", baseline)
    _write(out_dir / f"{sig_name}_repeat_1.wav", baseline)
    _write(out_dir / f"{sig_name}_repeat_2.wav", baseline)
    print(f"  Wrote baseline + repeats  ({sig_name}, peak={_peak_for_atten(0):.4f})")

    for atten_db in AMP_ATTENUATIONS_DB:
        peak = _peak_for_atten(atten_db)
        x = (unit_noise * peak).astype(np.float32)
        tag = str(atten_db)
        _write(out_dir / f"{sig_name}_var_{tag}.wav", x)
        print(f"  {sig_name}_var_{tag:>3}:  -{atten_db:>3} dBFS,  peak={peak:.3e}")

    print(f"\n  Done. Files written to: {out_dir}")
    print(f"  Total WAVs: {3 + len(AMP_ATTENUATIONS_DB)}")


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_OUT
    main(out)

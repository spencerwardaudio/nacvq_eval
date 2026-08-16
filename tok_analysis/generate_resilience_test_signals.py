"""Generate resilience test signals for EnCodec token sensitivity experiments.

Two tasks share one generator:

  TASK 0 — Circular rotation of stochastic noise.
    For each noise color (white, pink, brown), build a 5 s @ 48 kHz mono buffer,
    write baseline + 2 identical repeats (determinism sanity), then write
    np.roll() copies for several sample shifts. Question: is the first codebook
    resilient to where the noise "starts"?

  TASK 1 — Phase offset of a 1 kHz sine.
    Write baseline (phase=0) + 2 identical repeats, then sin(2πft + φ) for a set
    of phase offsets. Question: which codebooks change when only phase changes?

Output layout (signal subdirs under <output_base>/):
    white_noise_rot/   <name>_baseline.wav, _repeat_{1,2}.wav, _var_<samples>.wav
    pink_noise_rot/    same
    brown_noise_rot/   same
    sine_phase/        sine_phase_baseline.wav, _repeat_{1,2}.wav, _var_<deg>.wav

Files use the shared "_var_<tag>" naming so analyze_resilience_test.py can
discover and label the variant axis correctly per signal.

Usage:
    python generate_resilience_test_signals.py [output_base]

Default output_base: <project_root>/datasets/dsp_resilience_recordings/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
_DEFAULT_OUT = _PROJ_ROOT / "datasets" / "dsp_resilience_recordings"

SAMPLE_RATE = 48000
DURATION_S = 5.0
PEAK = 0.5
SINE_FREQ = 1000.0

# Rotation amounts in samples: ~21 µs, 1 ms, 10 ms, 100 ms, 1 s, 2.5 s
ROT_SAMPLES = [1, 48, 480, 4800, 48000, 120000]  # spans from sub-frame shift to large temporal displacement

# Phase offsets in degrees
PHASE_DEGREES = [1, 5, 15, 30, 45, 90, 135, 180]

NOISE_COLORS = ["white_noise_rot", "pink_noise_rot", "brown_noise_rot"]


# ---------------------------------------------------------------------------
# Noise generators (mono, length n, returned float32 normalized to PEAK)
# ---------------------------------------------------------------------------

def _normalize(x: np.ndarray) -> np.ndarray:
    peak = np.abs(x).max()
    if peak > 0:
        x = x * (PEAK / peak)
    return x.astype(np.float32)


def _white(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return _normalize(rng.standard_normal(n))


def _pink(n: int, seed: int = 1) -> np.ndarray:
    """1/f noise via spectral shaping of white noise."""
    rng = np.random.default_rng(seed)
    # Build complex spectrum with magnitude 1/sqrt(f), random phase
    n_freqs = n // 2 + 1
    freqs = np.arange(n_freqs)
    freqs[0] = 1  # avoid div by zero at DC
    mag = 1.0 / np.sqrt(freqs)  # 1/f power ⇒ 1/√f magnitude (10 dB/decade slope)
    mag[0] = 0.0  # zero DC
    phase = rng.uniform(0, 2 * np.pi, n_freqs)
    spectrum = mag * np.exp(1j * phase)
    x = np.fft.irfft(spectrum, n=n)
    return _normalize(x)


def _brown(n: int, seed: int = 2) -> np.ndarray:
    """Brownian (red) noise: cumulative sum of white noise, DC-removed."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    x = np.cumsum(w)  # integration of white noise produces 1/f² (brown) spectrum
    x = x - x.mean()  # remove DC offset introduced by cumsum
    return _normalize(x)


def _sine(n: int, freq: float, phase_rad: float = 0.0) -> np.ndarray:
    t = np.arange(n) / SAMPLE_RATE
    x = np.sin(2 * np.pi * freq * t + phase_rad)
    return _normalize(x)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def _write(path: Path, data: np.ndarray) -> None:
    sf.write(str(path), data.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")


def _emit_common(out_dir: Path, name: str, baseline: np.ndarray) -> None:
    """Write baseline + two identical repeats for determinism sanity."""
    _write(out_dir / f"{name}_baseline.wav", baseline)
    _write(out_dir / f"{name}_repeat_1.wav", baseline)
    _write(out_dir / f"{name}_repeat_2.wav", baseline)


# ---------------------------------------------------------------------------
# Task 0: rotation of noise
# ---------------------------------------------------------------------------

def generate_rotation_signal(name: str, out_base: Path) -> None:
    out_dir = out_base / name
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(SAMPLE_RATE * DURATION_S)

    if name == "white_noise_rot":
        x = _white(n)
    elif name == "pink_noise_rot":
        x = _pink(n)
    elif name == "brown_noise_rot":
        x = _brown(n)
    else:
        raise ValueError(f"Unknown rotation signal: {name}")

    _emit_common(out_dir, name, x)
    for shift in ROT_SAMPLES:
        rolled = np.roll(x, shift).astype(np.float32)
        _write(out_dir / f"{name}_var_{shift}.wav", rolled)

    print(f"  ✓ {name}: wrote {3 + len(ROT_SAMPLES)} files to {out_dir}")


# ---------------------------------------------------------------------------
# Task 1: phase offset of 1 kHz sine
# ---------------------------------------------------------------------------

def generate_phase_signal(out_base: Path) -> None:
    name = "sine_phase"
    out_dir = out_base / name
    out_dir.mkdir(parents=True, exist_ok=True)
    n = int(SAMPLE_RATE * DURATION_S)

    baseline = _sine(n, SINE_FREQ, phase_rad=0.0)
    _emit_common(out_dir, name, baseline)

    for deg in PHASE_DEGREES:
        phi = np.deg2rad(deg)
        x = _sine(n, SINE_FREQ, phase_rad=phi)
        _write(out_dir / f"{name}_var_{deg}.wav", x)

    print(f"  ✓ {name}: wrote {3 + len(PHASE_DEGREES)} files to {out_dir}")


def main() -> None:
    out_base = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_OUT
    out_base.mkdir(parents=True, exist_ok=True)
    print(f"Resilience test signal generation → {out_base}")
    for color in NOISE_COLORS:
        generate_rotation_signal(color, out_base)
    generate_phase_signal(out_base)
    print("Done.")


if __name__ == "__main__":
    main()

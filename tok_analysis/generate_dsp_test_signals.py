"""Generate controlled DSP test signals for EnCodec determinism / sensitivity tests.

For each base signal (sine, impulse, white noise, chirp) writes:
  <signal>_baseline.wav            — original
  <signal>_repeat_1.wav            — bit-identical copy (determinism check)
  <signal>_repeat_2.wav            — bit-identical copy (determinism check)
  <signal>_eps_<mag>.wav           — baseline + uniform[-eps, +eps] noise

Signals are mono float32 PCM @ 48 kHz, normalized to peak 0.5 so that
eps=1e-2 perturbations cannot push samples past clipping.

Usage:
    python generate_dsp_test_signals.py [output_base]

Default output_base: <project_root>/datasets/dsp_test_recordings/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal as sig

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
_DEFAULT_OUT = _PROJ_ROOT / "datasets" / "dsp_test_recordings"

SAMPLE_RATE = 48000
DURATION_S = 5.0
PEAK = 0.5
EPS_LIST = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]  # log-spaced from sub-float32-precision to audible perturbation
SIGNALS = ["sine_1k", "impulse", "white_noise", "chirp"]


def _make_signal(name: str) -> np.ndarray:
    n = int(SAMPLE_RATE * DURATION_S)
    t = np.arange(n) / SAMPLE_RATE
    if name == "sine_1k":
        x = np.sin(2 * np.pi * 1000.0 * t)
    elif name == "impulse":
        x = np.zeros(n, dtype=np.float64)
        x[n // 2] = 1.0
    elif name == "white_noise":
        rng = np.random.default_rng(0)
        x = rng.standard_normal(n)
    elif name == "chirp":
        x = sig.chirp(t, f0=50.0, f1=20000.0, t1=DURATION_S, method="logarithmic")
    else:
        raise ValueError(f"Unknown signal: {name}")

    peak = np.abs(x).max()
    if peak > 0:
        x = x * (PEAK / peak)
    return x.astype(np.float32)


def _eps_tag(eps: float) -> str:
    # e.g. 1e-10 -> "1e-10"
    return f"{eps:g}"


def _write(path: Path, data: np.ndarray) -> None:
    sf.write(str(path), data.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")


def generate_for_signal(name: str, out_base: Path) -> None:
    out_dir = out_base / name
    out_dir.mkdir(parents=True, exist_ok=True)
    x = _make_signal(name)

    _write(out_dir / f"{name}_baseline.wav", x)
    _write(out_dir / f"{name}_repeat_1.wav", x)
    _write(out_dir / f"{name}_repeat_2.wav", x)

    for i, eps in enumerate(EPS_LIST):
        rng = np.random.default_rng(1000 + i)
        noise = rng.uniform(-eps, eps, size=x.shape).astype(np.float32)
        y = (x + noise).astype(np.float32)
        # Clip just in case (peak=0.5 + 1e-2 stays well within ±1.0, but be safe)
        np.clip(y, -1.0, 1.0, out=y)  # in-place clip avoids an extra allocation
        _write(out_dir / f"{name}_eps_{_eps_tag(eps)}.wav", y)

    print(f"  ✓ {name}: wrote {3 + len(EPS_LIST)} files to {out_dir}")


def main() -> None:
    out_base = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_OUT
    out_base.mkdir(parents=True, exist_ok=True)
    print(f"DSP test signal generation → {out_base}")
    for name in SIGNALS:
        generate_for_signal(name, out_base)
    print("Done.")


if __name__ == "__main__":
    main()

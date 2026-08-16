"""Generate SNR noise-tolerance test signals.

For each of 4 waveform shapes × 20 log-spaced frequencies, the BASELINE is
a clean signal at −6 dBFS.  Variants add white noise at 5 SNR levels where
the tag is the SNR in dB (noise is that many dB below the signal RMS):

  tag 100 → noise 100 dB below signal  (nearly inaudible)
  tag  80 → noise  80 dB below signal
  tag  60 → noise  60 dB below signal  (light noise floor)
  tag  40 → noise  40 dB below signal  (moderate noise)
  tag  20 → noise  20 dB below signal  (heavy noise, signal still audible)

This quantifies the SNR at which EnCodec's token sequence starts changing —
identifying the "noise floor" of the tokenisation.

Output structure under <rec_root>/:
  snr_{shape}_{freq}hz/
    snr_{shape}_{freq}hz_baseline.wav    # clean signal at −6 dBFS
    snr_{shape}_{freq}hz_repeat_1.wav    # determinism check (identical)
    snr_{shape}_{freq}hz_repeat_2.wav
    snr_{shape}_{freq}hz_var_100.wav     # SNR = 100 dB  (almost clean)
    snr_{shape}_{freq}hz_var_80.wav
    snr_{shape}_{freq}hz_var_60.wav
    snr_{shape}_{freq}hz_var_40.wav
    snr_{shape}_{freq}hz_var_20.wav      # SNR = 20 dB   (noisy)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import sawtooth
from scipy.signal import square as sp_square

# ── signal constants ──────────────────────────────────────────────────────────
SR = 48_000
DURATION = 5.0
N_SAMPLES = int(SR * DURATION)
T = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)

SEED = 0
PEAK_0DB = 0.5

# Baseline amplitude: −6 dBFS
BASELINE_ATTEN_DB: int = 6
BASELINE_PEAK: float = PEAK_0DB * 10 ** (-BASELINE_ATTEN_DB / 20.0)  # −6 dBFS leaves room to add noise without clipping

# 20 log-spaced frequencies 10 Hz → 20 kHz
FREQ_TAGS: list[int] = sorted({int(round(f)) for f in np.geomspace(10, 20_000, 20)})

# Waveform shapes: (name, generator of phase array → waveform)
SHAPES: dict[str, object] = {
    "sine":     lambda ph: np.sin(ph),
    "saw":      lambda ph: sawtooth(ph, width=1.0),
    "triangle": lambda ph: sawtooth(ph, width=0.5),
    "square":   lambda ph: sp_square(ph),
}

# SNR levels: tag = SNR in dB where noise RMS = signal_RMS * 10^(−tag/20)
SNR_TAGS: list[int] = [100, 80, 60, 40, 20]


def _normalize_to_peak(signal: np.ndarray, peak: float) -> np.ndarray:
    max_val = np.max(np.abs(signal))
    if max_val < 1e-12:
        return signal
    return signal * (peak / max_val)


def _add_noise_at_snr(
    signal: np.ndarray, snr_db: float, rng: np.random.Generator
) -> np.ndarray:
    """Add white noise so that signal_rms / noise_rms = 10^(snr_db/20)."""
    signal_rms = float(np.sqrt(np.mean(signal ** 2)))
    noise = rng.standard_normal(len(signal))
    noise_rms = float(np.sqrt(np.mean(noise ** 2)))
    if noise_rms < 1e-12 or signal_rms < 1e-12:
        return signal
    target_noise_rms = signal_rms * 10 ** (-snr_db / 20.0)  # SNR=dB means noise is that many dB below signal
    return signal + noise * (target_noise_rms / noise_rms)


def _write(path: Path, signal: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.clip(signal, -1.0, 1.0), SR, subtype="PCM_16")


def generate_all(rec_root: Path) -> None:
    rec_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    n_written = 0

    for shape_name, shape_fn in SHAPES.items():
        for freq in FREQ_TAGS:
            sig_name = f"snr_{shape_name}_{freq}hz"
            out_dir = rec_root / sig_name
            out_dir.mkdir(parents=True, exist_ok=True)

            # Clean baseline at −6 dBFS
            phase = 2 * np.pi * freq * T
            raw = shape_fn(phase)
            baseline = _normalize_to_peak(raw, BASELINE_PEAK)

            _write(out_dir / f"{sig_name}_baseline.wav", baseline)
            _write(out_dir / f"{sig_name}_repeat_1.wav", baseline)
            _write(out_dir / f"{sig_name}_repeat_2.wav", baseline)

            for snr in SNR_TAGS:
                noisy = _add_noise_at_snr(baseline, snr, rng)
                _write(out_dir / f"{sig_name}_var_{snr}.wav", noisy)

            print(f"  [{sig_name}]  SNR levels={SNR_TAGS}")
            n_written += 1

    n_files = n_written * (3 + len(SNR_TAGS))
    print(f"\n  {n_written} subdirs / {n_files} WAV files → {rec_root}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "rec_root",
        nargs="?",
        default="datasets/dsp_snr_recordings",
        help="Root directory for recordings",
    )
    args = ap.parse_args()
    generate_all(Path(args.rec_root))
    return 0


if __name__ == "__main__":
    sys.exit(main())

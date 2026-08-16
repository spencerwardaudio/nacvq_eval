"""End-to-end EnCodec latent sensitivity pipeline.

Experiments:
  time  — time-offset perturbations   (generate_timeoffsets.py)
  room  — RT60 / room IR convolution  (generate_room_ir_offsets.py)

Stages (per experiment, per source file):
  1. Generate perturbed audio recordings for every WAV in datasets/audio_files/
  2. Encode all WAVs to .ecdc bitstreams (EnCodec 48 kHz, 24 kbps)
  3. Convert .ecdc bitstreams to token .npy arrays
  4. Analyze token flip rates and save JSON stats (+ optional plots/PDF)
  5. Aggregate all per-file stats into a consolidated PDF report

Directory layout (all under <project_root>/datasets/):
  audio_files/                                   ← input WAVs
  <experiment>_recordings/<stem>/                ← step 1 output
  ecdc/<experiment>/<stem>/                      ← step 2 output
  audio_tokens/<experiment>/<stem>/              ← step 3 output
  analysis/<experiment>/<stem>/stats_bw*.json    ← step 4 output
  analysis/<experiment>/aggregate_report_*.pdf   ← step 5 output

Usage:
    cd tok_analysis/
    python run_pipeline.py --experiment time room  [--device cuda] [--bandwidth 24.0]
                           [--stats-only] [--cleanup]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
_ENCODEC_DIR = _PROJ_ROOT / "Encodec"

# Base paths
_AUDIO_FILES_DIR = _PROJ_ROOT / "datasets" / "audio_files"
_DATASETS_DIR = _PROJ_ROOT / "datasets"

# Experiment configs: map experiment name → (generator script, recordings folder prefix)
EXPERIMENT_MAP: dict[str, dict] = {
    "time": {
        "script": "generate_timeoffsets.py",
        "recordings_dir": "time_offset_recordings",
    },
    "room": {
        "script": "generate_room_ir_offsets.py",
        "recordings_dir": "room_ir_recordings",
    },
    "dsp": {
        "script": "generate_dsp_test_signals.py",
        "recordings_dir": "dsp_test_recordings",
    },
    "dsp_resilience": {
        "script": "generate_resilience_test_signals.py",
        "recordings_dir": "dsp_resilience_recordings",
    },
    "dsp_amplitude": {
        "script": "generate_amplitude_test_signals.py",
        "recordings_dir": "dsp_amplitude_recordings",
    },
    "dsp_frequency": {
        "script": "generate_frequency_test_signals.py",
        "recordings_dir": "dsp_frequency_recordings",
    },
    "dsp_timbre": {
        "script": "generate_timbre_test_signals.py",
        "recordings_dir": "dsp_timbre_recordings",
    },
    "dsp_self_amp": {
        "script": "generate_self_amp_signals.py",
        "recordings_dir": "dsp_self_amp_recordings",
    },
    "dsp_self_phase": {
        "script": "generate_self_phase_signals.py",
        "recordings_dir": "dsp_self_phase_recordings",
    },
    "dsp_snr": {
        "script": "generate_snr_signals.py",
        "recordings_dir": "dsp_snr_recordings",
    },
    "time_sine": {
        "script": "generate_sine_timeoffsets.py",
        "recordings_dir": "time_sine_recordings",
    },
}

ALL_EXPERIMENTS = list(EXPERIMENT_MAP.keys())


def _run(cmd: list[str], label: str, env: dict | None = None) -> bool:
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  $ {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"\n  ✗ FAILED: {label}")
        return False
    print(f"\n  ✓ OK: {label}")
    return True


def _encodec_env() -> dict:
    """Return environment with Encodec directory prepended to PYTHONPATH."""
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_ENCODEC_DIR}{os.pathsep}{existing}" if existing else str(_ENCODEC_DIR)
    return env


def _recordings_dir(experiment: str) -> Path:
    return _DATASETS_DIR / EXPERIMENT_MAP[experiment]["recordings_dir"]


def _ecdc_dir(experiment: str) -> Path:
    return _DATASETS_DIR / "ecdc" / experiment


def _tokens_dir(experiment: str) -> Path:
    return _DATASETS_DIR / "audio_tokens" / experiment


def _analysis_dir(experiment: str) -> Path:
    return _DATASETS_DIR / "analysis" / experiment


# ---------------------------------------------------------------------------
# Stage 1: Generate perturbed recordings
# ---------------------------------------------------------------------------

def stage_generate(experiment: str) -> bool:
    """Generate perturbed recordings using the experiment-specific script."""
    cfg = EXPERIMENT_MAP[experiment]
    script = _HERE / cfg["script"]
    rec_dir = _recordings_dir(experiment)

    # All generators support batch mode with no args (processes datasets/audio_files/)
    # They also support: <source_file> [output_dir] for single-file mode.
    # Batch mode is the default when invoked with no arguments.
    wav_files = sorted(_AUDIO_FILES_DIR.glob("*.wav"))
    if not wav_files:
        print(f"  No .wav files found in {_AUDIO_FILES_DIR}")
        return False

    print(f"  [{experiment}] Found {len(wav_files)} WAV file(s)")

    if experiment == "time":
        # Time offset generator takes (source, output_dir) per file
        all_ok = True
        for wav_file in wav_files:
            out_dir = rec_dir / wav_file.stem
            ok = _run(
                [sys.executable, str(script), str(wav_file), str(out_dir)],
                f"[{experiment}] Generate: {wav_file.name}",
            )
            all_ok = all_ok and ok
        return all_ok
    else:
        # Batch-mode generators (room IR, sine timeoffsets, dsp variants, etc.)
        return _run(
            [sys.executable, str(script)],
            f"[{experiment}] Generate all perturbations",
        )


# ---------------------------------------------------------------------------
# Stage 2: Encode to ECDC
# ---------------------------------------------------------------------------

def stage_encode(experiment: str, bandwidth: float, device: str,
                 model: str = "encodec_48khz",
                 checkpoint: str | None = None) -> bool:
    """Encode perturbed WAVs to .ecdc for each source.

    Loads the model once and reuses it across all subdirectories for speed.
    """
    # Q2D2 tokens are extracted directly from WAV in stage_tokenize; no ECDC step
    if model == "q2d2":
        print(f"  [{experiment}] Q2D2: skipping ECDC encode — tokens extracted from WAV in stage 3")
        return True

    rec_dir = _recordings_dir(experiment)
    subdirs = sorted(p for p in rec_dir.iterdir() if p.is_dir()) \
        if rec_dir.exists() else []
    if not subdirs:
        print(f"  [{experiment}] No subdirectories in {rec_dir} — run stage 1 first.")
        return False

    # Import batch encoder and load model once for all subdirs
    _encodec_env_setup()
    from batch_encode_24kbps import _load_model, batch_encode_audio_folder

    print(f"\n  Loading model {model} (bandwidth={bandwidth}, device={device}) ...")
    try:
        loaded_model = _load_model(model, checkpoint, bandwidth, device)
    except Exception as exc:
        print(f"  ✗ Failed to load model: {exc}")
        return False
    print(f"  Model loaded.\n")

    all_ok = True
    for subdir in subdirs:
        out_dir = _ecdc_dir(experiment) / subdir.name
        print(f"\n{'='*70}\n  [{experiment}] Encode: {subdir.name}\n{'='*70}")
        try:
            batch_encode_audio_folder(
                subdir, output_dir=out_dir,
                bandwidth=bandwidth, device=device,
                _model=loaded_model,
            )
        except Exception as exc:
            print(f"  ✗ FAILED: {exc}")
            all_ok = False
    return all_ok


def _encodec_env_setup():
    """Ensure Encodec directory is on sys.path."""
    encodec_dir = str(_ENCODEC_DIR)
    if encodec_dir not in sys.path:
        sys.path.insert(0, encodec_dir)
    here_dir = str(_HERE)
    if here_dir not in sys.path:
        sys.path.insert(0, here_dir)


# ---------------------------------------------------------------------------
# Stage 3: Tokenize
# ---------------------------------------------------------------------------

def stage_tokenize(experiment: str, device: str, model: str,
                   checkpoint: str | None, bw_str: str = "24.0") -> bool:
    """Convert encoded bitstreams to token .npy arrays."""
    tokens_dir = _tokens_dir(experiment)

    if model == "q2d2":
        rec_dir = _recordings_dir(experiment)
        subdirs = sorted(p for p in rec_dir.iterdir() if p.is_dir()) if rec_dir.exists() else []
        if not subdirs:
            print(f"  [{experiment}] No subdirectories in {rec_dir}")
            return False
        all_ok = True
        for subdir in subdirs:
            out_dir = tokens_dir / subdir.name
            cmd = [
                sys.executable, str(_HERE / "q2d2_to_tokens_npy.py"),
                "--input", str(subdir),
                "--output", str(out_dir),
                "--model", "q2d2",
                "--bandwidth", "9.8",
                "--device", device,
            ]
            if checkpoint:
                cmd.extend(["--checkpoint", str(checkpoint)])
            ok = _run(cmd, f"[{experiment}] Q2D2 tokenize: {subdir.name}")
            all_ok = all_ok and ok
        return all_ok

    ecdc_dir = _ecdc_dir(experiment)
    cmd = [
        sys.executable, str(_HERE / "ecdc_to_tokens_npy.py"),
        "--input", str(ecdc_dir),
        "--output-dir", str(tokens_dir),
        "--model-name", model,
        "--bw-tag", bw_str,
        "--device", device,
    ]
    if checkpoint:
        cmd.extend(["--checkpoint", checkpoint])
    return _run(cmd, f"[{experiment}] Tokenize ECDC → NPY", env=_encodec_env())


# ---------------------------------------------------------------------------
# Stage 4: Analyze
# ---------------------------------------------------------------------------

def stage_analyze(experiment: str, bandwidth: str,
                  stats_only: bool = False) -> bool:
    """Analyze token flip rates and save JSON stats (+ optional plots/PDF)."""
    tokens_dir = _tokens_dir(experiment)
    analysis_dir = _analysis_dir(experiment)
    subdirs = sorted(p for p in tokens_dir.iterdir() if p.is_dir()) \
        if tokens_dir.exists() else []
    if not subdirs:
        print(f"  [{experiment}] No subdirectories in {tokens_dir} — run stage 3 first.")
        return False

    all_ok = True
    for subdir in subdirs:
        out_dir = analysis_dir / subdir.name
        cmd = [sys.executable, str(_HERE / "analyze_token_flips.py"),
               str(subdir), str(out_dir), bandwidth]
        if stats_only:
            cmd.append("--stats-only")
        ok = _run(cmd, f"[{experiment}] Analyze: {subdir.name}")
        all_ok = all_ok and ok
    return all_ok


def stage_aggregate(experiment: str, bandwidth: str) -> bool:
    """Generate aggregate PDF report from all per-file JSON stats."""
    return _run(
        [sys.executable, str(_HERE / "aggregate_analysis.py"),
         "--experiment", experiment, "--bandwidth", bandwidth],
        f"[{experiment}] Aggregate report",
    )


def cleanup_intermediates(experiment: str, stem: str) -> None:
    """Delete intermediate recordings, ecdc, and token files for one source."""
    dirs_to_clean = [
        _recordings_dir(experiment) / stem,
        _ecdc_dir(experiment) / stem,
        _tokens_dir(experiment) / stem,
    ]
    for d in dirs_to_clean:
        if d.exists():
            shutil.rmtree(d)
            print(f"  Cleaned: {d}")


# ---------------------------------------------------------------------------
# Controlled DSP determinism / sensitivity test (--sig-unit-test)
# ---------------------------------------------------------------------------

def run_resilience_test(args) -> int:
    """Run noise-rotation + sine-phase resilience test pipeline.

    Stages: generate WAVs -> encode -> tokenize -> analyze (single combined PDF).
    Uses experiment key "dsp_resilience" so directory helpers reuse the standard layout.
    """
    exp = "dsp_resilience"
    bw_str = str(args.bandwidth)

    print(f"\n{'#'*70}")
    print(f"  EnCodec Codebook Resilience Test  (--resilience-test)")
    print(f"  Project root : {_PROJ_ROOT}")
    print(f"  Device       : {args.device}")
    print(f"  Bandwidth    : {args.bandwidth} kbps")
    print(f"  Model        : {args.model}")
    print(f"{'#'*70}\n")

    results: dict[str, bool] = {}

    if not args.skip_generate:
        rec_dir = _recordings_dir(exp)
        rec_dir.mkdir(parents=True, exist_ok=True)
        results["[dsp_resilience] 1. Generate signals"] = _run(
            [sys.executable, str(_HERE / "generate_resilience_test_signals.py"), str(rec_dir)],
            "[dsp_resilience] Generate rotation + phase test signals",
        )

    if not args.skip_encode:
        results["[dsp_resilience] 2. Encode"] = stage_encode(exp, args.bandwidth, args.device, args.model, args.checkpoint)

    if not args.skip_tokenize:
        results["[dsp_resilience] 3. Tokenize"] = stage_tokenize(
            exp, args.device, args.model, args.checkpoint, bw_str)

    if not args.skip_analyze:
        tokens_root = _tokens_dir(exp)
        out_root = _analysis_dir(exp)
        results["[dsp_resilience] 4. Analyze"] = _run(
            [sys.executable, str(_HERE / "analyze_resilience_test.py"),
             str(tokens_root), str(out_root), bw_str],
            "[dsp_resilience] Analyze rotation + phase resilience",
        )

    print(f"\n{'#'*70}")
    print(f"  Resilience-Test Summary")
    print(f"{'#'*70}")
    all_ok = True
    for stage, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status}  {stage}")
        all_ok = all_ok and ok
    print(f"\n  Output directories:")
    print(f"      Recordings: {_recordings_dir(exp)}")
    print(f"      ECDC:       {_ecdc_dir(exp)}")
    print(f"      Tokens:     {_tokens_dir(exp)}")
    print(f"      Analysis:   {_analysis_dir(exp)}")
    print(f"      Combined PDF: {_analysis_dir(exp) / f'results_bw{bw_str}.pdf'}")
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# Amplitude sweep test (--amplitude-test)
# ---------------------------------------------------------------------------

def run_amplitude_test(args) -> int:
    """White noise × 5 amplitude levels (-140 to 0 dBFS). Single combined PDF."""
    exp = "dsp_amplitude"
    bw_str = str(args.bandwidth)

    print(f"\n{'#'*70}")
    print(f"  EnCodec Amplitude Sweep Test  (--amplitude-test)")
    print(f"  Device    : {args.device}")
    print(f"  Bandwidth : {args.bandwidth} kbps")
    print(f"{'#'*70}\n")

    results: dict[str, bool] = {}

    if not args.skip_generate:
        rec_dir = _recordings_dir(exp)
        rec_dir.mkdir(parents=True, exist_ok=True)
        results["[amplitude] 1. Generate"] = _run(
            [sys.executable, str(_HERE / "generate_amplitude_test_signals.py"), str(rec_dir)],
            "[amplitude] Generate white noise amplitude sweep signals",
        )

    if not args.skip_encode:
        results["[amplitude] 2. Encode"] = stage_encode(exp, args.bandwidth, args.device, args.model, args.checkpoint)

    if not args.skip_tokenize:
        results["[amplitude] 3. Tokenize"] = stage_tokenize(
            exp, args.device, args.model, args.checkpoint, bw_str)

    if not args.skip_analyze:
        results["[amplitude] 4. Analyze"] = _run(
            [sys.executable, str(_HERE / "analyze_amplitude_test.py"),
             str(_tokens_dir(exp)), str(_analysis_dir(exp)), bw_str],
            "[amplitude] Analyze amplitude sweep",
        )

    _print_test_summary("Amplitude-Test", results, exp, bw_str)
    return 0 if all(results.values()) else 1


# ---------------------------------------------------------------------------
# Frequency sweep test (--frequency-test)
# ---------------------------------------------------------------------------

def run_frequency_test(args) -> int:
    """Sine × 20 log-spaced frequencies (10 Hz–20 kHz) × 3 amplitude levels."""
    exp = "dsp_frequency"
    bw_str = str(args.bandwidth)

    print(f"\n{'#'*70}")
    print(f"  EnCodec Frequency Sweep Test  (--frequency-test)")
    print(f"  Device    : {args.device}")
    print(f"  Bandwidth : {args.bandwidth} kbps")
    print(f"{'#'*70}\n")

    results: dict[str, bool] = {}

    if not args.skip_generate:
        rec_dir = _recordings_dir(exp)
        rec_dir.mkdir(parents=True, exist_ok=True)
        results["[frequency] 1. Generate"] = _run(
            [sys.executable, str(_HERE / "generate_frequency_test_signals.py"), str(rec_dir)],
            "[frequency] Generate sine frequency sweep signals",
        )

    if not args.skip_encode:
        results["[frequency] 2. Encode"] = stage_encode(exp, args.bandwidth, args.device, args.model, args.checkpoint)

    if not args.skip_tokenize:
        results["[frequency] 3. Tokenize"] = stage_tokenize(
            exp, args.device, args.model, args.checkpoint, bw_str)

    if not args.skip_analyze:
        results["[frequency] 4. Analyze"] = _run(
            [sys.executable, str(_HERE / "analyze_frequency_test.py"),
             str(_tokens_dir(exp)), str(_analysis_dir(exp)), bw_str],
            "[frequency] Analyze frequency sweep",
        )

    _print_test_summary("Frequency-Test", results, exp, bw_str)
    return 0 if all(results.values()) else 1


# ---------------------------------------------------------------------------
# Timbre (waveform shape) test (--timbre-test)
# ---------------------------------------------------------------------------

def run_timbre_test(args) -> int:
    """Sine / sawtooth / triangle / square at 1 kHz × 3 amplitude levels."""
    exp = "dsp_timbre"
    bw_str = str(args.bandwidth)

    print(f"\n{'#'*70}")
    print(f"  EnCodec Timbre (Waveform Shape) Test  (--timbre-test)")
    print(f"  Device    : {args.device}")
    print(f"  Bandwidth : {args.bandwidth} kbps")
    print(f"{'#'*70}\n")

    results: dict[str, bool] = {}

    if not args.skip_generate:
        rec_dir = _recordings_dir(exp)
        rec_dir.mkdir(parents=True, exist_ok=True)
        results["[timbre] 1. Generate"] = _run(
            [sys.executable, str(_HERE / "generate_timbre_test_signals.py"), str(rec_dir)],
            "[timbre] Generate waveform shape test signals",
        )

    if not args.skip_encode:
        results["[timbre] 2. Encode"] = stage_encode(exp, args.bandwidth, args.device, args.model, args.checkpoint)

    if not args.skip_tokenize:
        results["[timbre] 3. Tokenize"] = stage_tokenize(
            exp, args.device, args.model, args.checkpoint, bw_str)

    if not args.skip_analyze:
        results["[timbre] 4. Analyze"] = _run(
            [sys.executable, str(_HERE / "analyze_timbre_test.py"),
             str(_tokens_dir(exp)), str(_analysis_dir(exp)), bw_str],
            "[timbre] Analyze waveform shape (timbre) test",
        )

    _print_test_summary("Timbre-Test", results, exp, bw_str)
    return 0 if all(results.values()) else 1


# ---------------------------------------------------------------------------
# Self-amplitude modulation test (--self-amp-test)
# ---------------------------------------------------------------------------

def run_self_amp_test(args) -> int:
    """Each of 20 frequencies compared against itself at 4 amplitude levels."""
    exp = "dsp_self_amp"
    bw_str = str(args.bandwidth)

    print(f"\n{'#'*70}")
    print(f"  EnCodec Self-Amplitude Test  (--self-amp-test)")
    print(f"  Device    : {args.device}")
    print(f"  Bandwidth : {args.bandwidth} kbps")
    print(f"{'#'*70}\n")

    results: dict[str, bool] = {}

    if not args.skip_generate:
        rec_dir = _recordings_dir(exp)
        rec_dir.mkdir(parents=True, exist_ok=True)
        results["[self_amp] 1. Generate"] = _run(
            [sys.executable, str(_HERE / "generate_self_amp_signals.py"), str(rec_dir)],
            "[self_amp] Generate self-amplitude test signals",
        )

    if not args.skip_encode:
        results["[self_amp] 2. Encode"] = stage_encode(exp, args.bandwidth, args.device, args.model, args.checkpoint)

    if not args.skip_tokenize:
        results["[self_amp] 3. Tokenize"] = stage_tokenize(
            exp, args.device, args.model, args.checkpoint, bw_str)

    if not args.skip_analyze:
        results["[self_amp] 4. Analyze"] = _run(
            [sys.executable, str(_HERE / "analyze_self_amp_test.py"),
             str(_tokens_dir(exp)), str(_analysis_dir(exp)),
             "--bandwidth", bw_str],
            "[self_amp] Analyze self-amplitude test",
        )

    _print_test_summary("Self-Amp-Test", results, exp, bw_str)
    return 0 if all(results.values()) else 1


# ---------------------------------------------------------------------------
# Self-phase modulation test (--self-phase-test)
# ---------------------------------------------------------------------------

def run_self_phase_test(args) -> int:
    """Each of 20 frequencies at 2 amplitude levels × 8 phase offsets."""
    exp = "dsp_self_phase"
    bw_str = str(args.bandwidth)

    print(f"\n{'#'*70}")
    print(f"  EnCodec Self-Phase Test  (--self-phase-test)")
    print(f"  Device    : {args.device}")
    print(f"  Bandwidth : {args.bandwidth} kbps")
    print(f"{'#'*70}\n")

    results: dict[str, bool] = {}

    if not args.skip_generate:
        rec_dir = _recordings_dir(exp)
        rec_dir.mkdir(parents=True, exist_ok=True)
        results["[self_phase] 1. Generate"] = _run(
            [sys.executable, str(_HERE / "generate_self_phase_signals.py"), str(rec_dir)],
            "[self_phase] Generate self-phase test signals",
        )

    if not args.skip_encode:
        results["[self_phase] 2. Encode"] = stage_encode(exp, args.bandwidth, args.device, args.model, args.checkpoint)

    if not args.skip_tokenize:
        results["[self_phase] 3. Tokenize"] = stage_tokenize(
            exp, args.device, args.model, args.checkpoint, bw_str)

    if not args.skip_analyze:
        results["[self_phase] 4. Analyze"] = _run(
            [sys.executable, str(_HERE / "analyze_self_phase_test.py"),
             str(_tokens_dir(exp)), str(_analysis_dir(exp)),
             "--bandwidth", bw_str],
            "[self_phase] Analyze self-phase test",
        )

    _print_test_summary("Self-Phase-Test", results, exp, bw_str)
    return 0 if all(results.values()) else 1


# ---------------------------------------------------------------------------
# SNR noise tolerance test (--snr-test)
# ---------------------------------------------------------------------------

def run_snr_test(args) -> int:
    """4 waveform shapes × 20 frequencies with white noise at 5 SNR levels."""
    exp = "dsp_snr"
    bw_str = str(args.bandwidth)

    print(f"\n{'#'*70}")
    print(f"  EnCodec SNR Noise Tolerance Test  (--snr-test)")
    print(f"  Device    : {args.device}")
    print(f"  Bandwidth : {args.bandwidth} kbps")
    print(f"{'#'*70}\n")

    results: dict[str, bool] = {}

    if not args.skip_generate:
        rec_dir = _recordings_dir(exp)
        rec_dir.mkdir(parents=True, exist_ok=True)
        results["[snr] 1. Generate"] = _run(
            [sys.executable, str(_HERE / "generate_snr_signals.py"), str(rec_dir)],
            "[snr] Generate SNR test signals",
        )

    if not args.skip_encode:
        results["[snr] 2. Encode"] = stage_encode(exp, args.bandwidth, args.device, args.model, args.checkpoint)

    if not args.skip_tokenize:
        results["[snr] 3. Tokenize"] = stage_tokenize(
            exp, args.device, args.model, args.checkpoint, bw_str)

    if not args.skip_analyze:
        results["[snr] 4. Analyze"] = _run(
            [sys.executable, str(_HERE / "analyze_snr_test.py"),
             str(_tokens_dir(exp)), str(_analysis_dir(exp)), bw_str],
            "[snr] Analyze SNR noise tolerance test",
        )

    _print_test_summary("SNR-Test", results, exp, bw_str)
    return 0 if all(results.values()) else 1


# ---------------------------------------------------------------------------
# Sine temporal test (--sine-temporal-test)
# ---------------------------------------------------------------------------

def run_sine_temporal_test(args) -> int:
    """20 pure sine frequencies × 25 time offsets (1–20 ms + 40/60/80/100 ms)."""
    exp = "time_sine"
    bw_str = str(args.bandwidth)

    print(f"\n{'#'*70}")
    print(f"  EnCodec Sine Temporal Sensitivity Test  (--sine-temporal-test)")
    print(f"  Device    : {args.device}")
    print(f"  Bandwidth : {args.bandwidth} kbps")
    print(f"{'#'*70}\n")

    results: dict[str, bool] = {}

    if not args.skip_generate:
        rec_dir = _recordings_dir(exp)
        rec_dir.mkdir(parents=True, exist_ok=True)
        results["[time_sine] 1. Generate"] = _run(
            [sys.executable, str(_HERE / "generate_sine_timeoffsets.py"), str(rec_dir)],
            "[time_sine] Generate sine wave time-offset signals",
        )

    if not args.skip_encode:
        results["[time_sine] 2. Encode"] = stage_encode(exp, args.bandwidth, args.device, args.model, args.checkpoint)

    if not args.skip_tokenize:
        results["[time_sine] 3. Tokenize"] = stage_tokenize(
            exp, args.device, args.model, args.checkpoint, bw_str)

    if not args.skip_analyze:
        results["[time_sine] 4. Analyze"] = _run(
            [sys.executable, str(_HERE / "analyze_sine_temporal.py"),
             str(_tokens_dir(exp)), str(_analysis_dir(exp)),
             "--bandwidth", bw_str],
            "[time_sine] Analyze sine temporal sensitivity",
        )

    _print_test_summary("Sine-Temporal-Test", results, exp, bw_str)
    return 0 if all(results.values()) else 1


# ---------------------------------------------------------------------------
# Combined sensitivity report (--combined-report)
# ---------------------------------------------------------------------------

def run_combined_report(args) -> int:
    """Build combined PDF from amplitude, phase, and temporal sine token data."""
    bw_str = str(args.bandwidth)
    out_pdf = (_DATASETS_DIR / "analysis"
               / f"combined_sensitivity_report_bw{bw_str}.pdf")

    cmd = [
        sys.executable, str(_HERE / "report_combined_sensitivity.py"),
        "--amp-tokens",      str(_tokens_dir("dsp_self_amp")),
        "--phase-tokens",    str(_tokens_dir("dsp_self_phase")),
        "--temporal-tokens", str(_tokens_dir("time_sine")),
        "--bandwidth",       bw_str,
        "--model",           args.model,
        "--output",          str(out_pdf),
    ]
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    ok = _run(cmd, "Build combined sensitivity report")
    if ok:
        print(f"\n  Combined report: {out_pdf}")
    return 0 if ok else 1


def _print_test_summary(label: str, results: dict[str, bool], exp: str, bw_str: str) -> None:
    print(f"\n{'#'*70}")
    print(f"  {label} Summary")
    print(f"{'#'*70}")
    for stage, ok in results.items():
        print(f"  {'✓' if ok else '✗'}  {stage}")
    print(f"\n  Output directories:")
    print(f"      Recordings: {_recordings_dir(exp)}")
    print(f"      ECDC:       {_ecdc_dir(exp)}")
    print(f"      Tokens:     {_tokens_dir(exp)}")
    print(f"      Analysis:   {_analysis_dir(exp)}")
    print(f"      Combined PDF: {_analysis_dir(exp) / f'results_bw{bw_str}.pdf'}")


def run_sig_unit_test(args) -> int:
    """Run the controlled-signal determinism + roundoff sensitivity pipeline.

    Stages: generate synthetic WAVs → encode → tokenize → analyze (single PDF).
    Uses experiment key "dsp" so directory helpers (_recordings_dir, _ecdc_dir,
    _tokens_dir, _analysis_dir) reuse the standard layout.
    """
    exp = "dsp"
    bw_str = str(args.bandwidth)

    print(f"\n{'#'*70}")
    print(f"  EnCodec Controlled-Signal Test  (--sig-unit-test)")
    print(f"  Project root : {_PROJ_ROOT}")
    print(f"  Device       : {args.device}")
    print(f"  Bandwidth    : {args.bandwidth} kbps")
    print(f"  Model        : {args.model}")
    print(f"{'#'*70}\n")

    results: dict[str, bool] = {}

    if not args.skip_generate:
        rec_dir = _recordings_dir(exp)
        rec_dir.mkdir(parents=True, exist_ok=True)
        results["[dsp] 1. Generate signals"] = _run(
            [sys.executable, str(_HERE / "generate_dsp_test_signals.py"), str(rec_dir)],
            "[dsp] Generate synthetic test signals",
        )

    if not args.skip_encode:
        results["[dsp] 2. Encode"] = stage_encode(exp, args.bandwidth, args.device, args.model, args.checkpoint)

    if not args.skip_tokenize:
        results["[dsp] 3. Tokenize"] = stage_tokenize(
            exp, args.device, args.model, args.checkpoint, bw_str)

    if not args.skip_analyze:
        tokens_root = _tokens_dir(exp)
        out_root = _analysis_dir(exp)
        results["[dsp] 4. Analyze"] = _run(
            [sys.executable, str(_HERE / "analyze_dsp_test.py"),
             str(tokens_root), str(out_root), bw_str],
            "[dsp] Analyze determinism + roundoff sensitivity",
        )

    print(f"\n{'#'*70}")
    print(f"  Sig-Unit-Test Summary")
    print(f"{'#'*70}")
    all_ok = True
    for stage, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status}  {stage}")
        all_ok = all_ok and ok
    print(f"\n  Output directories:")
    print(f"      Recordings: {_recordings_dir(exp)}")
    print(f"      ECDC:       {_ecdc_dir(exp)}")
    print(f"      Tokens:     {_tokens_dir(exp)}")
    print(f"      Analysis:   {_analysis_dir(exp)}")
    print(f"      Combined PDF: {_analysis_dir(exp) / f'results_bw{bw_str}.pdf'}")
    return 0 if all_ok else 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run EnCodec latent sensitivity pipeline for one or more experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment", "-e", nargs="+",
                        choices=ALL_EXPERIMENTS + ["all"], default=["all"],
                        help="Which experiment(s) to run")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device for EnCodec inference")
    parser.add_argument("--bandwidth", type=float, default=24.0,
                        help="EnCodec bandwidth (kbps)")
    parser.add_argument("--model", default="encodec_48khz",
                        help="EnCodec model name")
    parser.add_argument("--checkpoint", default=None,
                        help="Optional checkpoint path (for multi_dataset_encodec)")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip stage 1 (perturbation generation)")
    parser.add_argument("--skip-encode", action="store_true",
                        help="Skip stage 2 (ECDC encoding)")
    parser.add_argument("--skip-tokenize", action="store_true",
                        help="Skip stage 3 (tokenization)")
    parser.add_argument("--skip-analyze", action="store_true",
                        help="Skip stage 4 (token flip analysis)")
    parser.add_argument("--skip-aggregate", action="store_true",
                        help="Skip stage 5 (aggregate PDF report)")
    parser.add_argument("--stats-only", action="store_true",
                        help="Skip per-file PDF/plots, only save JSON stats")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete intermediate files (recordings/ecdc/tokens) after analysis")
    parser.add_argument("--sig-unit-test", action="store_true",
                        help="Run controlled DSP determinism / roundoff sensitivity test "
                             "(synthetic sine, impulse, white noise, chirp). Bypasses --experiment.")
    parser.add_argument("--resilience-test", action="store_true",
                        help="Run codebook resilience test: circular noise rotation "
                             "(white/pink/brown) + 1 kHz sine phase offset. Bypasses --experiment.")
    parser.add_argument("--amplitude-test", action="store_true",
                        help="Run amplitude sweep test: white noise at 5 levels "
                             "(-140 to 0 dBFS). Bypasses --experiment.")
    parser.add_argument("--frequency-test", action="store_true",
                        help="Run frequency sweep test: sine at 20 log-spaced frequencies "
                             "(10 Hz–20 kHz) × 3 amplitude levels. Bypasses --experiment.")
    parser.add_argument("--timbre-test", action="store_true",
                        help="Run waveform-shape (timbre) test: sine/saw/triangle/square "
                             "at 1 kHz × 3 amplitude levels. Bypasses --experiment.")
    parser.add_argument("--self-amp-test", action="store_true",
                        help="Run self-amplitude test: each of 20 frequencies compared "
                             "against itself at 4 amplitude levels. Bypasses --experiment.")
    parser.add_argument("--self-phase-test", action="store_true",
                        help="Run self-phase test: each of 20 frequencies at 2 amplitude "
                             "levels × 8 phase offsets. Bypasses --experiment.")
    parser.add_argument("--snr-test", action="store_true",
                        help="Run SNR noise tolerance test: 4 shapes × 20 frequencies "
                             "with additive white noise at 5 SNR levels. Bypasses --experiment.")
    parser.add_argument("--sine-temporal-test", action="store_true",
                        help="Run pure-sine temporal sensitivity test: 20 log-spaced "
                             "frequencies × 25 time offsets (1–20 ms + 40/60/80/100 ms). "
                             "Bypasses --experiment.")
    parser.add_argument("--combined-report", action="store_true",
                        help="Build combined sensitivity PDF from amplitude, phase, and "
                             "temporal sine token data. Runs report_combined_sensitivity.py. "
                             "Bypasses --experiment.")
    args = parser.parse_args()

    if args.sig_unit_test:
        rc = run_sig_unit_test(args)
        sys.exit(rc)

    if args.resilience_test:
        rc = run_resilience_test(args)
        sys.exit(rc)

    if args.amplitude_test:
        rc = run_amplitude_test(args)
        sys.exit(rc)

    if args.frequency_test:
        rc = run_frequency_test(args)
        sys.exit(rc)

    if args.timbre_test:
        rc = run_timbre_test(args)
        sys.exit(rc)

    if args.self_amp_test:
        rc = run_self_amp_test(args)
        sys.exit(rc)

    if args.self_phase_test:
        rc = run_self_phase_test(args)
        sys.exit(rc)

    if args.snr_test:
        rc = run_snr_test(args)
        sys.exit(rc)

    if args.sine_temporal_test:
        rc = run_sine_temporal_test(args)
        sys.exit(rc)

    if args.combined_report:
        rc = run_combined_report(args)
        sys.exit(rc)

    experiments = ALL_EXPERIMENTS if "all" in args.experiment else args.experiment

    print(f"\n{'#'*70}")
    print(f"  EnCodec Latent Sensitivity Pipeline")
    print(f"  Project root : {_PROJ_ROOT}")
    print(f"  Experiments  : {', '.join(experiments)}")
    print(f"  Device       : {args.device}")
    print(f"  Bandwidth    : {args.bandwidth} kbps")
    print(f"  Model        : {args.model}")
    print(f"{'#'*70}\n")

    results: dict[str, bool] = {}
    bw_str = str(args.bandwidth)

    for exp in experiments:
        print(f"\n{'*'*70}")
        print(f"  EXPERIMENT: {exp}")
        print(f"{'*'*70}")

        # Get source stems for per-file processing
        wav_files = sorted(_AUDIO_FILES_DIR.glob("*.wav"))
        stems = [w.stem for w in wav_files]

        if not args.skip_generate:
            results[f"[{exp}] 1. Generate"] = stage_generate(exp)

        if not args.skip_encode:
            results[f"[{exp}] 2. Encode"] = stage_encode(exp, args.bandwidth, args.device, args.model, args.checkpoint)

        if not args.skip_tokenize:
            results[f"[{exp}] 3. Tokenize"] = stage_tokenize(
                exp, args.device, args.model, args.checkpoint, bw_str)

        if not args.skip_analyze:
            results[f"[{exp}] 4. Analyze"] = stage_analyze(
                exp, bw_str, stats_only=args.stats_only)

        # Cleanup intermediate files per source if requested
        if args.cleanup:
            print(f"\n  [{exp}] Cleaning up intermediate files...")
            for stem in stems:
                cleanup_intermediates(exp, stem)

        if not args.skip_aggregate:
            results[f"[{exp}] 5. Aggregate"] = stage_aggregate(exp, bw_str)

    # Summary
    print(f"\n{'#'*70}")
    print(f"  Pipeline Summary")
    print(f"{'#'*70}")
    all_ok = True
    for stage, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status}  {stage}")
        all_ok = all_ok and ok

    print(f"\n  Output directories (per experiment):")
    for exp in experiments:
        print(f"    [{exp}]")
        if not args.cleanup:
            print(f"      Recordings: {_recordings_dir(exp)}")
            print(f"      ECDC:       {_ecdc_dir(exp)}")
            print(f"      Tokens:     {_tokens_dir(exp)}")
        print(f"      Analysis:   {_analysis_dir(exp)}")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

"""Generate mono recordings convolved with different Room Impulse Responses.

Simulates the same source/mic setup in different rooms (meshes) to produce
recordings with varying RT60 / room acoustics.  Can also use external SOFA
RIR files directly.

Output structure:
    <output_dir>/
        baseline_Oyens.wav
        room_Scottsmoor.wav
        room_Sumas.wav
        sofa_daga_foa.wav
        ...

Usage:
    python generate_room_ir_offsets.py [source_audio] [output_dir]

    No args → batch mode: processes all WAVs in datasets/audio_files/,
    outputs to datasets/room_ir_recordings/<stem>/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
_AL_ROOT = _PROJ_ROOT / "AudibleLight"
_TEST_RESOURCES = _AL_ROOT / "tests" / "test_resources"


def _ensure_audiblelight():
    if str(_AL_ROOT) not in sys.path:
        sys.path.insert(0, str(_AL_ROOT))


# ---------------------------------------------------------------------------
# Method 1: Re-simulate in different room meshes (RLR backend)
# ---------------------------------------------------------------------------

def _generate_from_mesh(
    source_file: Path,
    mesh_path: Path,
    output_path: Path,
    sample_rate: int = 48000,
    duration: float = 5.0,
    mic_pos: list[float] | None = None,
) -> Path | None:
    """Generate a mono recording using RLR ray-tracing in a specific room mesh."""
    _ensure_audiblelight()
    from audiblelight.core import Scene

    if mic_pos is None:
        mic_pos = [2.0, -3.5, 1.0]

    try:
        scene = Scene(
            duration=duration,
            sample_rate=sample_rate,
            backend="rlr",
            backend_kwargs=dict(
                mesh=str(mesh_path),
                add_to_context=False,
                empty_space_around_mic=0.0,
                empty_space_around_surface=0.0,
            ),
            fg_path=str(_TEST_RESOURCES / "soundevents"),
            ref_db=-50,
        )

        scene.add_microphone(
            microphone_type="monocapsule",
            alias="mic",
            position=mic_pos,
        )

        scene.add_event(
            event_type="static",
            filepath=str(source_file),
            alias="src",
            scene_start=0.0,
            duration=duration,
            shape="static",
            ensure_direct_path=False,
        )

        scene.generate(audio=True, metadata_json=False, metadata_dcase=False)
        audio = scene.audio["mic"].squeeze()

        n_samples = int(duration * sample_rate)
        if len(audio) > n_samples:
            audio = audio[:n_samples]  # trim to exact duration in case RLR adds tail
        elif len(audio) < n_samples:
            audio = np.pad(audio, (0, n_samples - len(audio)))  # zero-pad if simulation produced a short buffer

        sf.write(str(output_path), audio, sample_rate)
        return output_path

    except Exception as exc:
        print(f"    [FAIL] mesh={mesh_path.name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Method 2: Direct convolution with external IR (SOFA or raw numpy)
# ---------------------------------------------------------------------------

def _generate_from_ir(
    source_file: Path,
    ir: np.ndarray,
    output_path: Path,
    sample_rate: int = 48000,
    duration: float = 5.0,
) -> Path:
    """Convolve source audio with a provided IR and save as mono WAV."""
    from scipy.signal import fftconvolve

    audio, sr = sf.read(source_file)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * sample_rate / sr))

    # Ensure IR is 1D
    if ir.ndim > 1:
        ir = ir[:, 0] if ir.shape[1] < ir.shape[0] else ir[0, :]

    convolved = fftconvolve(audio, ir, mode="full")
    n_samples = int(duration * sample_rate)
    convolved = convolved[:n_samples]

    # Normalize
    peak = np.abs(convolved).max()
    if peak > 0:
        convolved = convolved / (peak + 1e-10) * 0.95

    sf.write(str(output_path), convolved, sample_rate)
    return output_path


def _load_sofa_ir(sofa_path: Path, source_idx: int = 0) -> np.ndarray | None:
    """Load a single IR from a SOFA file. Returns 1D array or None."""
    try:
        import netCDF4
        ds = netCDF4.Dataset(str(sofa_path), "r")
        # SOFA stores IRs in Data.IR with shape (M, R, N) — M=measurements, R=receivers, N=samples
        ir_data = ds.variables["Data.IR"][:]
        ds.close()
        # Take first measurement, first receiver
        idx = min(source_idx, ir_data.shape[0] - 1)
        ir = ir_data[idx, 0, :]
        return np.asarray(ir, dtype=np.float64)
    except ImportError:
        print("    [WARN] netCDF4 not installed — cannot read SOFA files")
        print("    Install with: pip3 install netCDF4")
        return None
    except Exception as exc:
        print(f"    [FAIL] SOFA load {sofa_path.name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Synthetic RT60 variation via exponential decay envelope
# ---------------------------------------------------------------------------

def _generate_synthetic_rt60_ir(
    rt60_s: float,
    sample_rate: int = 48000,
    ir_length_s: float = 1.0,
) -> np.ndarray:
    """
    Create a synthetic IR: direct impulse + exponentially decaying noise.
    Useful when no real RIRs are available, to test how RT60 magnitude
    affects token sensitivity.
    """
    n_samples = int(ir_length_s * sample_rate)
    t = np.arange(n_samples) / sample_rate

    # Exponential decay: amplitude = exp(-6.908 * t / RT60)
    # (6.908 = ln(10^3) so energy drops 60dB in RT60 seconds)
    decay = np.exp(-6.908 * t / max(rt60_s, 0.01))

    # Random noise envelope
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 1, n_samples)
    ir = noise * decay

    # Add direct impulse at sample 0
    ir[0] = 1.0

    # Normalize energy
    ir = ir / (np.sqrt(np.sum(ir ** 2)) + 1e-10)
    return ir


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_room_ir_recordings(
    source_file: Path,
    output_dir: Path,
    sample_rate: int = 48000,
    duration: float = 5.0,
) -> dict[str, Path]:
    """
    Generate recordings with different room acoustics:
      1. Different mesh rooms (via RLR backend)
      2. SOFA RIR files (if available)
      3. Synthetic RT60 sweeps (always available)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recordings: dict[str, Path] = {}

    # --- 0. Baseline: copy dry original (no reverb / RT60=0) ---
    audio, sr = sf.read(source_file)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * sample_rate / sr))
    n_samples = int(duration * sample_rate)
    if len(audio) > n_samples:
        audio = audio[:n_samples]
    elif len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    baseline_path = output_dir / "rt60_0.0s.wav"
    sf.write(str(baseline_path), audio, sample_rate)
    recordings["rt60_0.0s"] = baseline_path
    print(f"    [OK] rt60_0.0s.wav (dry baseline)")

    # --- 1. Mesh-based rooms (RLR) ---
    mesh_dir = _TEST_RESOURCES / "meshes"
    meshes = sorted(mesh_dir.glob("*.glb")) if mesh_dir.exists() else []

    if meshes:
        print(f"  Mesh rooms: {len(meshes)} available")
        baseline_mesh = None
        for mesh_path in meshes:
            room_name = mesh_path.stem
            label = f"baseline_{room_name}" if baseline_mesh is None else f"room_{room_name}"
            if baseline_mesh is None:
                baseline_mesh = mesh_path

            out_path = output_dir / f"{label}.wav"
            result = _generate_from_mesh(source_file, mesh_path, out_path,
                                         sample_rate, duration)
            if result:
                recordings[label] = result
                print(f"    [OK] {label}.wav")

    # --- 2. SOFA RIR files ---
    sofa_dirs = [
        _TEST_RESOURCES,
        _PROJ_ROOT / "resources",
    ]
    sofa_files = []
    for d in sofa_dirs:
        if d.exists():
            sofa_files.extend(d.glob("**/*.sofa"))

    if sofa_files:
        print(f"  SOFA RIRs: {len(sofa_files)} available")
        for sofa_path in sorted(sofa_files):
            ir = _load_sofa_ir(sofa_path)
            if ir is not None:
                label = f"sofa_{sofa_path.stem}"
                out_path = output_dir / f"{label}.wav"
                _generate_from_ir(source_file, ir, out_path, sample_rate, duration)
                recordings[label] = out_path
                print(f"    [OK] {label}.wav")

    # --- 3. Synthetic RT60 sweep ---
    rt60_values = [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    print(f"  Synthetic RT60 sweep: {len(rt60_values)} values")
    for rt60 in rt60_values:
        ir = _generate_synthetic_rt60_ir(rt60, sample_rate)
        label = f"rt60_{rt60:.1f}s"
        out_path = output_dir / f"{label}.wav"
        _generate_from_ir(source_file, ir, out_path, sample_rate, duration)
        recordings[label] = out_path
        print(f"    [OK] {label}.wav")

    return recordings


def batch_generate_room_ir(
    input_dir: Path,
    output_base: Path,
    **kwargs,
) -> dict[str, dict[str, Path]]:
    """Process all .wav files in input_dir."""
    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files in {input_dir}")
        return {}

    print(f"Batch Room IR: {len(wav_files)} file(s) from {input_dir}")
    all_recordings = {}
    for wav_file in wav_files:
        out_dir = output_base / wav_file.stem
        print(f"\n--- {wav_file.name} -> {out_dir} ---")
        recs = generate_room_ir_recordings(wav_file, out_dir, **kwargs)
        all_recordings[wav_file.stem] = recs
    return all_recordings


if __name__ == "__main__":
    DEFAULT_INPUT = _PROJ_ROOT / "datasets" / "audio_files"
    DEFAULT_OUTPUT = _PROJ_ROOT / "datasets" / "room_ir_recordings"

    if len(sys.argv) >= 2:
        source = Path(sys.argv[1])
        out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT / source.stem
        generate_room_ir_recordings(source, out)
    else:
        batch_generate_room_ir(DEFAULT_INPUT, DEFAULT_OUTPUT)
        print(f"\nAll done. Outputs in {DEFAULT_OUTPUT}")

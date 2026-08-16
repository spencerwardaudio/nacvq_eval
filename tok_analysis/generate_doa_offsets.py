"""Generate mono recordings at varying azimuths and elevations.

Uses AudibleLight's RLR backend to ray-trace a scene with a MonoCapsule mic
and a static source placed at different DOA angles relative to the mic.

Output structure:
    <output_dir>/
        baseline_az000_el000.wav       (azimuth=0, elevation=0)
        az045_el000.wav
        az090_el000.wav
        ...
        az000_el030.wav
        ...

Usage:
    python generate_doa_offsets.py [source_audio] [output_dir]

    No args → batch mode: processes all WAVs in datasets/audio_files/,
    outputs to datasets/doa_recordings/<stem>/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent


def _get_audiblelight():
    """Import AudibleLight, adding to path if needed."""
    al_root = _PROJ_ROOT / "AudibleLight"
    if str(al_root) not in sys.path:
        sys.path.insert(0, str(al_root))
    from audiblelight.core import Scene
    from audiblelight.micarrays import MonoCapsule
    return Scene


def generate_doa_recordings(
    source_file: Path,
    output_dir: Path,
    sample_rate: int = 48000,
    duration: float = 5.0,
    distance_m: float = 1.22,
    mesh: str | None = None,
    azimuths: list[int] | None = None,
    elevations: list[int] | None = None,
) -> dict[str, Path]:
    """
    Generate one mono WAV per (azimuth, elevation) pair.

    Args:
        source_file: Input WAV (dry source audio).
        output_dir:  Where to write the output WAVs.
        sample_rate: Target sample rate.
        duration:    Scene duration in seconds.
        distance_m:  Source distance from mic (meters).
        mesh:        Path to .glb mesh.  None → use default test mesh.
        azimuths:    List of azimuth angles (degrees). Default: 0-360 in 15° steps.
        elevations:  List of elevation angles (degrees). Default: [-30,-15,0,15,30].

    Returns:
        Dict mapping label → output path.
    """
    Scene = _get_audiblelight()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mesh is None:
        mesh = str(_PROJ_ROOT / "AudibleLight" / "tests" / "test_resources" / "meshes" / "Oyens.glb")

    if azimuths is None:
        azimuths = list(range(0, 360, 15))  # 24 azimuths at 15° steps — coarser than ITD JND but covers full sphere
    if elevations is None:
        elevations = [-30, -15, 0, 15, 30]  # limited elevation range typical of frontal sound sources

    mic_pos = [2.0, -3.5, 1.0]

    recordings: dict[str, Path] = {}
    total = len(azimuths) * len(elevations)
    count = 0

    for el in elevations:
        for az in azimuths:
            count += 1
            label = f"az{az:03d}_el{el:+03d}"
            if az == 0 and el == 0:
                label = "baseline_az000_el000"  # consistent with other generators’ _baseline naming convention

            scene = Scene(
                duration=duration,
                sample_rate=sample_rate,
                backend="rlr",
                backend_kwargs=dict(
                    mesh=mesh,
                    add_to_context=False,
                    empty_space_around_mic=0.0,
                    empty_space_around_surface=0.0,
                ),
                fg_path=str(_PROJ_ROOT / "AudibleLight" / "tests" / "test_resources" / "soundevents"),
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
                position=[float(az), float(el), distance_m],
                polar=True,
                mic="mic",
                scene_start=0.0,
                duration=duration,
                shape="static",
                ensure_direct_path=False,
            )

            scene.generate(audio=True, metadata_json=False, metadata_dcase=False)
            audio = scene.audio["mic"].squeeze()

            # Trim/pad to exact length
            n_samples = int(duration * sample_rate)
            if len(audio) > n_samples:
                audio = audio[:n_samples]
            elif len(audio) < n_samples:
                audio = np.pad(audio, (0, n_samples - len(audio)))

            fname = f"{label}.wav"
            out_path = output_dir / fname
            sf.write(str(out_path), audio, sample_rate)
            recordings[label] = out_path
            print(f"  [{count}/{total}] {fname}")

    return recordings


def batch_generate_doa(
    input_dir: Path,
    output_base: Path,
    **kwargs,
) -> dict[str, dict[str, Path]]:
    """Process all .wav files in input_dir."""
    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files in {input_dir}")
        return {}

    print(f"Batch DOA: {len(wav_files)} file(s) from {input_dir}")
    all_recordings = {}
    for wav_file in wav_files:
        out_dir = output_base / wav_file.stem
        print(f"\n--- {wav_file.name} -> {out_dir} ---")
        recs = generate_doa_recordings(wav_file, out_dir, **kwargs)
        all_recordings[wav_file.stem] = recs
    return all_recordings


if __name__ == "__main__":
    DEFAULT_INPUT = _PROJ_ROOT / "datasets" / "audio_files"
    DEFAULT_OUTPUT = _PROJ_ROOT / "datasets" / "doa_recordings"

    if len(sys.argv) >= 2:
        source = Path(sys.argv[1])
        out = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT / source.stem
        generate_doa_recordings(source, out)
    else:
        batch_generate_doa(DEFAULT_INPUT, DEFAULT_OUTPUT)
        print(f"\nAll done. Outputs in {DEFAULT_OUTPUT}")

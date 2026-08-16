from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
_DEFAULT_OUTPUT = _PROJ_ROOT / "datasets" / "analysis" / "benchmark_manifest.csv"

_SPEECH_DATASETS = (
    "libritts", "librispeech", "common_voice", "commonvoice", "vctk",
    "speech_commands", "libricount", "crema_d",
)
_MUSIC_DATASETS = ("musdb", "jamendo", "gtzan", "nsynth")
_GENERAL_DATASETS = (
    "audioset", "fsd50k", "esc50", "urbansound", "ecdc",
    "vocal_imitations",
)

_MACHINE_HINTS = ("car", "truck", "bus", "vehicle", "engine", "motor", "machine", "train", "airplane")
_FOOTSTEP_HINTS = ("footstep", "footsteps", "walk", "walking", "step", "steps")
_ENV_HINTS = ("rain", "wind", "thunder", "bird", "birds", "water", "ocean", "fire", "door", "knock")
_SYNTH_HINTS = ("sine", "square", "triangle", "saw", "chirp", "noise", "synth", "synthetic")


def _normalize_tokens(path: Path) -> list[str]:
    text = path.as_posix().lower()
    return re.split(r"[^a-z0-9]+", text)  # split on all non-alphanumeric chars to get searchable word tokens


def _infer_dataset(path: Path) -> str:
    tokens = _normalize_tokens(path)
    for token in tokens:  # match full path tokens to known dataset names before falling back to parent dir
        if token in _SPEECH_DATASETS + _MUSIC_DATASETS + _GENERAL_DATASETS:
            return token
    if len(path.parts) >= 2:
        return path.parent.name.lower()
    return "unknown"


def _infer_waveform_family(path: Path) -> str:
    stem = path.stem.lower()
    if "self_phase" in stem:
        return "self_phase"
    if "self_amp" in stem:
        return "self_amplitude"
    if "snr" in stem:
        return "snr"
    if "timbre" in stem:
        return "timbre"
    if "sine_freq" in stem or "frequency" in stem:
        return "frequency"
    if "amplitude" in stem:
        return "amplitude"
    if any(token in stem for token in ("sine", "square", "triangle", "saw", "noise", "chirp")):
        return "synthetic_waveform"
    dataset = _infer_dataset(path)
    if dataset in _SPEECH_DATASETS:
        return "speech"
    if dataset in _MUSIC_DATASETS:
        return "music"
    if dataset in _GENERAL_DATASETS:
        return "general_audio"
    return path.parent.name.lower() or "unknown"


def _infer_perturbation(path: Path) -> str:
    stem = path.stem.lower()
    if "baseline" in stem:
        return "baseline"
    for key in ("self_phase", "self_amp", "snr", "timbre", "frequency", "amplitude", "resilience", "room", "offset"):
        if key in stem:
            return key
    return "baseline"


def _infer_class_name(path: Path) -> str:
    tokens = _normalize_tokens(path)
    token_set = set(tokens)
    if "male" in token_set and "speech" in token_set:
        return "male_speech"
    if "female" in token_set and "speech" in token_set:
        return "female_speech"
    if any(hint in token_set for hint in _FOOTSTEP_HINTS):
        return "footsteps"
    if any(hint in token_set for hint in _MACHINE_HINTS):
        return "machine_vehicle"
    if any(hint in token_set for hint in _SYNTH_HINTS):
        return "synthetic_effects"
    if any(hint in token_set for hint in _ENV_HINTS):
        return "environmental"
    dataset = _infer_dataset(path)
    if dataset in _SPEECH_DATASETS:
        return "speech"
    if dataset in _MUSIC_DATASETS:
        return "music"
    if dataset in _GENERAL_DATASETS:
        return "general_sound"
    return "unknown"


def _load_transcripts(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    transcripts: dict[str, str] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row.get("file_id") or row.get("file_path") or "").strip()
            text = (row.get("reference_text") or row.get("transcript") or "").strip()
            if key and text:
                transcripts[key] = text
    return transcripts


def _iter_wavs(inputs: list[Path]) -> list[Path]:
    wavs: list[Path] = []
    for input_path in inputs:
        resolved = input_path.resolve()
        if resolved.is_file() and resolved.suffix.lower() == ".wav":
            wavs.append(resolved)
        elif resolved.is_dir():
            wavs.extend(sorted(resolved.rglob("*.wav")))
    return wavs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate benchmark_manifest.csv using dataset-native naming where possible and deterministic heuristics otherwise."
    )
    parser.add_argument("--input", action="append", type=Path, required=True,
                        help="WAV file or directory to include. Can be passed multiple times.")
    parser.add_argument("--output-csv", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--transcripts-csv", type=Path, default=None,
                        help="Optional CSV containing file_id/file_path and reference_text/transcript columns.")
    parser.add_argument("--dataset-override", default=None,
                        help="Optional dataset value to apply to every row.")
    parser.add_argument("--waveform-family-override", default=None,
                        help="Optional waveform_family value to apply to every row.")
    parser.add_argument("--class-override", default=None,
                        help="Optional class_name value to apply to every row.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wavs = _iter_wavs(args.input)
    if not wavs:
        raise ValueError("No WAV inputs found for manifest generation")

    transcripts = _load_transcripts(args.transcripts_csv)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "file_path",
                "dataset",
                "waveform_family",
                "perturbation",
                "class_name",
                "reference_text",
                "file_id",
            ],
        )
        writer.writeheader()
        for wav in wavs:
            file_id = wav.stem
            reference_text = transcripts.get(file_id, transcripts.get(str(wav), ""))
            writer.writerow(
                {
                    "file_path": str(wav),
                    "dataset": args.dataset_override or _infer_dataset(wav),
                    "waveform_family": args.waveform_family_override or _infer_waveform_family(wav),
                    "perturbation": _infer_perturbation(wav),
                    "class_name": args.class_override or _infer_class_name(wav),
                    "reference_text": reference_text,
                    "file_id": file_id,
                }
            )

    print(f"Wrote manifest with {len(wavs)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
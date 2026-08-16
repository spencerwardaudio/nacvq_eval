from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - dependency is declared, but keep a graceful fallback
    tqdm = None


_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
_DEFAULT_OUTPUT_DIR = _PROJ_ROOT / "datasets" / "analysis" / "benchmark_runs"
_DEFAULT_CSV = _PROJ_ROOT / "datasets" / "analysis" / "benchmark_metrics.csv"
_ENCODEC_MAIN = _PROJ_ROOT / "Encodec" / "main.py"
_ENCODEC_DIR = _PROJ_ROOT / "Encodec"

if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))


def _safe_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _duration_seconds(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def _realized_bitrate_kbps(compressed_path: Path, duration_seconds: float) -> float:
    if duration_seconds <= 0:
        return float("nan")
    return compressed_path.stat().st_size * 8.0 / 1000.0 / duration_seconds


def _normalize_text(text: str) -> list[str]:
    return " ".join(text.strip().lower().split()).split()


def _word_error_rate(reference: str, hypothesis: str) -> float:
    ref = _normalize_text(reference)
    hyp = _normalize_text(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1] / max(1, len(ref))


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def _run_shell_capture(command: str) -> str:
    result = subprocess.run(command, shell=True, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def _compute_signal_metrics(reference_path: Path, decoded_path: Path) -> dict[str, float]:
    from Encodec.cal_metrics import calculate_si_snr, calculate_stoi

    ref_audio, ref_sr = sf.read(str(reference_path), always_2d=False)
    deg_audio, deg_sr = sf.read(str(decoded_path), always_2d=False)

    ref_audio = np.asarray(ref_audio, dtype=np.float32)
    deg_audio = np.asarray(deg_audio, dtype=np.float32)

    if ref_audio.ndim > 1:
        ref_audio = ref_audio.mean(axis=1)
    if deg_audio.ndim > 1:
        deg_audio = deg_audio.mean(axis=1)

    target_sr = ref_sr
    if deg_sr != target_sr:
        deg_audio = librosa.resample(deg_audio, orig_sr=deg_sr, target_sr=target_sr)

    min_len = min(len(ref_audio), len(deg_audio))
    ref_audio = ref_audio[:min_len]
    deg_audio = deg_audio[:min_len]

    metrics = {
        "si_snr": float(calculate_si_snr(ref_audio, deg_audio)),
    }

    stoi_sr = 16000
    ref_stoi = librosa.resample(ref_audio, orig_sr=target_sr, target_sr=stoi_sr)
    deg_stoi = librosa.resample(deg_audio, orig_sr=target_sr, target_sr=stoi_sr)
    min_len = min(len(ref_stoi), len(deg_stoi))
    metrics["stoi"] = float(calculate_stoi(ref_stoi[:min_len], deg_stoi[:min_len], stoi_sr))
    return metrics


@dataclass
class ManifestRow:
    file_path: Path
    dataset: str
    waveform_family: str
    perturbation: str
    class_name: str
    reference_text: str
    file_id: str


def _manifest_from_csv(path: Path) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_path = Path(row["file_path"]).expanduser()
            if not file_path.is_absolute():
                file_path = (_PROJ_ROOT / file_path).resolve()
            rows.append(
                ManifestRow(
                    file_path=file_path,
                    dataset=(row.get("dataset") or file_path.parent.name or "unknown").strip(),
                    waveform_family=(row.get("waveform_family") or file_path.parent.name or "unknown").strip(),
                    perturbation=(row.get("perturbation") or "baseline").strip(),
                    class_name=(row.get("class_name") or "unknown").strip(),
                    reference_text=(row.get("reference_text") or "").strip(),
                    file_id=(row.get("file_id") or file_path.stem).strip(),
                )
            )
    return rows


def _manifest_from_inputs(inputs: Iterable[Path]) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for input_root in inputs:
        input_root = input_root.resolve()
        if input_root.is_file() and input_root.suffix.lower() == ".wav":
            wav_paths = [input_root]
            base_name = input_root.parent.name
        else:
            wav_paths = sorted(input_root.rglob("*.wav"))
            base_name = input_root.name
        for wav_path in wav_paths:
            rows.append(
                ManifestRow(
                    file_path=wav_path,
                    dataset=base_name,
                    waveform_family=base_name,
                    perturbation="baseline",
                    class_name="unknown",
                    reference_text="",
                    file_id=wav_path.stem,
                )
            )
    return rows


class CodecAdapter:
    name: str

    def available(self) -> tuple[bool, str]:
        raise NotImplementedError

    def supported_bitrates(self, requested: list[float]) -> list[float]:
        return requested

    def reconstruct(self, source_path: Path, output_dir: Path, target_bitrate: float) -> tuple[Path, Path, float]:
        raise NotImplementedError


class EncodecAdapter(CodecAdapter):
    name = "encodec"

    def __init__(self, checkpoint: str | None, model_name: str, device: str) -> None:
        self.checkpoint = checkpoint
        self.model_name = model_name
        self.device = device

    def available(self) -> tuple[bool, str]:
        if not _ENCODEC_MAIN.exists():
            return False, f"Missing {_ENCODEC_MAIN}"
        if self.model_name == "multi_dataset_encodec" and not self.checkpoint:
            return False, "--encodec-checkpoint is required for multi_dataset_encodec"
        return True, "ok"

    def reconstruct(self, source_path: Path, output_dir: Path, target_bitrate: float) -> tuple[Path, Path, float]:
        output_dir.mkdir(parents=True, exist_ok=True)
        compressed_path = output_dir / f"{source_path.stem}_bw{target_bitrate}.ecdc"
        decoded_path = output_dir / f"{source_path.stem}_bw{target_bitrate}_decoded.wav"
        if compressed_path.exists() and decoded_path.exists():
            return compressed_path, decoded_path, _realized_bitrate_kbps(compressed_path, _duration_seconds(source_path))
        env = dict(**os_environ_with_pythonpath())

        encode_cmd = [
            sys.executable,
            str(_ENCODEC_MAIN),
            str(source_path),
            str(compressed_path),
            "--model_name",
            self.model_name,
            "--bandwidth",
            str(target_bitrate),
            "--device",
            self.device,
            "--force",
        ]
        if self.checkpoint:
            encode_cmd.extend(["--checkpoint", self.checkpoint])
        _run(encode_cmd, env=env)

        decode_cmd = [
            sys.executable,
            str(_ENCODEC_MAIN),
            str(compressed_path),
            str(decoded_path),
            "--model_name",
            self.model_name,
            "--device",
            self.device,
            "--force",
        ]
        if self.checkpoint:
            decode_cmd.extend(["--checkpoint", self.checkpoint])
        _run(decode_cmd, env=env)
        realized = _realized_bitrate_kbps(compressed_path, _duration_seconds(source_path))
        return compressed_path, decoded_path, realized


class DacAdapter(CodecAdapter):
    name = "dac"

    def available(self) -> tuple[bool, str]:
        if importlib.util.find_spec("dac") is None:
            return False, "Python package 'dac' is not installed"
        return True, "ok"

    def reconstruct(self, source_path: Path, output_dir: Path, target_bitrate: float) -> tuple[Path, Path, float]:
        output_dir.mkdir(parents=True, exist_ok=True)
        codes_dir = output_dir / "codes"
        decoded_dir = output_dir / "decoded"
        compressed_path = codes_dir / f"{source_path.stem}.dac"
        decoded_path = decoded_dir / f"{source_path.stem}.wav"
        if compressed_path.exists() and decoded_path.exists():
            return compressed_path, decoded_path, _realized_bitrate_kbps(compressed_path, _duration_seconds(source_path))
        codes_dir.mkdir(parents=True, exist_ok=True)
        decoded_dir.mkdir(parents=True, exist_ok=True)
        _run([sys.executable, "-m", "dac", "encode", str(source_path), "--output", str(codes_dir)])
        _run([sys.executable, "-m", "dac", "decode", str(codes_dir), "--output", str(decoded_dir)])
        realized = _realized_bitrate_kbps(compressed_path, _duration_seconds(source_path))
        return compressed_path, decoded_path, realized


class OpusAdapter(CodecAdapter):
    name = "opus"

    def available(self) -> tuple[bool, str]:
        if shutil.which("opusenc") is None or shutil.which("opusdec") is None:
            return False, "opusenc/opusdec not found on PATH"
        return True, "ok"

    def reconstruct(self, source_path: Path, output_dir: Path, target_bitrate: float) -> tuple[Path, Path, float]:
        output_dir.mkdir(parents=True, exist_ok=True)
        compressed_path = output_dir / f"{source_path.stem}_bw{target_bitrate}.opus"
        decoded_path = output_dir / f"{source_path.stem}_bw{target_bitrate}_decoded.wav"
        if compressed_path.exists() and decoded_path.exists():
            return compressed_path, decoded_path, _realized_bitrate_kbps(compressed_path, _duration_seconds(source_path))
        _run([
            "opusenc",
            "--quiet",
            "--bitrate",
            str(target_bitrate),
            "--hard-cbr",
            str(source_path),
            str(compressed_path),
        ])
        _run(["opusdec", "--quiet", str(compressed_path), str(decoded_path)])
        realized = _realized_bitrate_kbps(compressed_path, _duration_seconds(source_path))
        return compressed_path, decoded_path, realized


class SemantiCodecAdapter(CodecAdapter):
    name = "semanticodec"

    _MODES = [
        (0.35, 25, 32768),
        (0.71, 50, 32768),
        (1.43, 100, 32768),
    ]

    def available(self) -> tuple[bool, str]:
        if importlib.util.find_spec("semanticodec") is None:
            return False, "Python package 'semanticodec' is not installed"
        return True, "ok"

    def supported_bitrates(self, requested: list[float]) -> list[float]:
        return [mode[0] for mode in self._MODES]

    def reconstruct(self, source_path: Path, output_dir: Path, target_bitrate: float) -> tuple[Path, Path, float]:
        output_dir.mkdir(parents=True, exist_ok=True)
        selected = min(self._MODES, key=lambda item: abs(item[0] - target_bitrate))
        realized, token_rate, vocab_size = selected
        compressed_path = output_dir / f"{source_path.stem}_bw{realized:.2f}.tokens.txt"
        decoded_path = output_dir / f"{source_path.stem}_bw{realized:.2f}_decoded.wav"
        if compressed_path.exists() and decoded_path.exists():
            return compressed_path, decoded_path, realized

        semanticodec_module = importlib.import_module("semanticodec")
        SemantiCodec = getattr(semanticodec_module, "SemantiCodec")

        model = SemantiCodec(token_rate=token_rate, semantic_vocab_size=vocab_size)
        tokens = model.encode(str(source_path))
        waveform = model.decode(tokens)
        import torch as _torch
        tokens_np = tokens.cpu().numpy() if isinstance(tokens, _torch.Tensor) else np.asarray(tokens)
        waveform_np = waveform.cpu().numpy() if isinstance(waveform, _torch.Tensor) else np.asarray(waveform)
        np.savetxt(compressed_path, tokens_np.reshape(-1), fmt="%d")
        audio = waveform_np[0, 0].astype(np.float32)
        sf.write(str(decoded_path), audio, 16000)
        return compressed_path, decoded_path, realized


class WavTokenizerAdapter(CodecAdapter):
    name = "wavtokenizer"

    def __init__(
        self,
        repo_path: str | None,
        config_path: str | None,
        checkpoint_path: str | None,
        device: str,
        bandwidth_id: int,
        sample_rate: int,
        vocab_size: int,
    ) -> None:
        self.repo_path = Path(repo_path).expanduser().resolve() if repo_path else None
        self.config_path = Path(config_path).expanduser().resolve() if config_path else None
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else None
        self.device = device
        self.bandwidth_id = bandwidth_id
        self.sample_rate = sample_rate
        self.vocab_size = vocab_size

    def available(self) -> tuple[bool, str]:
        if not self.repo_path:
            return False, "--wavtokenizer-repo is required"
        if not self.repo_path.exists():
            return False, f"WavTokenizer repo path not found: {self.repo_path}"
        if not self.config_path or not self.config_path.exists():
            return False, "--wavtokenizer-config is required and must exist"
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            return False, "--wavtokenizer-checkpoint is required and must exist"
        return True, "ok"

    def supported_bitrates(self, requested: list[float]) -> list[float]:
        bits_per_token = math.ceil(math.log2(self.vocab_size))
        estimated = 40.0 * bits_per_token / 1000.0
        return [estimated]

    def reconstruct(self, source_path: Path, output_dir: Path, target_bitrate: float) -> tuple[Path, Path, float]:
        output_dir.mkdir(parents=True, exist_ok=True)
        compressed_path = output_dir / f"{source_path.stem}_wavtokenizer_codes.npy"
        decoded_path = output_dir / f"{source_path.stem}_wavtokenizer_decoded.wav"
        if compressed_path.exists() and decoded_path.exists():
            codes_np = np.load(str(compressed_path))
            duration_seconds = _duration_seconds(source_path)
            bits_per_token = math.ceil(math.log2(self.vocab_size))
            realized = int(codes_np.size) * bits_per_token / max(duration_seconds, 1e-9) / 1000.0
            return compressed_path, decoded_path, realized

        sys.path.insert(0, str(self.repo_path))
        try:
            torch = importlib.import_module("torch")
            torchaudio = importlib.import_module("torchaudio")
            encoder_utils = importlib.import_module("encoder.utils")
            decoder_pretrained = importlib.import_module("decoder.pretrained")
            convert_audio = getattr(encoder_utils, "convert_audio")
            WavTokenizer = getattr(decoder_pretrained, "WavTokenizer")
        finally:
            if sys.path and sys.path[0] == str(self.repo_path):
                sys.path.pop(0)

        device = torch.device(self.device)
        model = WavTokenizer.from_pretrained0802(str(self.config_path), str(self.checkpoint_path))
        model = model.to(device)

        wav, sr = torchaudio.load(str(source_path))
        wav = convert_audio(wav, sr, self.sample_rate, 1)
        wav = wav.to(device)
        bandwidth_id = torch.tensor([self.bandwidth_id], device=device)
        features, discrete_code = model.encode_infer(wav, bandwidth_id=bandwidth_id)
        audio_out = model.decode(features, bandwidth_id=bandwidth_id)
        torchaudio.save(str(decoded_path), audio_out.cpu(), sample_rate=self.sample_rate, encoding='PCM_S', bits_per_sample=16)

        codes_np = discrete_code.detach().cpu().numpy()
        np.save(compressed_path, codes_np)
        duration_seconds = _duration_seconds(source_path)
        token_count = int(codes_np.size)
        bits_per_token = math.ceil(math.log2(self.vocab_size))
        realized = token_count * bits_per_token / max(duration_seconds, 1e-9) / 1000.0
        return compressed_path, decoded_path, realized


def os_environ_with_pythonpath() -> dict[str, str]:
    env = dict(**os.environ)
    current = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_ENCODEC_DIR}{os.pathsep}{current}" if current else str(_ENCODEC_DIR)
    return env


def _write_metric_row(writer: csv.DictWriter, base: dict[str, str | float], metric: str, value: float) -> None:
    row = dict(base)
    row["metric"] = metric
    row["value"] = value
    writer.writerow(row)


def _format_elapsed(seconds: float) -> str:
    minutes, remaining = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{remaining:02d}s"
    return f"{remaining}s"


def _maybe_write_semantic_rows(
    writer: csv.DictWriter,
    base: dict[str, str | float],
    decoded_path: Path,
    manifest_row: ManifestRow,
    asr_command: str | None,
    classifier_command: str | None,
) -> None:
    if asr_command and manifest_row.reference_text:
        command = asr_command.format(input=str(decoded_path), reference=manifest_row.reference_text)
        hypothesis = _run_shell_capture(command)
        _write_metric_row(writer, base, "wer", _word_error_rate(manifest_row.reference_text, hypothesis))
    if classifier_command and manifest_row.class_name and manifest_row.class_name != "unknown":
        command = classifier_command.format(input=str(decoded_path), label=manifest_row.class_name)
        prediction = _run_shell_capture(command).strip()
        accuracy = 1.0 if prediction == manifest_row.class_name else 0.0
        _write_metric_row(writer, base, "classification_accuracy", accuracy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a normalized multi-codec benchmark and write benchmark_metrics.csv")
    parser.add_argument("--manifest-csv", type=Path, default=None,
                        help="Optional CSV with file_path,dataset,waveform_family,perturbation,class_name,reference_text,file_id")
    parser.add_argument("--input", action="append", type=Path, default=[],
                        help="Input WAV file or directory when no manifest CSV is provided. Can be passed multiple times.")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-csv", type=Path, default=_DEFAULT_CSV)
    parser.add_argument("--codecs", nargs="+", default=["encodec", "dac", "semanticodec", "opus"])
    parser.add_argument("--bitrates", nargs="+", type=float, default=[24.0, 12.0, 6.0, 3.0, 1.5])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encodec-model-name", default="multi_dataset_encodec")
    parser.add_argument("--encodec-checkpoint", default=None)
    parser.add_argument("--skip-missing-codecs", action="store_true",
                        help="Skip codecs whose official tooling is not installed instead of failing.")
    parser.add_argument("--asr-command", default=None,
                        help="Optional shell command template for WER. Use {input} and {reference} placeholders.")
    parser.add_argument("--classifier-command", default=None,
                        help="Optional shell command template for classification accuracy. Use {input} and {label} placeholders.")
    parser.add_argument("--wavtokenizer-repo", default=None,
                        help="Path to a cloned official WavTokenizer repository/model checkout.")
    parser.add_argument("--wavtokenizer-config", default=None,
                        help="Path to the WavTokenizer config YAML used by from_pretrained0802.")
    parser.add_argument("--wavtokenizer-checkpoint", default=None,
                        help="Path to the WavTokenizer checkpoint used by from_pretrained0802.")
    parser.add_argument("--wavtokenizer-bandwidth-id", type=int, default=0,
                        help="Official WavTokenizer bandwidth_id for encode_infer/decode.")
    parser.add_argument("--wavtokenizer-sample-rate", type=int, default=24000,
                        help="Input/output sample rate used by the selected WavTokenizer model.")
    parser.add_argument("--wavtokenizer-vocab-size", type=int, default=4096,
                        help="Vocabulary size used to estimate WavTokenizer realized bitrate from token count.")
    return parser.parse_args()


def build_adapters(args: argparse.Namespace) -> dict[str, CodecAdapter]:
    adapters: dict[str, CodecAdapter] = {
        "encodec": EncodecAdapter(args.encodec_checkpoint, args.encodec_model_name, args.device),
        "dac": DacAdapter(),
        "semanticodec": SemantiCodecAdapter(),
        "opus": OpusAdapter(),
        "wavtokenizer": WavTokenizerAdapter(
            args.wavtokenizer_repo,
            args.wavtokenizer_config,
            args.wavtokenizer_checkpoint,
            args.device,
            args.wavtokenizer_bandwidth_id,
            args.wavtokenizer_sample_rate,
            args.wavtokenizer_vocab_size,
        ),
    }
    return {name: adapters[name] for name in args.codecs}


def load_manifest(args: argparse.Namespace) -> list[ManifestRow]:
    if args.manifest_csv:
        return _manifest_from_csv(args.manifest_csv)
    if args.input:
        return _manifest_from_inputs(args.input)
    raise ValueError("Provide either --manifest-csv or at least one --input path")


def main() -> None:
    args = parse_args()
    manifest_rows = load_manifest(args)
    if not manifest_rows:
        raise ValueError("No WAV inputs discovered for the benchmark")

    adapters = build_adapters(args)
    selected_adapters: dict[str, CodecAdapter] = {}
    for name, adapter in adapters.items():
        ok, detail = adapter.available()
        if ok:
            selected_adapters[name] = adapter
            print(f"[codec] {name}: available")
            continue
        if args.skip_missing_codecs:
            print(f"[codec] {name}: skipped ({detail})")
            continue
        raise RuntimeError(f"Codec {name} unavailable: {detail}")

    if not selected_adapters:
        raise RuntimeError("No codecs are available to run. Install at least one codec or remove --skip-missing-codecs.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "codec",
        "dataset",
        "waveform_family",
        "perturbation",
        "class_name",
        "file_id",
        "source_path",
        "decoded_path",
        "compressed_path",
        "bitrate_target",
        "bitrate_realized",
        "metric",
        "value",
    ]

    rows_written = 0
    rows_by_codec = {name: 0 for name in selected_adapters}
    total_jobs = len(manifest_rows) * sum(len(adapter.supported_bitrates(args.bitrates)) for adapter in selected_adapters.values())
    progress_context = None
    progress = None
    last_printed_pct = -1

    if tqdm is not None:
        progress_context = tqdm(total=total_jobs, unit="job", dynamic_ncols=True, desc="benchmark")
        progress = progress_context
    else:
        print(f"Progress: 0/{total_jobs} jobs")
    started_at = time.monotonic()

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        completed_jobs = 0
        for manifest_row in manifest_rows:
            for codec_name, adapter in selected_adapters.items():
                supported_bitrates = adapter.supported_bitrates(args.bitrates)
                for target_bitrate in supported_bitrates:
                    codec_dir = args.output_dir / codec_name / f"bw_{target_bitrate:g}" / manifest_row.file_id
                    compressed_path, decoded_path, realized_bitrate = adapter.reconstruct(
                        manifest_row.file_path,
                        codec_dir,
                        target_bitrate,
                    )

                    if not compressed_path.exists():
                        raise RuntimeError(
                            f"Codec {codec_name} did not produce compressed output for {manifest_row.file_id} at {target_bitrate:g} kbps: {compressed_path}"
                        )
                    if not decoded_path.exists():
                        raise RuntimeError(
                            f"Codec {codec_name} did not produce decoded output for {manifest_row.file_id} at {target_bitrate:g} kbps: {decoded_path}"
                        )

                    metrics = _compute_signal_metrics(manifest_row.file_path, decoded_path)
                    if not metrics:
                        raise RuntimeError(
                            f"Codec {codec_name} produced no metrics for {manifest_row.file_id} at {target_bitrate:g} kbps"
                        )
                    base = {
                        "codec": codec_name,
                        "dataset": manifest_row.dataset,
                        "waveform_family": manifest_row.waveform_family,
                        "perturbation": manifest_row.perturbation,
                        "class_name": manifest_row.class_name,
                        "file_id": manifest_row.file_id,
                        "source_path": str(manifest_row.file_path),
                        "decoded_path": str(decoded_path),
                        "compressed_path": str(compressed_path),
                        "bitrate_target": target_bitrate,
                        "bitrate_realized": realized_bitrate,
                    }
                    for metric_name, value in metrics.items():
                        _write_metric_row(writer, base, metric_name, value)
                        rows_written += 1
                        rows_by_codec[codec_name] += 1
                    _maybe_write_semantic_rows(
                        writer,
                        base,
                        decoded_path,
                        manifest_row,
                        args.asr_command,
                        args.classifier_command,
                    )
                    handle.flush()
                    completed_jobs += 1
                    elapsed = time.monotonic() - started_at
                    rate = completed_jobs / elapsed if elapsed > 0 else 0.0
                    remaining = max(total_jobs - completed_jobs, 0)
                    eta = remaining / rate if rate > 0 else float("inf")
                    
                    # Calculate current percentage
                    current_pct = int(100 * completed_jobs / total_jobs) if total_jobs > 0 else 0
                    
                    # Only print detailed output when percentage changes
                    if current_pct != last_printed_pct:
                        msg = (
                            f"[{current_pct:3d}%] codec={codec_name} file={manifest_row.file_id} "
                            f"target={target_bitrate:g} realized={realized_bitrate:.3f}"
                        )
                        if progress is not None:
                            # Use tqdm.write() to avoid interfering with progress bar
                            progress.write(msg)
                        else:
                            print(msg)
                        last_printed_pct = current_pct
                    
                    if progress is not None:
                        progress.update(1)
                        progress.set_postfix(
                            codec=codec_name,
                            file=manifest_row.file_id,
                            eta=_format_elapsed(eta) if math.isfinite(eta) else "?",
                        )
                    else:
                        if current_pct != last_printed_pct:
                            print(
                                f"Progress: {completed_jobs}/{total_jobs} jobs | "
                                f"elapsed={_format_elapsed(elapsed)} | "
                                f"eta={_format_elapsed(eta) if math.isfinite(eta) else '?'}"
                            )

    if rows_written == 0:
        raise RuntimeError(f"Benchmark completed but wrote no metric rows to {args.output_csv}")

    empty_codecs = [codec for codec, count in rows_by_codec.items() if count == 0]
    if empty_codecs:
        raise RuntimeError(
            "The following codecs produced no metric rows: " + ", ".join(empty_codecs)
        )

    if not args.output_csv.exists() or args.output_csv.stat().st_size <= len(",".join(fieldnames)):
        raise RuntimeError(f"Benchmark CSV is missing or empty: {args.output_csv}")

    if progress_context is not None:
        progress_context.close()

    print(f"Wrote normalized benchmark metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
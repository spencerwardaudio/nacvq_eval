import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
_ENCODEC_DIR = _PROJ_ROOT / "Encodec"

# Add Encodec directory to path so we can import its modules directly
if str(_ENCODEC_DIR) not in sys.path:
    sys.path.insert(0, str(_ENCODEC_DIR))


def _load_model(model_name: str, checkpoint, bandwidth: float, device: str):
    """Load and configure the EnCodec model once."""
    from compress import MODELS
    if model_name == "multi_dataset_encodec":
        if not checkpoint:
            raise ValueError("--checkpoint is required when using multi_dataset_encodec model")
        model = MODELS[model_name](str(checkpoint))
    elif model_name in ("my_encodec", "encodec_bw"):
        model = MODELS[model_name](str(checkpoint), [bandwidth])
    else:
        model = MODELS[model_name]()
    model = model.to(device)
    if bandwidth not in model.target_bandwidths:
        raise ValueError(f"Bandwidth {bandwidth} not supported by {model_name}")
    model.set_target_bandwidth(bandwidth)
    return model


def _encode_files(wav_files, output_path, model, bandwidth, device):
    """Encode a list of wav files using a pre-loaded model. Returns (success, failed) counts."""
    import torch
    import soundfile as sf
    from compress import compress
    from utils import convert_audio

    success_count = 0
    failed_count = 0
    for wav_file in wav_files:
        output_file = output_path / f"{wav_file.stem}_bw{bandwidth}.ecdc"
        print(f"Encoding: {wav_file.name} → {output_file.name}")
        try:
            wav, sr = sf.read(str(wav_file))
            wav = torch.from_numpy(wav).float()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)
            else:
                wav = wav.transpose(0, 1)  # soundfile gives [T, C]; Encodec expects [C, T]
            wav = convert_audio(wav, sr, model.sample_rate, model.channels)
            wav = wav.to(device)
            with torch.no_grad():
                compressed = compress(model, wav, use_lm=False)  # skip language model for speed; LM not needed for token extraction
            output_file.write_bytes(compressed)
            print(f"  ✓ Success")
            success_count += 1
        except Exception as exc:
            print(f"  ✗ FAILED\n  Error: {exc}")
            failed_count += 1
    return success_count, failed_count


def batch_encode_audio_folder(input_folder, output_dir=None, model_name="encodec_48khz",
                               bandwidth=24.0, device="cuda", checkpoint=None,
                               _model=None):
    """
    Encode all .wav files in a folder using the specified model and bandwidth.
    Model is loaded once and reused for all files — much faster than one subprocess per file.
    Pass _model to reuse an already-loaded model across multiple calls.
    """
    input_path = Path(input_folder)
    if not input_path.exists():
        print(f"Input folder {input_folder} does not exist")
        return

    output_path = Path(output_dir) if output_dir else input_path
    output_path.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(input_path.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files found in {input_folder}")
        return

    print(f"Found {len(wav_files)} .wav files to encode")
    if _model is None:
        print(f"Model: {model_name}, Bandwidth: {bandwidth} kbps, Device: {device}\n")
        model = _load_model(model_name, checkpoint, bandwidth, device)  # load once and reuse for all files
    else:
        model = _model

    success_count, failed_count = _encode_files(wav_files, output_path, model, bandwidth, device)

    print(f"\n{'='*60}")
    print(f"Encoding complete: {success_count} succeeded, {failed_count} failed")
    print(f"Output files saved in: {output_path}")


if __name__ == "__main__":
    DEFAULT_INPUT_BASE = _PROJ_ROOT / "datasets" / "time_offset_recordings"
    DEFAULT_OUTPUT_BASE = _PROJ_ROOT / "datasets" / "ecdc"

    if len(sys.argv) < 2:
        # Batch mode: encode all subdirectories in time_offset_recordings/
        subdirs = sorted(p for p in DEFAULT_INPUT_BASE.iterdir() if p.is_dir())
        if not subdirs:
            print(f"No subdirectories found in {DEFAULT_INPUT_BASE}")
            sys.exit(1)
        print(f"Batch mode: encoding {len(subdirs)} subdirectorie(s)")
        for subdir in subdirs:
            out_dir = DEFAULT_OUTPUT_BASE / subdir.name
            print(f"\n--- {subdir.name} → {out_dir} ---")
            batch_encode_audio_folder(subdir, output_dir=out_dir, model_name="encodec_48khz")
        sys.exit(0)

    input_folder = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].replace(".", "").isdigit() else None
    bandwidth_idx = 2 if output_dir else 2
    bandwidth = float(sys.argv[bandwidth_idx + (1 if output_dir else 0)]) if len(sys.argv) > bandwidth_idx + (1 if output_dir else 0) else 24.0
    device_idx = bandwidth_idx + (1 if output_dir else 0) + 1
    device = sys.argv[device_idx] if len(sys.argv) > device_idx else "cuda"

    model_name = sys.argv[device_idx + 1] if len(sys.argv) > device_idx + 1 else "encodec_48khz"
    checkpoint = sys.argv[device_idx + 2] if len(sys.argv) > device_idx + 2 else None
    batch_encode_audio_folder(input_folder, output_dir=output_dir, model_name=model_name,
                              bandwidth=bandwidth, device=device, checkpoint=checkpoint)
"""Generate time-offset audio recordings for latent sensitivity analysis."""

from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import signal as sig

def create_time_delayed_audio(
    source_audio: np.ndarray,
    delay_ms: float,
    sample_rate: int = 48000,
) -> np.ndarray:
    """Apply fractional-sample delay to audio."""
    delay_samples = (delay_ms / 1000.0) * sample_rate
    
    # Windowed-sinc interpolation (same as your spatial offset method)
    n_taps = 201  # 201-tap FIR gives good stopband attenuation without excessive computation
    center = (n_taps - 1) / 2
    n = np.arange(n_taps)
    sinc_vals = np.sinc(n - center - delay_samples)  # fractional delay via shifted sinc kernel
    window = np.hanning(n_taps)
    fir = sinc_vals * window  # Hann window suppresses Gibbs ripple at band edges
    fir /= fir.sum()  # DC gain = 1.0 so overall amplitude is preserved
    
    delayed = sig.fftconvolve(source_audio, fir, mode='full')
    half = (len(fir) - 1) // 2
    return delayed[half : half + len(source_audio)]  # trim convolution artefacts to preserve original length


def generate_time_offset_recordings(
    source_file: Path,
    output_dir: Path,
    sample_rate: int = 48000,
):
    """Generate recordings with 1-20ms linear, then 20-100ms in 20ms increments."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load source audio (force mono)
    audio_data, sr = sf.read(source_file)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)  # mix to mono
    if sr != sample_rate:
        from scipy.signal import resample
        audio_data = resample(audio_data, int(len(audio_data) * sample_rate / sr))
    
    # Define offset schedule
    offsets_ms = list(range(0, 21))  # 0-20ms linear
    offsets_ms.extend(range(40, 101, 20))  # 40, 60, 80, 100ms
    
    recordings = {}
    for offset_ms in offsets_ms:
        if offset_ms == 0:
            delayed_audio = audio_data
            fname = "baseline_0ms.wav"
        else:
            delayed_audio = create_time_delayed_audio(audio_data, offset_ms, sample_rate)
            fname = f"offset_{offset_ms:03d}ms.wav"
        
        # Normalize
        max_val = np.abs(delayed_audio).max()
        delayed_audio = delayed_audio / (max_val + 1e-10)  # 1e-10 guards against perfectly silent audio
        
        output_path = output_dir / fname
        sf.write(str(output_path), delayed_audio, sample_rate)
        recordings[offset_ms] = output_path
        print(f"✓ {fname}")
    
    return recordings

def batch_generate_time_offset_recordings(
    input_dir: Path,
    output_base: Path,
    sample_rate: int = 48000,
) -> dict:
    """Process all .wav files in input_dir, outputting each to output_base/<stem>/."""
    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        print(f"No .wav files found in {input_dir}")
        return {}

    print(f"Batch mode: processing {len(wav_files)} file(s) from {input_dir}")
    all_recordings = {}
    for wav_file in wav_files:
        out_dir = output_base / wav_file.stem
        print(f"\n--- {wav_file.name} → {out_dir} ---")
        recordings = generate_time_offset_recordings(wav_file, out_dir, sample_rate)
        all_recordings[wav_file.stem] = recordings
    return all_recordings


if __name__ == "__main__":
    import sys

    HERE = Path(__file__).resolve().parent
    PROJ_ROOT = HERE.parent
    DEFAULT_INPUT_DIR = PROJ_ROOT / "datasets" / "audio_files"
    DEFAULT_OUTPUT_BASE = PROJ_ROOT / "datasets" / "time_offset_recordings"

    if len(sys.argv) >= 2:
        # Single-file mode
        source_file = Path(sys.argv[1])
        output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT_BASE / source_file.stem
        generate_time_offset_recordings(source_file, output_dir)
    else:
        # Batch mode: process all WAVs in audio_files/
        batch_generate_time_offset_recordings(DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_BASE)
        print(f"\nAll done. Outputs in {DEFAULT_OUTPUT_BASE}")
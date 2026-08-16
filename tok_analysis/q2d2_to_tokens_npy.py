"""Extract Q2D2 grid-pair token indices from a Q2D2 Lightning checkpoint.

Output shape: [n_pairs, T] — all 16 grid pairs from the single quantizer stage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

# Add both Q2D2 and Encodec to path
_HERE = Path(__file__).resolve().parent
_Q2D2_DIR = _HERE.parent / "Q2D2"
_ENCODEC_DIR = _HERE.parent / "Encodec"
for path in [_Q2D2_DIR, _ENCODEC_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

def load_q2d2_lightning(checkpoint_path: Path, device: str):
    """Load Q2D2 encodec directly from Lightning checkpoint state_dict."""
    print(f"Loading Q2D2 Lightning checkpoint")
    from decoder.feature_extractors import EncodecFeatures
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "state_dict" not in ckpt:
        raise ValueError(f"Not a valid Lightning checkpoint: {checkpoint_path}")
    
    # Build EncodecFeatures with Q2D2 config (from configs/Q2D2_fsd50k_9.8kbps_dim512_attn_b16.yaml)
    feature_extractor = EncodecFeatures(
        encodec_model="encodec_24khz",
        bandwidths=[9.8],
        train_codebooks=True,
        num_quantizers=1,
        dowmsamples=[6, 4, 3, 1],
        vq_kmeans=200,
        vq_type='rhombic',
        codebook_dim=[9]*16 + [7]*16
    )
    
    # Extract and load encodec weights from Lightning state_dict
    encodec_state = {
        k.replace("feature_extractor.encodec.", ""): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("feature_extractor.encodec")  # Q2D2 stores EnCodec backbone weights under this prefix
    }
    feature_extractor.encodec.load_state_dict(encodec_state, strict=False)
    feature_extractor.encodec.eval()
    feature_extractor.encodec.to(device)
    
    print(f"  Q2D2 encodec loaded, sample_rate={feature_extractor.encodec.sample_rate}")
    
    # Wrap in simple namespace to match extract_tokens_q2d2 expectations
    class Q2D2Wrapper:
        def __init__(self, encodec):
            self.feature_extractor = type('obj', (object,), {'encodec': encodec})()
            self.hparams = type('obj', (object,), {'sample_rate': encodec.sample_rate})()
    
    return Q2D2Wrapper(feature_extractor.encodec), 'q2d2'


def extract_tokens_q2d2(
    wav_path: Path,
    model,  # Q2D2Wrapper (loaded via load_q2d2_lightning)
    device: str,
) -> np.ndarray:
    """Extract per-pair grid indices from Q2D2. Output shape: [n_pairs, T]."""
    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    sample_rate = model.hparams.sample_rate
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)

    wav = wav.to(device)  # [1, T]

    with torch.no_grad():
        z = model.feature_extractor.encodec.encoder(wav.unsqueeze(0))  # [1, D, T_enc]
        quantizer = model.feature_extractor.encodec.quantizer
        # encode_pairs returns [n_q=1, n_pairs, B=1, T]; stage 0 → [n_pairs, T]
        pair_codes = quantizer.encode_pairs(z)  # n_pairs=16 grid pairs for 9.8kbps config
        tokens = pair_codes[0, :, 0, :].cpu().numpy()

    return tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Input .wav file or directory")
    parser.add_argument("--output", type=Path, default=None, help="Output directory (default: same as input)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint file")
    parser.add_argument("--model", choices=["q2d2"], required=True,
                        help="Model type: q2d2 (Q2D2 Lightning VocosExp)")
    parser.add_argument("--bandwidth", default="9.8", help="Bandwidth string for output filename")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading Q2D2 checkpoint: {args.checkpoint}")
    model, _ = load_q2d2_lightning(args.checkpoint, args.device)

    # Process input
    if args.input.is_file():
        wav_files = [args.input]
    else:
        wav_files = sorted(args.input.glob("**/*.wav"))

    output_dir = args.output or args.input.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for wav_path in wav_files:
        print(f"Processing: {wav_path.name}")
        tokens = extract_tokens_q2d2(wav_path, model, args.device)

        # naming convention: {stem}_bw{bandwidth}_tokens.npy
        output_path = output_dir / f"{wav_path.stem}_bw{args.bandwidth}_tokens.npy"
        np.save(output_path, tokens)
        print(f"  → {output_path.name} (shape: {tokens.shape})")
    
    print(f"\nProcessed {len(wav_files)} files")


if __name__ == "__main__":
    main()

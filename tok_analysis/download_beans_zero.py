#!/usr/bin/env python3
"""Download and prepare BEANS-Zero dataset from HuggingFace for Q2D2 evaluation.

BEANS-Zero is a zero-shot bird classification dataset from Earth Species Project
featuring wildlife acoustics with ultra-high-frequency chirps and complex phase patterns.

This script downloads ONLY the test split for out-of-domain reconstruction quality evaluation.
BEANS wildlife acoustics are genuinely out-of-domain for Q2D2 (trained on FSD50k environmental sounds).

Dataset: https://huggingface.co/datasets/EarthSpeciesProject/BEANS-Zero

Usage:
    # Download test set for reconstruction evaluation
    python download_beans_zero.py --output-dir datasets/beans_zero
    
    # Download limited samples for testing
    python download_beans_zero.py --output-dir datasets/beans_zero --max-samples 50
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import itertools

import numpy as np
import soundfile as sf
from tqdm import tqdm


def download_beans_zero(output_dir: Path, max_samples: int = None) -> None:
    """Download BEANS-Zero test split from HuggingFace for reconstruction evaluation."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required to download from HuggingFace.\n"
            "Install it with: pip install datasets"
        )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("BEANS-Zero Dataset Download (TEST SPLIT ONLY - STREAMING)")
    print("="*60)
    print(f"Loading from HuggingFace Hub (streaming mode)...")
    print(f"Dataset: EarthSpeciesProject/BEANS-Zero")
    print(f"Output: {output_dir}")
    print(f"Purpose: Out-of-domain reconstruction quality evaluation")
    if max_samples:
        print(f"Limiting to {max_samples} samples")
    print("="*60)
    
    # Load ONLY test split in STREAMING mode - no Arrow cache!
    dataset = load_dataset(
        "EarthSpeciesProject/BEANS-Zero",
        split='test',
        streaming=True,  # streaming avoids caching ~100 GB to disk; just-in-time decoding
    )
    
    # Process test split
    print(f"\nStreaming test split (no disk cache)...")
    
    # Limit samples if specified
    if max_samples:
        test_data = itertools.islice(dataset, max_samples)  # islice streams lazily without buffering the whole dataset
        total_estimate = max_samples
    else:
        test_data = dataset
        total_estimate = 91965  # Known test split size
    
    file_list = []
    sr = 32000
    
    for idx, sample in enumerate(tqdm(test_data, desc="Downloading test samples", total=total_estimate)):
        # Extract audio (streaming mode can return list or dict depending on feature decoding)
        audio_data = sample['audio']

        # Prefer explicit sampling rate from metadata if present.
        raw_meta = sample.get('metadata')
        if isinstance(raw_meta, str):
            try:
                parsed_meta = json.loads(raw_meta)
                sr = int(parsed_meta.get('sample_rate', parsed_meta.get('sampling_rate', sr)))
            except Exception:
                pass
        elif isinstance(raw_meta, dict):
            sr = int(raw_meta.get('sample_rate', raw_meta.get('sampling_rate', sr)))

        if isinstance(audio_data, dict):
            array = np.array(audio_data.get('array', []), dtype=np.float32)
            sr = int(audio_data.get('sampling_rate', sr))
        else:
            array = np.array(audio_data, dtype=np.float32)
        
        # Get label (species name) for reference only
        label = sample.get('species', sample.get('label', sample.get('source_dataset', 'unknown')))
        
        # Save audio file
        safe_label = str(label).replace(' ', '_').replace('/', '-')
        filename = f"test_{idx:05d}_{safe_label}.wav"
        filepath = output_dir / filename
        sf.write(str(filepath), array, sr)
        
        file_list.append({
            'filename': filename,
            'label': label,
            'duration_sec': len(array) / sr
        })
    
    # Save minimal metadata for reconstruction evaluation
    metadata = {
        'dataset': 'BEANS-Zero',
        'source': 'EarthSpeciesProject/BEANS-Zero',
        'split': 'test',
        'purpose': 'out-of-domain reconstruction quality evaluation',
        'num_samples': len(file_list),
        'original_sample_rate': sr,
        'files': file_list
    }
    
    metadata_file = output_dir / 'beans_zero_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)
    print(f"  Test samples: {len(file_list)}")
    print(f"  Original sample rate: {sr} Hz")
    print(f"  Total duration: {sum(f['duration_sec'] for f in file_list):.1f} seconds")
    print(f"  Metadata: {metadata_file}")
    print("="*60)
    print("\nNext steps:")
    print("  Run reconstruction evaluation with:")
    print(f"    python tok_analysis/evaluate_q2d2_beans_zero.py \\")
    print(f"        --checkpoint <your_q2d2_checkpoint.ckpt> \\")
    print(f"        --data-root {output_dir} \\")
    print(f"        --device cuda")
    print("\n  This will compute reconstruction metrics (SNR, spectral distance)")
    print("  to evaluate Q2D2's out-of-domain performance on wildlife acoustics.")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Download BEANS-Zero test split for out-of-domain reconstruction evaluation'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('datasets/beans_zero'),
        help='Output directory for dataset (default: datasets/beans_zero)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Limit number of test samples (for testing, default: download all)'
    )
    
    args = parser.parse_args()
    
    try:
        download_beans_zero(args.output_dir, args.max_samples)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

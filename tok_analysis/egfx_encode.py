"""Encode all EGFx clean/processed pairs through multiple codecs.

Usage:
    python tok_analysis/egfx_encode.py \\
        --pairs datasets/egfx/effect_pairs.json \\
        --codecs encodec q2d2 \\
        --encodec-checkpoint checkpoints_multi_dataset/checkpoint.pt \\
        --q2d2-checkpoint Q2D2/checkpoints/q2d2_fsd50k.ckpt \\
        --output-dir datasets/audio_tokens/egfx
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from codec_interface import get_codec_encoder


def encode_pair(encoder, clean_path: str, processed_path: str, output_dir: Path):
    """Encode a clean/processed pair and save tokens + embeddings."""
    pair_name = Path(clean_path).stem
    
    # Encode clean
    clean_data = encoder.encode(clean_path)
    np.save(output_dir / f"{pair_name}_clean_tokens.npy", clean_data["tokens"])
    for i, emb in enumerate(clean_data["embeddings"]):
        np.save(output_dir / f"{pair_name}_clean_emb_layer{i}.npy", emb)
    
    # Encode processed
    processed_data = encoder.encode(processed_path)
    np.save(output_dir / f"{pair_name}_processed_tokens.npy", processed_data["tokens"])
    for i, emb in enumerate(processed_data["embeddings"]):
        np.save(output_dir / f"{pair_name}_processed_emb_layer{i}.npy", emb)


def sample_pairs_by_category(
    pairs_by_category: Dict[str, List[Dict]],
    max_per_category: int,
    seed: int,
) -> Dict[str, List[Dict]]:
    """Return a strictly balanced subset across all selected categories.

    If max_per_category > 0, each category gets exactly:
      min(max_per_category, min_category_count)
    samples.
    """
    rng = random.Random(seed)
    if not pairs_by_category:
        return {}

    counts = {k: len(v) for k, v in pairs_by_category.items()}
    min_count = min(counts.values())
    if min_count <= 0:
        empty = [k for k, v in counts.items() if v == 0]
        raise ValueError(
            "Cannot build balanced EGFx subset; category has zero pairs: "
            + ", ".join(empty)
        )

    target = min(max_per_category, min_count)
    sampled: Dict[str, List[Dict]] = {}
    for category, pairs in pairs_by_category.items():
        if len(pairs) == target:
            sampled[category] = list(pairs)
            continue
        sampled[category] = rng.sample(pairs, target)
    return sampled


def main():
    parser = argparse.ArgumentParser(description="Encode EGFx pairs through codecs")
    parser.add_argument("--pairs", type=Path, required=True, help="Effect pairs JSON file")
    parser.add_argument("--codecs", nargs="+", required=True,
                        choices=["encodec", "q2d2", "hificodec", "dac_fsq", "speechtokenizer"],
                        help="Codecs to use for encoding")
    parser.add_argument("--encodec-checkpoint", type=Path, help="Encodec checkpoint path")
    parser.add_argument("--q2d2-checkpoint", type=Path, help="Q2D2 checkpoint path")
    parser.add_argument("--hificodec-checkpoint", type=Path, help="HiFiCodec checkpoint")
    parser.add_argument("--dac-fsq-checkpoint", type=Path, help="DAC-FSQ checkpoint dir")
    parser.add_argument("--speechtokenizer-checkpoint", type=Path, help="SpeechTokenizer checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/audio_tokens/egfx"),
                        help="Output directory for tokens")
    parser.add_argument("--device", default="cuda", help="Device for inference")
    parser.add_argument("--max-per-category", type=int, default=0,
                        help="Balanced subsample size per category (0 = use all pairs)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed used when --max-per-category > 0")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Optional category filter (e.g. distortion modulation time_based)")
    parser.add_argument("--write-sampled-pairs", type=Path, default=None,
                        help="Optional path to save the sampled pairs JSON for reproducibility")
    args = parser.parse_args()
    
    # Load pairs
    with open(args.pairs) as f:
        pairs_by_category = json.load(f)

    if args.categories:
        requested = set(args.categories)
        pairs_by_category = {
            k: v for k, v in pairs_by_category.items() if k in requested
        }

    if args.max_per_category and args.max_per_category > 0:
        pairs_by_category = sample_pairs_by_category(
            pairs_by_category,
            max_per_category=args.max_per_category,
            seed=args.seed,
        )

    total_pairs = sum(len(v) for v in pairs_by_category.values())
    print("\nSelected EGFx categories and pair counts:")
    for category, pairs in pairs_by_category.items():
        print(f"  {category}: {len(pairs)}")
    print(f"  TOTAL: {total_pairs} pairs")

    if args.write_sampled_pairs:
        args.write_sampled_pairs.parent.mkdir(parents=True, exist_ok=True)
        with open(args.write_sampled_pairs, "w") as f:
            json.dump(pairs_by_category, f, indent=2)
        print(f"\n✓ Wrote sampled pairs to {args.write_sampled_pairs}")
    
    # Process each codec
    for codec_name in args.codecs:
        print(f"\n{'='*60}")
        print(f"Encoding with {codec_name.upper()}")
        print('='*60)
        
        # Get checkpoint
        checkpoint_arg = f"{codec_name}_checkpoint"
        checkpoint = getattr(args, checkpoint_arg, None)
        if not checkpoint:
            print(f"⚠ Skipping {codec_name}: no checkpoint specified (--{checkpoint_arg})")
            continue
        
        # Load encoder
        try:
            encoder = get_codec_encoder(codec_name, checkpoint, device=args.device)
        except Exception as e:
            print(f"⚠ Skipping {codec_name}: failed to initialize encoder: {e}")
            continue
        
        # Process each category
        for category, pairs in pairs_by_category.items():
            print(f"\n📁 Category: {category} ({len(pairs)} pairs)")
            output_dir = args.output_dir / codec_name / category
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for pair in tqdm(pairs, desc=f"Encoding {category}"):
                try:
                    encode_pair(encoder, pair["clean"], pair["processed"], output_dir)
                except Exception as e:
                    print(f"\n✗ Failed to encode {pair['effect']}: {e}")
        
        print(f"\n✓ Finished encoding with {codec_name}")
    
    print(f"\n✓ All encoding complete. Tokens saved to {args.output_dir}")
    print(f"\nNext step: python tok_analysis/egfx_metrics.py --tokens-dir {args.output_dir}")


if __name__ == "__main__":
    main()

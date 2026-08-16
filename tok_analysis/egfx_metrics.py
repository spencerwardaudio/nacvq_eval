"""Compute geometric metrics for EGFx clean/processed pairs.

Metrics:
- Token Flip Rate (TFR): Fraction of tokens that changed
- Cosine Similarity: Angular alignment between embeddings
- L2 Distance: Euclidean distance between embeddings
- Centroid Magnitudes: Mean magnitude per layer

Usage:
    python tok_analysis/egfx_metrics.py \\
        --tokens-dir datasets/audio_tokens/egfx \\
        --output datasets/analysis/egfx_metrics.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm


def align_token_streams(tokens_clean: np.ndarray, tokens_processed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align token tensors to shared [layers, time] support.

    Clean and processed variants of the same clip can differ by a few frames after
    codec framing; truncate to the shared overlap for stable metric computation.
    """
    tc = np.asarray(tokens_clean)
    tp = np.asarray(tokens_processed)

    if tc.ndim == 1:
        tc = tc[np.newaxis, :]
    elif tc.ndim > 2:
        tc = tc.reshape(tc.shape[0], -1)

    if tp.ndim == 1:
        tp = tp[np.newaxis, :]
    elif tp.ndim > 2:
        tp = tp.reshape(tp.shape[0], -1)

    n_layers = min(tc.shape[0], tp.shape[0])
    t_len = min(tc.shape[1], tp.shape[1])
    if n_layers <= 0 or t_len <= 0:
        raise ValueError(f"No overlapping token support: clean={tc.shape}, processed={tp.shape}")

    return tc[:n_layers, :t_len], tp[:n_layers, :t_len]


def align_embeddings(emb_clean: np.ndarray, emb_processed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align embedding tensors to shared support before distance/similarity metrics."""
    ec = np.asarray(emb_clean)
    ep = np.asarray(emb_processed)

    # Prefer preserving 2D [D, T] structure when available.
    if ec.ndim == 2 and ep.ndim == 2:
        d = min(ec.shape[0], ep.shape[0])
        t = min(ec.shape[1], ep.shape[1])
        if d <= 0 or t <= 0:
            raise ValueError(f"No overlapping embedding support: clean={ec.shape}, processed={ep.shape}")
        return ec[:d, :t], ep[:d, :t]

    # Fallback: flatten mismatched ranks and align by shared length.
    ec_flat = ec.reshape(-1)
    ep_flat = ep.reshape(-1)
    n = min(ec_flat.shape[0], ep_flat.shape[0])
    if n <= 0:
        raise ValueError(f"No overlapping embedding support: clean={ec.shape}, processed={ep.shape}")
    return ec_flat[:n], ep_flat[:n]


def compute_tfr(tokens_clean: np.ndarray, tokens_processed: np.ndarray) -> np.ndarray:
    """Compute token flip rate per layer."""
    # tokens shape: [n_layers, T]
    mismatches = (tokens_clean != tokens_processed).astype(float)
    tfr_per_layer = mismatches.mean(axis=1)  # [n_layers]
    return tfr_per_layer


def compute_cosine_similarity(emb_clean: np.ndarray, emb_processed: np.ndarray) -> float:
    """Compute cosine similarity between embeddings."""
    # emb shape: [D, T]
    # Flatten and compute cosine similarity
    clean_flat = emb_clean.flatten()
    proc_flat = emb_processed.flatten()
    
    dot_product = np.dot(clean_flat, proc_flat)
    norm_clean = np.linalg.norm(clean_flat)
    norm_proc = np.linalg.norm(proc_flat)
    
    if norm_clean == 0 or norm_proc == 0:
        return 0.0
    
    return dot_product / (norm_clean * norm_proc)


def compute_l2_distance(emb_clean: np.ndarray, emb_processed: np.ndarray) -> float:
    """Compute L2 Euclidean distance between embeddings."""
    return np.linalg.norm(emb_clean - emb_processed)


def compute_centroid_magnitude(emb: np.ndarray) -> float:
    """Compute mean magnitude of embedding vectors."""
    return np.linalg.norm(emb, axis=0).mean()


def process_pair(pair_dir: Path, pair_name: str) -> Dict:
    """Compute metrics for a single clean/processed pair."""
    # Load tokens
    tokens_clean = np.load(pair_dir / f"{pair_name}_clean_tokens.npy")
    tokens_processed = np.load(pair_dir / f"{pair_name}_processed_tokens.npy")
    tokens_clean, tokens_processed = align_token_streams(tokens_clean, tokens_processed)
    n_layers = int(tokens_clean.shape[0])
    
    # Compute TFR
    tfr_per_layer = compute_tfr(tokens_clean, tokens_processed)
    
    # Compute per-layer metrics
    cosine_sim_per_layer = []
    l2_dist_per_layer = []
    centroid_mag_clean = []
    centroid_mag_processed = []
    
    for i in range(n_layers):
        emb_clean = np.load(pair_dir / f"{pair_name}_clean_emb_layer{i}.npy")
        emb_processed = np.load(pair_dir / f"{pair_name}_processed_emb_layer{i}.npy")

        emb_clean_aligned, emb_processed_aligned = align_embeddings(emb_clean, emb_processed)

        cosine_sim_per_layer.append(float(compute_cosine_similarity(emb_clean_aligned, emb_processed_aligned)))
        l2_dist_per_layer.append(float(compute_l2_distance(emb_clean_aligned, emb_processed_aligned)))
        centroid_mag_clean.append(float(compute_centroid_magnitude(emb_clean_aligned)))
        centroid_mag_processed.append(float(compute_centroid_magnitude(emb_processed_aligned)))
    
    return {
        "tfr": [float(x) for x in tfr_per_layer.tolist()],
        "cosine_similarity": cosine_sim_per_layer,
        "l2_distance": l2_dist_per_layer,
        "centroid_magnitude_clean": centroid_mag_clean,
        "centroid_magnitude_processed": centroid_mag_processed,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute EGFx geometric metrics")
    parser.add_argument("--tokens-dir", type=Path, required=True,
                        help="Directory containing encoded tokens")
    parser.add_argument("--output", type=Path, default=Path("datasets/analysis/egfx_metrics.json"),
                        help="Output JSON file for metrics")
    args = parser.parse_args()
    
    # Scan known codec directories; infer layer counts from saved tokens per pair.
    codec_names = ["encodec", "q2d2", "hificodec", "dac_fsq", "speechtokenizer"]
    
    all_metrics = {}
    
    # Process each codec
    for codec_name in codec_names:
        codec_dir = args.tokens_dir / codec_name
        if not codec_dir.exists():
            print(f"⚠ Skipping {codec_name}: no tokens found")
            continue
        
        print(f"\n{'='*60}")
        print(f"Computing metrics for {codec_name.upper()}")
        print('='*60)
        
        codec_metrics = {}
        
        # Process each category
        for category_dir in codec_dir.iterdir():
            if not category_dir.is_dir():
                continue
            
            category = category_dir.name
            print(f"\n📁 Category: {category}")
            
            # Find all pairs
            pair_names = set(
                f.stem.replace("_clean_tokens", "").replace("_processed_tokens", "")
                for f in category_dir.glob("*_tokens.npy")
            )
            
            category_metrics = []
            for pair_name in tqdm(list(pair_names), desc=f"Processing {category}"):
                try:
                    metrics = process_pair(category_dir, pair_name)
                    metrics["pair_name"] = pair_name
                    category_metrics.append(metrics)
                except Exception as e:
                    print(f"\n✗ Failed to process {pair_name}: {e}")
            
            codec_metrics[category] = category_metrics
        
        all_metrics[codec_name] = codec_metrics
    
    # Save metrics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n✓ Metrics saved to {args.output}")
    print(f"\nNext step: python tok_analysis/egfx_analyze.py --metrics {args.output}")


if __name__ == "__main__":
    main()

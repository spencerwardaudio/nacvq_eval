"""Export centroid magnitude statistics for each codec (table 4 data).

Writes one JSON per codec to datasets/analysis/centroids/<codec>_centroid_stats.json
containing per-codebook: mean_mag, std_mag, skewness.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "Encodec"))
sys.path.insert(0, str(_HERE.parent / "hificodec"))
sys.path.insert(0, str(_HERE.parent / "SpeechTokenizer"))

from compute_svd_evr2 import (
    _load_encodec_weights,
    _load_q2d2_weights,
    _load_hificodec_weights,
    _load_speechtokenizer_weights,
    _build_fsq_grid,
    _fsq_tokens_to_coords,
)


def _build_fsq_centroid_grid() -> list[np.ndarray]:
    """Build a sampled FSQ grid for centroid magnitude statistics."""
    import itertools
    levels = [8, 8, 8, 8, 5, 5, 5, 5]
    per_dim = [np.linspace(-1, 1, L) for L in levels]
    grid_4d = list(itertools.product(*per_dim[:4]))
    mid_rest = [np.median(d) for d in per_dim[4:]]
    sample = np.array([list(g) + mid_rest for g in grid_4d], dtype=np.float32)
    return [sample]


def _centroid_stats(weights: list[np.ndarray]) -> list[dict]:
    """Compute magnitude stats per codebook."""
    results = []
    for W in weights:
        mags = np.linalg.norm(W, axis=-1)
        results.append({
            "mean_mag": float(np.mean(mags)),
            "std_mag": float(np.std(mags)),
            "skewness": float(sp_stats.skew(mags)),
            "n_codes": int(W.shape[0]),
            "dim": int(W.shape[1]),
        })
    return results


def export_centroids(codecs: dict[str, str], out_dir: Path) -> None:
    """Export centroid stats for each codec.

    Args:
        codecs: mapping of codec_name -> checkpoint_path
        out_dir: output directory for JSON files
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    loaders = {
        "encodec": _load_encodec_weights,
        "multi_dataset_encodec": _load_encodec_weights,
        "q2d2": _load_q2d2_weights,
        "hificodec": _load_hificodec_weights,
        "speechtokenizer": _load_speechtokenizer_weights,
        "dac_fsq": lambda _ckpt: _build_fsq_centroid_grid(),
    }

    for codec_name, ckpt in codecs.items():
        loader = loaders.get(codec_name)
        if loader is None:
            print(f"  [SKIP] {codec_name}: no learned codebook (FSQ)")
            continue
        if codec_name != "dac_fsq" and (not ckpt or not Path(ckpt).exists()):
            print(f"  [SKIP] {codec_name}: checkpoint not found ({ckpt})")
            continue

        print(f"  Exporting centroids for {codec_name}...")
        weights = loader(ckpt)
        if weights is None:
            continue

        stats = _centroid_stats(weights)
        out_path = out_dir / f"{codec_name}_centroid_stats.json"
        with open(out_path, "w") as f:
            json.dump({"codec": codec_name, "codebooks": stats}, f, indent=2)
        print(f"    -> {out_path} ({len(stats)} codebooks)")


if __name__ == "__main__":
    # CLI usage: python export_centroid_stats.py <codec_name> <checkpoint_path> [out_dir]
    if len(sys.argv) < 3:
        print("Usage: python export_centroid_stats.py <codec_name> <checkpoint> [out_dir]")
        sys.exit(1)
    name = sys.argv[1]
    ckpt = sys.argv[2]
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else _HERE.parent / "datasets" / "analysis" / "centroids"
    export_centroids({name: ckpt}, out)

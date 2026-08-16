"""Compute SVD(ΔZ) EVR metrics for all three perturbation types.

Reads pre-saved token .npy files and model checkpoint weights.
For each codebook unit and perturbation type builds delta-embedding matrix:
  ΔZ[m] = W[var_tok[cb]].mean(0) − W[base_tok[cb]].mean(0)
stacked across all (signal, variant) pairs, then computes:
  evr2  = (σ₁² + σ₂²) / Σσᵢ²   ideal ≈ 1 for phase (circle → 2-D plane)
  evr1  = σ₁² / Σσᵢ²            ideal ≈ 1 for amp/temporal (line → 1-D)
  eff_rank = exp(−Σ pᵢ log pᵢ)  Shannon entropy rank

Output: --output-dir/{codec}/svd_stats_bw{bw}.json

Usage:
    python tok_analysis/compute_svd_evr2.py \\
        --codec         encodec \\
        --checkpoint    Encodec/outputs/.../checkpoint.pt \\
        --model-type    encodec \\
        --bw-tag        24.0 \\
        --amp-tokens    datasets/audio_tokens/dsp_self_amp \\
        --phase-tokens  datasets/audio_tokens/dsp_self_phase \\
        --temporal-tokens datasets/audio_tokens/time_sine \\
        --output-dir    datasets/analysis/svd
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_PROJ_ROOT / "Encodec"))
sys.path.insert(0, str(_PROJ_ROOT / "hificodec"))
sys.path.insert(0, str(_PROJ_ROOT / "SpeechTokenizer"))

# ---------------------------------------------------------------------------
# SVD constants
# ---------------------------------------------------------------------------
_MIN_VARIANTS_FOR_SVD = 2          # SVD requires ≥ 2 rows in ΔZ
_NEAR_ZERO = 1e-12                  # floor to avoid divide-by-zero / log(0) in EVR
_PHASE_SIGNAL_GLOB = "self_phase_0dB_*"   # 0 dBFS only — pure phase variation (excludes -70 dBFS group)
_AMP_SIGNAL_GLOB   = "self_amp_*"         # amplitude sinusoid test signals
# Using only 0 dBFS phase signals isolates phase from amplitude effects


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def _load_npy(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        return np.load(str(path))
    except Exception as exc:
        print(f"  [WARN] Could not load {path.name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Codebook weight loading
# ---------------------------------------------------------------------------

def _load_codebook_weights(checkpoint: str, model_type: str) -> list[np.ndarray] | None:
    """Return list of [codebook_size, dim] weight arrays, one per RVQ unit.

    Returns None if the checkpoint cannot be loaded or model_type has no loader.
    """
    if not checkpoint:
        return None
    cp = Path(checkpoint)
    if not cp.exists():
        print(f"  [WARN] Checkpoint not found: {cp}")
        return None

    if model_type == "q2d2":
        return _load_q2d2_weights(checkpoint)
    if model_type in ("encodec", "multi_dataset_encodec"):
        return _load_encodec_weights(checkpoint)
    if model_type == "hificodec":
        return _load_hificodec_weights(checkpoint)
    if model_type == "speechtokenizer":
        return _load_speechtokenizer_weights(checkpoint)
    if model_type == "dac_fsq":
        return _build_fsq_grid()
    print(f"  [INFO] No weight loader for model_type={model_type!r} — SVD skipped")
    return None


def _load_encodec_weights(checkpoint: str) -> list[np.ndarray] | None:
    try:
        from compress import MODELS as _MODELS  # noqa: F401
        print(f"  Loading EnCodec checkpoint: {Path(checkpoint).name}")
        model = _MODELS["multi_dataset_encodec"](checkpoint)
        model.eval()
        weights = [
            layer._codebook.embed.detach().cpu().float().numpy()
            for layer in model.quantizer.vq.layers
        ]
        print(f"  Extracted {len(weights)} codebooks [{weights[0].shape}]")
        del model
        return weights
    except Exception as exc:
        print(f"  [WARN] EnCodec weight extraction failed: {exc}")
        return None


def _load_q2d2_weights(checkpoint: str) -> list[np.ndarray] | None:
    try:
        from q2d2_to_tokens_npy import load_q2d2_lightning  # noqa: F401
        print(f"  Loading Q2D2 checkpoint: {Path(checkpoint).name}")
        model, _ = load_q2d2_lightning(Path(checkpoint), "cpu")
        quantizer = model.feature_extractor.encodec.quantizer.quantizer.vq
        grids = [g.detach().cpu().float().numpy() for g in quantizer.tile_grid]
        print(f"  Extracted {len(grids)} Q2D2 grid pairs, shapes: {[g.shape for g in grids[:4]]}")
        del model
        return grids
    except Exception as exc:
        print(f"  [WARN] Q2D2 weight extraction failed: {exc}")
        return None


def _load_hificodec_weights(checkpoint: str) -> list[np.ndarray] | None:
    try:
        import torch, json
        from pathlib import Path as _P
        from academicodec.models.hificodec.env import AttrDict
        from academicodec.models.hificodec.models import Quantizer
        from academicodec.utils import scan_checkpoint, load_checkpoint

        cp_dir = _P(checkpoint)
        if cp_dir.is_file():
            cp_dir = cp_dir.parent

        cfg_candidates = list(cp_dir.rglob("config*.json"))
        if not cfg_candidates:
            hifi_root = _P(__file__).resolve().parent.parent / "hificodec"
            cfg_candidates = list(hifi_root.rglob("config*.json"))
        if not cfg_candidates:
            print("  [WARN] HiFiCodec config not found")
            return None
        with open(cfg_candidates[0]) as f:
            h = AttrDict(json.load(f))

        g_best = cp_dir / "g_best"
        if not g_best.exists():
            print(f"  [WARN] No g_best checkpoint in {cp_dir}")
            return None
        state = load_checkpoint(str(g_best), "cpu")

        quantizer = Quantizer(h)
        quantizer.load_state_dict(state["quantizer"])
        quantizer.eval()

        weights = []
        for mod in quantizer.quantizer_modules:
            weights.append(mod.embedding.weight.detach().cpu().float().numpy())
        for mod in quantizer.quantizer_modules2:
            weights.append(mod.embedding.weight.detach().cpu().float().numpy())
        print(f"  Extracted {len(weights)} HiFiCodec codebooks [{weights[0].shape}]")
        del quantizer
        return weights
    except Exception as exc:
        print(f"  [WARN] HiFiCodec weight extraction failed: {exc}")
        return None


def _load_speechtokenizer_weights(checkpoint: str) -> list[np.ndarray] | None:
    try:
        from pathlib import Path as _P
        from speechtokenizer.model import SpeechTokenizer as _ST

        cp_path = _P(checkpoint)
        st_root = _P(__file__).resolve().parent.parent / "SpeechTokenizer"
        cfg_candidates = [
            st_root / "config" / "fsd50k_cfg.json",
            st_root / "config" / "fsd50k_cfg_test_2ep.json",
            st_root / "config" / "spt_base_cfg.json",
            st_root / "speechtokenizer" / "config.json",
        ]
        config_path = next((p for p in cfg_candidates if p.exists()), None)
        if config_path is None:
            print("  [WARN] SpeechTokenizer config not found")
            return None

        model = _ST.load_from_checkpoint(str(config_path), str(cp_path))
        model.eval()
        weights = []
        for layer in model.quantizer.vq.layers:
            weights.append(layer._codebook.embed.detach().cpu().float().numpy())
        print(f"  Extracted {len(weights)} SpeechTokenizer codebooks [{weights[0].shape}]")
        del model
        return weights
    except Exception as exc:
        print(f"  [WARN] SpeechTokenizer weight extraction failed: {exc}")
        return None


def _build_fsq_grid(levels: list[int] | None = None) -> str:
    """Return sentinel; FSQ tokens are decoded to coordinates directly."""
    if levels is None:
        levels = [8, 8, 8, 8, 5, 5, 5, 5]
    print(f"  FSQ direct-coordinate mode: levels={levels}, dim={len(levels)}")
    return "FSQ_DIRECT"


_FSQ_LEVELS = [8, 8, 8, 8, 5, 5, 5, 5]


def _fsq_tokens_to_coords(tokens: np.ndarray, levels: list[int] | None = None) -> np.ndarray:
    """Decode flat FSQ token indices into 8-D coordinate vectors in [-1, 1]."""
    if levels is None:
        levels = _FSQ_LEVELS
    per_dim = [np.linspace(-1, 1, L) for L in levels]
    coords = np.zeros((len(tokens), len(levels)), dtype=np.float32)
    remainder = tokens.astype(np.int64).copy()
    for d in range(len(levels) - 1, -1, -1):
        coords[:, d] = per_dim[d][remainder % levels[d]]
        remainder //= levels[d]
    return coords


# ---------------------------------------------------------------------------
# SVD metrics
# ---------------------------------------------------------------------------

def _svd_metrics(delta_z: np.ndarray) -> tuple[float, float, float]:
    """Compute EVR₂, EVR₁, and EffRank from an [n_variants, dim] delta matrix."""
    if delta_z.shape[0] < _MIN_VARIANTS_FOR_SVD:
        return float("nan"), float("nan"), float("nan")
    _, s, _ = np.linalg.svd(delta_z, full_matrices=False)
    s2 = s ** 2
    total = s2.sum()
    if total < _NEAR_ZERO:
        return 0.0, 0.0, 1.0
    p = s2 / total
    evr2 = float(s2[:2].sum() / total)
    evr1 = float(s2[0] / total)
    eff_rank = float(np.exp(-np.sum(p * np.log(p + _NEAR_ZERO))))
    return evr2, evr1, eff_rank


def _compute_depth_slope(evr_values: list[float]) -> float:
    """Linear regression slope of EVR across unit index. NaN if < 2 valid points."""
    valid = [(i, v) for i, v in enumerate(evr_values) if not np.isnan(float(v))]
    if len(valid) < 2:
        return float("nan")
    from scipy.stats import linregress  # noqa: F401
    xs, ys = zip(*valid)
    slope = linregress(xs, ys).slope
    return float(slope)


# ---------------------------------------------------------------------------
# Delta-Z collection: sinusoid-style tests (phase, amplitude)
# ---------------------------------------------------------------------------

def _collect_sinusoid_deltas(
    tokens_root: Path,
    signal_glob: str,
    bw: str,
    weights,
) -> dict[int, list[np.ndarray]]:
    """Return per-unit dict of delta mean-embedding vectors.

    Each variant of each matching signal contributes one ΔZ per codebook unit.
    signal_glob matches signal subdirectories, e.g. "self_phase_0dB_*".
    """
    n_cb = 1 if weights == "FSQ_DIRECT" else len(weights)
    cb_deltas: dict[int, list[np.ndarray]] = {cb: [] for cb in range(n_cb)}

    signal_dirs = sorted(tokens_root.glob(signal_glob))
    if not signal_dirs:
        print(f"  [WARN] No signal dirs found: {tokens_root / signal_glob}")
        return cb_deltas

    n_processed = 0
    for sig_dir in signal_dirs:
        signal = sig_dir.name
        baseline = _load_npy(sig_dir / f"{signal}_baseline_bw{bw}_tokens.npy")
        if baseline is None or baseline.ndim != 2:
            continue
        for vp in sorted(sig_dir.glob(f"{signal}_var_*_bw{bw}_tokens.npy")):
            variant = _load_npy(vp)
            if variant is None or variant.ndim != 2:
                continue
            n_frames = min(baseline.shape[1], variant.shape[1])
            for cb in range(min(n_cb, baseline.shape[0], variant.shape[0])):
                # FSQ: decode token indices to 8-D coordinates directly
                if weights == "FSQ_DIRECT":
                    z_base = _fsq_tokens_to_coords(baseline[cb, :n_frames]).mean(axis=0)
                    z_var  = _fsq_tokens_to_coords(variant[cb, :n_frames]).mean(axis=0)
                else:
                    W = weights[cb]
                    z_base = W[baseline[cb, :n_frames]].mean(axis=0)
                    z_var  = W[variant[cb, :n_frames]].mean(axis=0)
                cb_deltas[cb].append(z_var - z_base)
            n_processed += 1

    print(f"  {len(signal_dirs)} signal dirs, {n_processed} variant files processed")
    return cb_deltas


# ---------------------------------------------------------------------------
# Delta-Z collection: temporal test (baseline_0ms / offset_*ms naming)
# ---------------------------------------------------------------------------

def _collect_temporal_deltas(
    tokens_root: Path,
    bw: str,
    weights,
) -> dict[int, list[np.ndarray]]:
    """Return per-unit dict of delta vectors for the time-sine test."""
    n_cb = 1 if weights == "FSQ_DIRECT" else len(weights)
    cb_deltas: dict[int, list[np.ndarray]] = {cb: [] for cb in range(n_cb)}

    freq_dirs = sorted([d for d in tokens_root.iterdir() if d.is_dir()])
    if not freq_dirs:
        print(f"  [WARN] No frequency subdirs found in {tokens_root}")
        return cb_deltas

    n_processed = 0
    for freq_dir in freq_dirs:
        baseline = _load_npy(freq_dir / f"baseline_0ms_bw{bw}_tokens.npy")
        if baseline is None or baseline.ndim != 2:
            continue
        for vp in sorted(freq_dir.glob(f"offset_*ms_bw{bw}_tokens.npy")):
            variant = _load_npy(vp)
            if variant is None or variant.ndim != 2:
                continue
            n_frames = min(baseline.shape[1], variant.shape[1])
            for cb in range(min(n_cb, baseline.shape[0], variant.shape[0])):
                # FSQ: decode token indices to 8-D coordinates directly
                if weights == "FSQ_DIRECT":
                    z_base = _fsq_tokens_to_coords(baseline[cb, :n_frames]).mean(axis=0)
                    z_var  = _fsq_tokens_to_coords(variant[cb, :n_frames]).mean(axis=0)
                else:
                    W = weights[cb]
                    z_base = W[baseline[cb, :n_frames]].mean(axis=0)
                    z_var  = W[variant[cb, :n_frames]].mean(axis=0)
                cb_deltas[cb].append(z_var - z_base)
            n_processed += 1

    print(f"  {len(freq_dirs)} frequency dirs, {n_processed} offset files processed")
    return cb_deltas


# ---------------------------------------------------------------------------
# Per-perturbation SVD aggregation
# ---------------------------------------------------------------------------

def _compute_svd_stats(
    cb_deltas: dict[int, list[np.ndarray]],
    n_cb: int,
) -> dict[str, list[float]]:
    """Compute EVR₂, EVR₁, EffRank for each codebook unit."""
    evr2_per_cb:  list[float] = []
    evr1_per_cb:  list[float] = []
    eff_rank_per_cb: list[float] = []

    for cb in range(n_cb):
        deltas = cb_deltas.get(cb, [])
        if len(deltas) < 2:
            evr2_per_cb.append(float("nan"))
            evr1_per_cb.append(float("nan"))
            eff_rank_per_cb.append(float("nan"))
            continue
        ev2, ev1, er = _svd_metrics(np.stack(deltas))
        evr2_per_cb.append(ev2)
        evr1_per_cb.append(ev1)
        eff_rank_per_cb.append(er)

    return {"evr2": evr2_per_cb, "evr1": evr1_per_cb, "eff_rank": eff_rank_per_cb}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    _ds = _PROJ_ROOT / "datasets"
    ap.add_argument("--codec", required=True,
                    help="Codec name for output subdir, e.g. encodec")
    ap.add_argument("--checkpoint", required=True,
                    help="Path to model checkpoint")
    ap.add_argument("--model-type", required=True,
                    choices=["encodec", "q2d2", "hificodec", "speechtokenizer", "dac_fsq"],
                    help="Model architecture identifier")
    ap.add_argument("--bw-tag", required=True,
                    help="Bandwidth tag used in token filenames, e.g. 24.0")
    ap.add_argument("--amp-tokens",
                    default=str(_ds / "audio_tokens" / "dsp_self_amp"),
                    help="Token root for amplitude test")
    ap.add_argument("--phase-tokens",
                    default=str(_ds / "audio_tokens" / "dsp_self_phase"),
                    help="Token root for phase test")
    ap.add_argument("--temporal-tokens",
                    default=str(_ds / "audio_tokens" / "time_sine"),
                    help="Token root for temporal sine test")
    ap.add_argument("--output-dir",
                    default=str(_ds / "analysis" / "svd"),
                    help="Parent output dir; codec subdir created inside")
    args = ap.parse_args()

    bw           = args.bw_tag
    amp_root     = Path(args.amp_tokens)
    phase_root   = Path(args.phase_tokens)
    temporal_root = Path(args.temporal_tokens)
    out_dir      = Path(args.output_dir) / args.codec
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  SVD(ΔZ) EVR — {args.codec}  bw={bw}")
    print(f"{'='*60}")

    weights = _load_codebook_weights(args.checkpoint, args.model_type)
    if weights is None:
        out_path = out_dir / f"svd_stats_bw{bw}.json"
        stub = {
            "codec": args.codec, "model_type": args.model_type, "bw": bw,
            "n_units": 0, "error": "codebook weights unavailable",
            "phase":     {"evr2": [], "evr1": [], "eff_rank": [], "depth_evr2_slope": float("nan")},
            "amplitude": {"evr2": [], "evr1": [], "eff_rank": [], "depth_evr2_slope": float("nan")},
            "temporal":  {"evr2": [], "evr1": [], "eff_rank": [], "depth_evr2_slope": float("nan")},
        }
        with out_path.open("w") as f:
            json.dump(stub, f, indent=2)
        print(f"  [SKIP] Stub JSON written: {out_path}")
        return

    # FSQ returns a sentinel string; all other codecs return a list of arrays
    n_cb = 1 if weights == "FSQ_DIRECT" else len(weights)
    print(f"  {n_cb} codebook unit(s) loaded\n")

    # Phase (0 dBFS only — pure phase variation)
    print("[1/3] Phase perturbation …")
    phase_deltas  = _collect_sinusoid_deltas(phase_root, _PHASE_SIGNAL_GLOB, bw, weights)
    phase_stats   = _compute_svd_stats(phase_deltas, n_cb)
    # slope over EVR₂ — how fast circular structure degrades with depth
    phase_stats["depth_evr2_slope"] = _compute_depth_slope(phase_stats["evr2"])

    # Amplitude
    print("\n[2/3] Amplitude perturbation …")
    amp_deltas  = _collect_sinusoid_deltas(amp_root, _AMP_SIGNAL_GLOB, bw, weights)
    amp_stats   = _compute_svd_stats(amp_deltas, n_cb)
    # slope over EVR₁ — monotonic line structure across depth
    amp_stats["depth_evr2_slope"] = _compute_depth_slope(amp_stats["evr1"])

    # Temporal
    print("\n[3/3] Temporal perturbation …")
    temporal_deltas = _collect_temporal_deltas(temporal_root, bw, weights)
    temporal_stats  = _compute_svd_stats(temporal_deltas, n_cb)
    temporal_stats["depth_evr2_slope"] = _compute_depth_slope(temporal_stats["evr1"])

    out_path = out_dir / f"svd_stats_bw{bw}.json"
    result = {
        "codec": args.codec, "model_type": args.model_type,
        "bw": bw, "n_units": n_cb,
        "phase":     phase_stats,
        "amplitude": amp_stats,
        "temporal":  temporal_stats,
    }
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  ✓ Written: {out_path}")

    def _summary(name: str, vals: list[float]) -> None:
        valid = [v for v in vals if not np.isnan(float(v))]
        if valid:
            print(f"  {name}: mean={np.mean(valid):.3f}  "
                  f"range=[{min(valid):.3f}, {max(valid):.3f}]")

    _summary("Phase  EVR₂", phase_stats["evr2"])
    _summary("Amp    EVR₁", amp_stats["evr1"])
    _summary("Temp   EVR₁", temporal_stats["evr1"])
    print(f"  Phase depth slope : {phase_stats['depth_evr2_slope']:.4f}")


if __name__ == "__main__":
    main()

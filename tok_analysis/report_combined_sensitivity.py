"""Combined sensitivity report: amplitude, phase, and temporal tests.

Reads per-codebook analysis outputs from three experiments (all using pure
sine waves) and assembles a single PDF:

  Page 1  — Title / signal inventory
  Page 2  — Amplitude: 4×8 per-codebook grid  (flip rate vs dBFS level, codebooks 1–32)
  Page 3  — Phase:     4×8 per-codebook grid  (flip rate vs phase 15°–360°, codebooks 1–32)
  Page 4  — Temporal:  4×8 per-codebook grid  (flip rate vs time offset ms, codebooks 1–32)
  Page 5  — Amplitude: 4×8 per-codebook grid  (mean L2 distance vs dBFS level, codebooks 1–32)
  Page 6  — Phase:     4×8 per-codebook grid  (mean L2 distance vs phase 15°–360°, codebooks 1–32)
  Page 7  — Temporal:  4×8 per-codebook grid  (mean L2 distance vs time offset ms, codebooks 1–32)
  Page 8  — Amplitude: 4×8 per-codebook grid  (mean cosine similarity vs dBFS level, codebooks 1–32)
  Page 9  — Phase:     4×8 per-codebook grid  (mean cosine similarity vs phase 15°–360°, codebooks 1–32)
  Page 10 — Temporal:  4×8 per-codebook grid  (mean cosine similarity vs time offset ms, codebooks 1–32)
  Page 11 — Static:    4×8 per-codebook grid  (centroid magnitude histogram + Gaussian fit, codebooks 1–32)
  Page 12 — Static:    cross-codebook summary (mean magnitude ± std and skew vs codebook 1–32)
  Page 13 — Amplitude: 4×8 per-codebook grid  (mean selected-centroid magnitude vs dBFS level, codebooks 1–32)
  Page 14 — Phase:     4×8 per-codebook grid  (mean selected-centroid magnitude vs phase 15°–360°, codebooks 1–32)
  Page 15 — Temporal:  4×8 per-codebook grid  (mean selected-centroid magnitude vs time offset ms, codebooks 1–32)
  Page 16 — Amplitude: 4×8 per-codebook grid  (codebook perplexity vs dBFS level, codebooks 1–32)
  Page 17 — Phase:     4×8 per-codebook grid  (codebook perplexity vs phase 0°–360°, codebooks 1–32)
  Page 18 — Temporal:  4×8 per-codebook grid  (codebook perplexity vs time offset ms, codebooks 1–32)

Pages 5–15 require --checkpoint so the codebook embedding matrices can be
extracted from the model.  If --checkpoint is omitted or loading fails those
pages are silently skipped.

Pages 16–18 (perplexity) are always generated — they only need the token .npy
files, not the checkpoint.  Perplexity = exp(H) where H is the Shannon entropy
of the per-codebook token distribution for a single file.  Maximum = vocab_size
(all tokens equally used), minimum ≈ 1 (all frames map to one token).

The analysis code is imported directly from the individual analyzers so the
same plot-building functions are reused without duplication.

Usage:
    python report_combined_sensitivity.py \\
        --amp-tokens   <path/to/audio_tokens/dsp_self_amp>   \\
        --phase-tokens <path/to/audio_tokens/dsp_self_phase> \\
        --temporal-tokens <path/to/audio_tokens/time_sine>   \\
        --bandwidth 24.0 \\
        --checkpoint  <path/to/checkpoint.pt> \\
        --output   <path/to/combined_sensitivity_report_bw24.0.pdf>

All --*-tokens arguments default to the standard dataset layout under
<project_root>/datasets/audio_tokens/.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
_ENCODEC_DIR = _PROJ_ROOT / "Encodec"

# Import per-codebook figure builders from individual analyzers
sys.path.insert(0, str(_HERE))
if str(_ENCODEC_DIR) not in sys.path:
    sys.path.insert(0, str(_ENCODEC_DIR))
sys.path.insert(0, str(_HERE.parent / "hificodec"))
sys.path.insert(0, str(_HERE.parent / "SpeechTokenizer"))
from analyze_self_amp_test import (  # noqa: E402
    SIGNALS as AMP_SIGNALS,
    _TAG_TO_LABEL as AMP_TAG_TO_LABEL,
    analyze_signal as amp_analyze_signal,
    _per_codebook_amplitude_figure,
)
from analyze_self_phase_test import (  # noqa: E402
    SIGNALS as PHASE_SIGNALS,
    analyze_signal as phase_analyze_signal,
    _per_codebook_phase_figure,
)
from analyze_sine_temporal import (  # noqa: E402
    FREQ_TAGS as TEMPORAL_FREQ_TAGS,
    analyze_freq as temporal_analyze_freq,
    _per_codebook_temporal_figure,
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

# Supported model types and their per-unit labels
_MODEL_UNIT_LABELS: dict[str, str] = {
    "encodec":          "Codebook",
    "q2d2":             "Grid Pair",
    "hificodec":        "Codebook",
    "speechtokenizer":  "RVQ Codebook",
    "dac_fsq":          "FSQ Stream",
}


def _derive_is_q2d2(model_type: str) -> bool:
    """Return True for Q2D2 (grid-pair topology) — keeps existing is_q2d2 logic."""
    return model_type == "q2d2"


def _get_unit_label(cb: int, is_q2d2: bool = False, model_type: str = "") -> str:
    """Return per-unit label for subplot titles."""
    base = _MODEL_UNIT_LABELS.get(model_type, "Grid Pair" if is_q2d2 else "Codebook")
    return f"{base} {cb+1}"


# ---------------------------------------------------------------------------
# Codebook weight extraction
# ---------------------------------------------------------------------------

def _load_q2d2_grid_centroids(
    checkpoint: str,
    device: str = "cpu",
) -> "list[np.ndarray] | None":
    """Load Q2D2 checkpoint and extract grid centroids.
    
    Returns a list of [G_i, 2] float32 arrays, one per grid pair.
    Returns None on failure so distance pages are silently skipped.
    """
    if not checkpoint:
        return None
    try:
        import sys
        # Import load_q2d2_lightning from q2d2_to_tokens_npy.py
        sys.path.insert(0, str(_HERE))
        from q2d2_to_tokens_npy import load_q2d2_lightning
        
        print(f"  Loading Q2D2 checkpoint for grid extraction: {checkpoint}")
        model, model_type = load_q2d2_lightning(Path(checkpoint), device)
        
        # Extract grids from quantizer
        quantizer = model.feature_extractor.encodec.quantizer.quantizer.vq
        grids = [g.detach().cpu().float().numpy() for g in quantizer.tile_grid]
        
        print(f"  Extracted {len(grids)} grid pairs, sizes: {[g.shape for g in grids[:4]]}...")
        return grids
    except Exception as exc:
        print(f"  [WARN] Q2D2 grid extraction failed — distance pages will be skipped: {exc}")
        return None


def _load_hificodec_weights(checkpoint: str) -> "list[np.ndarray] | None":
    """Extract HiFiCodec quantizer embedding weights."""
    try:
        import torch, json
        from academicodec.models.hificodec.env import AttrDict
        from academicodec.models.hificodec.models import Quantizer
        from academicodec.utils import scan_checkpoint, load_checkpoint

        cp_dir = Path(checkpoint)
        if cp_dir.is_file():
            cp_dir = cp_dir.parent
        cfg_candidates = list(cp_dir.rglob("config*.json"))
        if not cfg_candidates:
            hifi_root = _HERE.parent / "hificodec"
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


def _load_speechtokenizer_weights(checkpoint: str) -> "list[np.ndarray] | None":
    """Extract SpeechTokenizer VQ codebook embeddings."""
    try:
        from speechtokenizer.model import SpeechTokenizer as _ST

        cp_path = Path(checkpoint)
        st_root = _HERE.parent / "SpeechTokenizer"
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


def _build_fsq_grid(levels=None) -> "list[np.ndarray]":
    """Construct the full FSQ codebook grid; index ordering matches vector_quantize_pytorch."""
    if levels is None:
        levels = [8, 8, 8, 8, 5, 5, 5, 5]
    per_dim = [np.linspace(-1, 1, L, dtype=np.float32) for L in levels]
    # meshgrid with indexing='ij' + C-order ravel == itertools.product order
    grids = np.meshgrid(*per_dim, indexing="ij")
    grid = np.stack([g.ravel() for g in grids], axis=1)  # [vocab_size, n_dims]
    print(f"  Built FSQ full grid: {grid.shape}")
    return [grid]


def _load_codebook_weights(
    checkpoint: str,
    model_name: str = "multi_dataset_encodec",
) -> "list[np.ndarray] | None":
    """Load model checkpoint and extract RVQ codebook embeddings or Q2D2 grids.

    For EnCodec: Returns list of [codebook_size, dim] float32 arrays, one per RVQ layer.
    For Q2D2: Returns list of [grid_size, 2] float32 arrays, one per grid pair.
    Returns None on failure so distance pages are silently skipped.
    """
    # dac_fsq grid is deterministic — no checkpoint needed
    if model_name == "dac_fsq":
        return _build_fsq_grid()

    if not checkpoint:
        return None

    # Detect Q2D2 vs EnCodec from model name
    if model_name == "q2d2":
        return _load_q2d2_grid_centroids(checkpoint, device="cpu")

    if model_name == "hificodec":
        return _load_hificodec_weights(checkpoint)

    if model_name == "speechtokenizer":
        return _load_speechtokenizer_weights(checkpoint)
    
    # EnCodec path (existing code)
    try:
        import torch  # noqa: F401
        from compress import MODELS as _MODELS  # noqa: F401
        print(f"  Loading checkpoint for codebook extraction: {checkpoint}")
        model = _MODELS[model_name](checkpoint)
        model.eval()
        weights: list[np.ndarray] = []
        for layer in model.quantizer.vq.layers:
            embed = layer._codebook.embed.detach().cpu().float().numpy()
            weights.append(embed)
        print(f"  Extracted {len(weights)} codebooks, each {weights[0].shape}")
        del model
        return weights
    except Exception as exc:
        print(f"  [WARN] Codebook extraction failed — distance pages will be skipped: {exc}")
        return None


def _random_baseline_distances(
    codebook_weights: list[np.ndarray],
    n_samples: int = 20_000,
    rng_seed: int = 42,
) -> np.ndarray:
    """Per-codebook expected L2 distance between two randomly sampled vectors.

    This is the 'null hypothesis' reference line drawn on distance plots:
    if the embedding space is unmapped, every flip should land at this distance
    regardless of the magnitude of the acoustic perturbation.
    """
    rng = np.random.default_rng(rng_seed)
    baselines = np.zeros(len(codebook_weights))
    for i, W in enumerate(codebook_weights):
        n = W.shape[0]
        idx_a = rng.integers(0, n, n_samples)
        idx_b = rng.integers(0, n, n_samples)
        # Avoid trivially comparing a vector with itself
        collision = idx_a == idx_b
        idx_b[collision] = (idx_b[collision] + 1) % n
        baselines[i] = np.linalg.norm(W[idx_a] - W[idx_b], axis=1).mean()
    return baselines


def _l2_dist_per_codebook(
    baseline_tokens: np.ndarray,       # [n_cb, n_frames]
    variant_tokens: np.ndarray,        # [n_cb, n_frames]
    codebook_weights: list[np.ndarray],  # list of [codebook_size, dim]
) -> np.ndarray:                       # [n_cb] mean L2 distance at flip sites
    """Mean L2 distance in embedding space between anchor and flipped vectors.

    Only frame positions where baseline_tokens != variant_tokens are included.
    Returns 0.0 for codebooks with no flips.
    """
    n_cb = min(baseline_tokens.shape[0], len(codebook_weights))
    n    = min(baseline_tokens.shape[1], variant_tokens.shape[1])
    distances = np.full(n_cb, np.nan)
    for cb in range(n_cb):
        a_idx = baseline_tokens[cb, :n]
        b_idx = variant_tokens[cb, :n]
        flip_mask = a_idx != b_idx
        if not flip_mask.any():
            distances[cb] = 0.0
            continue
        W = codebook_weights[cb]
        c_anchor  = W[a_idx[flip_mask]]
        c_flipped = W[b_idx[flip_mask]]
        distances[cb] = np.linalg.norm(c_anchor - c_flipped, axis=1).mean()
    return distances


def _cosine_sim_per_codebook(
    baseline_tokens: np.ndarray,         # [n_cb, n_frames]
    variant_tokens: np.ndarray,          # [n_cb, n_frames]
    codebook_weights: list[np.ndarray],  # list of [codebook_size, dim]
) -> np.ndarray:                         # [n_cb] mean cosine similarity at flip sites
    """Mean cosine similarity in embedding space between anchor and flipped vectors.

    Only frame positions where baseline_tokens != variant_tokens are included.
    Returns 1.0 for codebooks with no flips (identical vectors → cos sim = 1).
    """
    n_cb = min(baseline_tokens.shape[0], len(codebook_weights))
    n    = min(baseline_tokens.shape[1], variant_tokens.shape[1])
    sims = np.full(n_cb, np.nan)
    for cb in range(n_cb):
        a_idx = baseline_tokens[cb, :n]
        b_idx = variant_tokens[cb, :n]
        flip_mask = a_idx != b_idx
        if not flip_mask.any():
            sims[cb] = 1.0
            continue
        W = codebook_weights[cb]
        c_anchor  = W[a_idx[flip_mask]]   # [n_flips, dim]
        c_flipped = W[b_idx[flip_mask]]   # [n_flips, dim]
        eps = 1e-8
        norm_a = np.linalg.norm(c_anchor,  axis=1, keepdims=True).clip(min=eps)
        norm_b = np.linalg.norm(c_flipped, axis=1, keepdims=True).clip(min=eps)
        cos = (c_anchor / norm_a * (c_flipped / norm_b)).sum(axis=1)
        sims[cb] = cos.mean()
    return sims


# ---------------------------------------------------------------------------
# Codebook perplexity helper
# ---------------------------------------------------------------------------

def _codebook_perplexity(
    tokens: np.ndarray,
    vocab_size: "int | list[int]" = 1024,
) -> np.ndarray:
    """Per-codebook perplexity of the token distribution for a single file.

    tokens : shape [n_codebooks, T]  integer token indices.
    vocab_size : single int applied to all units, or a per-unit list.
    Returns [n_codebooks] perplexity values.
    """
    n_codebooks = tokens.shape[0]
    perps = np.zeros(n_codebooks, dtype=np.float64)
    vs_list = vocab_size if isinstance(vocab_size, list) else [vocab_size] * n_codebooks
    for cb in range(n_codebooks):
        vs = vs_list[min(cb, len(vs_list) - 1)]
        flat = tokens[cb].flatten().astype(np.int64)
        flat = np.clip(flat, 0, vs - 1)
        counts = np.bincount(flat, minlength=vs).astype(np.float64)
        total = counts.sum()
        if total == 0:
            perps[cb] = 1.0
            continue
        probs = counts / total
        nonzero = probs[probs > 0]
        H = -np.sum(nonzero * np.log(nonzero))
        perps[cb] = np.exp(H)
    return perps


# ---------------------------------------------------------------------------
# Perplexity figure builders (pages 16–18; no checkpoint required)
# ---------------------------------------------------------------------------

def _max_vocab(vocab_size) -> int:
    return max(vocab_size) if isinstance(vocab_size, list) else int(vocab_size)


def _per_codebook_amplitude_perplexity_figure(
    all_stats: list[dict],
    amp_tokens_root: Path,
    bw: str,
    vocab_size: "int | list[int]" = 1024,
    is_q2d2: bool = False,
) -> plt.Figure:
    """4×8 per-codebook grid — codebook perplexity vs amplitude level.

    Baseline (0 dBFS) perplexity is plotted as the first x-point.  A low and
    *falling* perplexity as amplitude decreases (signal fades toward silence)
    indicates that the codebook collapses onto a small token subset — the
    dictionary underutilises its capacity for near-silence content.
    """
    if not all_stats:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return fig

    n_cb    = all_stats[0]["n_codebooks"]
    n_freqs = len(all_stats)

    variant_vals_raw = all_stats[0]["variant_values"]
    x_labels = ["0 dBFS"] + [
        AMP_TAG_TO_LABEL.get(int(v), f"−{int(v)} dBFS") for v in variant_vals_raw
    ]
    n_x = 1 + len(variant_vals_raw)

    # data[x_idx, freq_idx, cb_idx]
    data = np.full((n_x, n_freqs, n_cb), np.nan)

    for fi, stats in enumerate(all_stats):
        signal    = stats["signal"]
        sub       = amp_tokens_root / signal
        base_path = sub / f"{signal}_baseline_bw{bw}_tokens.npy"
        if base_path.exists():
            base_tok = np.load(str(base_path)).astype(np.int64)
            perps = _codebook_perplexity(base_tok, vocab_size)
            data[0, fi, :len(perps)] = perps
        for xi, tag in enumerate(stats["variant_tags"], start=1):
            var_path = sub / f"{signal}_var_{tag}_bw{bw}_tokens.npy"
            if not var_path.exists():
                continue
            var_tok = np.load(str(var_path)).astype(np.int64)
            perps = _codebook_perplexity(var_tok, vocab_size)
            data[xi, fi, :len(perps)] = perps

    mean_data = np.nanmean(data, axis=1)  # [n_x, n_cb]
    std_data  = np.nanstd(data,  axis=1)

    n_cols = 8; n_rows = max(4, (n_cb + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, max(12, n_rows * 3)), sharey=True)
    fig.suptitle(
        f"Amplitude Per Codebook — Codebook Perplexity\n"
        f"(mean ± std across {n_freqs} frequencies, sine waves, bw={bw} kbps)\n"
        f"Max perplexity = {_max_vocab(vocab_size)} (uniform dictionary use); min ≈ 1 (fully collapsed)",
        fontsize=11,
    )

    y_max = _max_vocab(vocab_size) * 1.05
    for cb in range(n_cb):
        row, col = divmod(cb, 8)
        ax = axes[row][col]
        y_mean = mean_data[:, cb]
        y_std  = std_data[:, cb]

        ax.fill_between(
            range(n_x), y_mean - y_std, y_mean + y_std,
            alpha=0.25, color="darkcyan", label="±1 std"
        )
        ax.plot(range(n_x), y_mean, color="darkcyan", linewidth=1.8,
                marker="o", markersize=4, label="mean")

        baseline_perp = y_mean[0] if not np.isnan(y_mean[0]) else np.nan
        stats_text = (
            f"base={baseline_perp:.1f}\n"
            f"mean={np.nanmean(y_mean[1:]):.1f}\n"
            f"std ={np.nanmean(y_std[1:]):.1f}"
        )
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                fontsize=6, va="top", ha="left", color="#333",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

        ax.set_title(_get_unit_label(cb, is_q2d2), fontsize=9)
        ax.set_xticks(range(n_x))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=6)
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("Perplexity", fontsize=8)
        if row == n_rows - 1:
            ax.set_xlabel("Amplitude (dBFS)", fontsize=8)

    for cb in range(n_cb, n_rows * n_cols):
        row, col = divmod(cb, 8)
        if row < n_rows:
            axes[row][col].set_visible(False)

    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def _per_codebook_phase_perplexity_figure(
    stats_0dB: list[dict],
    phase_tokens_root: Path,
    bw: str,
    vocab_size: "int | list[int]" = 1024,
    is_q2d2: bool = False,
) -> plt.Figure:
    """4×8 per-codebook grid — codebook perplexity vs phase offset (0 dBFS only).

    A codebook that maintains high perplexity across all phase angles is
    distributing its vocabulary robustly.  A codebook that collapses at
    specific phase angles (e.g. 90°) is sensitive to phase-induced token
    clustering — a form of late token saturation.
    """
    if not stats_0dB:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no 0 dBFS data", ha="center", va="center")
        return fig

    n_cb    = stats_0dB[0]["n_codebooks"]
    n_freqs = len(stats_0dB)
    phase_vals = stats_0dB[0]["variant_values"]
    phase_tags = stats_0dB[0]["variant_tags"]

    # Prepend 0° baseline
    x = [0.0] + list(phase_vals)
    n_x_plot = len(x)

    data = np.full((n_x_plot, n_freqs, n_cb), np.nan)

    for fi, stats in enumerate(stats_0dB):
        signal    = stats["signal"]
        sub       = phase_tokens_root / signal
        base_path = sub / f"{signal}_baseline_bw{bw}_tokens.npy"
        if base_path.exists():
            base_tok = np.load(str(base_path)).astype(np.int64)
            perps = _codebook_perplexity(base_tok, vocab_size)
            data[0, fi, :len(perps)] = perps
        for xi, tag in enumerate(phase_tags, start=1):
            var_path = sub / f"{signal}_var_{tag}_bw{bw}_tokens.npy"
            if not var_path.exists():
                continue
            var_tok = np.load(str(var_path)).astype(np.int64)
            perps = _codebook_perplexity(var_tok, vocab_size)
            data[xi, fi, :len(perps)] = perps

    mean_data = np.nanmean(data, axis=1)
    std_data  = np.nanstd(data,  axis=1)

    n_cols = 8; n_rows = max(4, (n_cb + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 12), sharey=True)
    fig.suptitle(
        f"Phase Per Codebook — Codebook Perplexity\n"
        f"(mean ± std across {n_freqs} frequencies at 0 dBFS, sine waves, bw={bw} kbps)\n"
        f"Max perplexity = {_max_vocab(vocab_size)} (uniform dictionary use); min ≈ 1 (fully collapsed)",
        fontsize=11,
    )

    y_max = _max_vocab(vocab_size) * 1.05
    for cb in range(n_cb):
        row, col = divmod(cb, 8)
        ax = axes[row][col]
        y_mean = mean_data[:, cb]
        y_std  = std_data[:, cb]

        ax.fill_between(
            x, y_mean - y_std, y_mean + y_std,
            alpha=0.25, color="orchid", label="±1 std"
        )
        ax.plot(x, y_mean, color="orchid", linewidth=1.8,
                marker="o", markersize=3, label="mean")

        baseline_perp = y_mean[0] if not np.isnan(y_mean[0]) else np.nan
        stats_text = (
            f"base={baseline_perp:.1f}\n"
            f"mean={np.nanmean(y_mean[1:]):.1f}\n"
            f"std ={np.nanmean(y_std[1:]):.1f}"
        )
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                fontsize=6, va="top", ha="left", color="#333",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

        ax.set_title(_get_unit_label(cb, is_q2d2), fontsize=9)
        tick_x = [v for v in x if v == 0.0 or (int(v) % 90 == 0)]
        ax.set_xticks(tick_x)
        ax.set_xticklabels([f"{int(v)}°" for v in tick_x], fontsize=7)
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("Perplexity", fontsize=8)
        if row == n_rows - 1:
            ax.set_xlabel("Phase Offset (degrees)", fontsize=8)

    for cb in range(n_cb, n_rows * n_cols):
        row, col = divmod(cb, 8)
        if row < n_rows:
            axes[row][col].set_visible(False)

    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def _per_codebook_temporal_perplexity_figure(
    all_freq_data: dict,
    temporal_tokens_root: Path,
    bw: str,
    vocab_size: "int | list[int]" = 1024,
    is_q2d2: bool = False,
) -> plt.Figure:
    """4×8 per-codebook grid — codebook perplexity vs time offset.

    A sharp perplexity drop at small offsets (e.g. 5–10 ms) indicates that
    the codec's token vocabulary collapses when the signal is placed at a
    temporal boundary — early token saturation triggered by alignment effects.
    """
    from analyze_sine_temporal import ALL_OFFSETS_MS  # noqa: F401

    if not all_freq_data:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return fig

    freqs   = sorted(all_freq_data.keys())
    n_freqs = len(freqs)

    # Infer n_cb from first available token file
    n_cb = 0
    for freq in freqs:
        subdir    = temporal_tokens_root / f"{freq}hz"
        base_path = subdir / f"baseline_0ms_bw{bw}_tokens.npy"
        if base_path.exists():
            n_cb = np.load(str(base_path)).shape[0]
            break
    if n_cb == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no token files found", ha="center", va="center")
        return fig

    offset_sets    = [set(v.keys()) for v in all_freq_data.values() if v]
    common_offsets = sorted(
        offset_sets[0].intersection(*offset_sets[1:])
        if len(offset_sets) > 1 else offset_sets[0]
    ) if offset_sets else ALL_OFFSETS_MS

    display_offsets = [0] + list(common_offsets)
    n_x_plot = len(display_offsets)
    x_labels = [
        str(v) if v == 0 or (v <= 20 and v % 5 == 0) or v > 20 else ""
        for v in display_offsets
    ]

    data = np.full((n_x_plot, n_freqs, n_cb), np.nan)

    for fi, freq in enumerate(freqs):
        subdir    = temporal_tokens_root / f"{freq}hz"
        base_path = subdir / f"baseline_0ms_bw{bw}_tokens.npy"
        if base_path.exists():
            base_tok = np.load(str(base_path)).astype(np.int64)
            perps = _codebook_perplexity(base_tok, vocab_size)
            data[0, fi, :len(perps)] = perps
        for xi, offset_ms in enumerate(common_offsets, start=1):
            var_path = subdir / f"offset_{offset_ms:03d}ms_bw{bw}_tokens.npy"
            if not var_path.exists():
                continue
            var_tok = np.load(str(var_path)).astype(np.int64)
            perps = _codebook_perplexity(var_tok, vocab_size)
            data[xi, fi, :len(perps)] = perps

    mean_data = np.nanmean(data, axis=1)
    std_data  = np.nanstd(data,  axis=1)

    n_cols = 8; n_rows = max(4, (n_cb + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, max(12, n_rows * 3)), sharey=True)
    fig.suptitle(
        f"Time Delay Per Codebook — Codebook Perplexity\n"
        f"(mean ± std across {n_freqs} sine frequencies, 0 dBFS, bw={bw} kbps)\n"
        f"Max perplexity = {_max_vocab(vocab_size)} (uniform dictionary use); min ≈ 1 (fully collapsed)",
        fontsize=11,
    )

    y_max = _max_vocab(vocab_size) * 1.05
    for cb in range(n_cb):
        row, col = divmod(cb, 8)
        ax = axes[row][col]
        y_mean = mean_data[:, cb]
        y_std  = std_data[:, cb]

        ax.fill_between(
            range(n_x_plot), y_mean - y_std, y_mean + y_std,
            alpha=0.25, color="sienna", label="±1 std"
        )
        ax.plot(range(n_x_plot), y_mean, color="sienna", linewidth=1.8,
                marker="o", markersize=3, label="mean")

        baseline_perp = y_mean[0] if not np.isnan(y_mean[0]) else np.nan
        stats_text = (
            f"base={baseline_perp:.1f}\n"
            f"mean={np.nanmean(y_mean[1:]):.1f}\n"
            f"std ={np.nanmean(y_std[1:]):.1f}"
        )
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                fontsize=6, va="top", ha="left", color="#333",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

        ax.set_title(_get_unit_label(cb, is_q2d2), fontsize=9)
        ax.set_xticks(range(n_x_plot))
        ax.set_xticklabels(x_labels, fontsize=6)
        ax.set_ylim(0, y_max)
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("Perplexity", fontsize=8)
        if row == n_rows - 1:
            ax.set_xlabel("Time Offset (ms)", fontsize=8)

    for cb in range(n_cb, n_rows * n_cols):
        row, col = divmod(cb, 8)
        if row < n_rows:
            axes[row][col].set_visible(False)

    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def _mean_or_nan(values: np.ndarray) -> float:
    """Return nan-safe mean or NaN if no finite values are present."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _perplexity_summary_table_figure(
    amp_all_stats: list[dict],
    phase_0dB: list[dict],
    temporal_freq_data: dict,
    amp_tokens_root: Path,
    phase_tokens_root: Path,
    temporal_tokens_root: Path,
    bw: str,
    vocab_size: int = 1024,
    is_q2d2: bool = False,
    model_type: str = "",
) -> plt.Figure:
    """One-page per-unit perplexity summary table across amplitude/phase/temporal tests."""

    def _add_mean_perps(path_list: list[Path], n_cb_hint: int | None = None) -> tuple[np.ndarray | None, np.ndarray | None]:
        sums: np.ndarray | None = None
        counts: np.ndarray | None = None
        for token_path in path_list:
            if not token_path.exists():
                continue
            tokens = np.load(str(token_path)).astype(np.int64)
            perps = _codebook_perplexity(tokens, vocab_size=vocab_size)
            if sums is None:
                n_cb = len(perps) if n_cb_hint is None else min(n_cb_hint, len(perps))
                sums = np.zeros(n_cb, dtype=np.float64)
                counts = np.zeros(n_cb, dtype=np.float64)
            n = min(len(perps), len(sums))
            sums[:n] += perps[:n]
            counts[:n] += 1
        return sums, counts

    n_cb = 0
    if amp_all_stats:
        n_cb = int(amp_all_stats[0].get("n_codebooks", 0))
    elif phase_0dB:
        n_cb = int(phase_0dB[0].get("n_codebooks", 0))
    elif temporal_freq_data:
        for freq in sorted(temporal_freq_data.keys()):
            p = temporal_tokens_root / f"{freq}hz" / f"baseline_0ms_bw{bw}_tokens.npy"
            if p.exists():
                n_cb = int(np.load(str(p)).shape[0])
                break

    if n_cb <= 0:
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        ax.text(0.5, 0.5, "No tokens found for perplexity summary", ha="center", va="center")
        return fig

    amp_base_paths: list[Path] = []
    amp_var_paths: list[Path] = []
    for stats in amp_all_stats:
        signal = stats["signal"]
        sub = amp_tokens_root / signal
        amp_base_paths.append(sub / f"{signal}_baseline_bw{bw}_tokens.npy")
        for tag in stats.get("variant_tags", []):
            amp_var_paths.append(sub / f"{signal}_var_{tag}_bw{bw}_tokens.npy")

    phase_base_paths: list[Path] = []
    phase_var_paths: list[Path] = []
    for stats in phase_0dB:
        signal = stats["signal"]
        sub = phase_tokens_root / signal
        phase_base_paths.append(sub / f"{signal}_baseline_bw{bw}_tokens.npy")
        for tag in stats.get("variant_tags", []):
            phase_var_paths.append(sub / f"{signal}_var_{tag}_bw{bw}_tokens.npy")

    temporal_base_paths: list[Path] = []
    temporal_var_paths: list[Path] = []
    for freq in sorted(temporal_freq_data.keys()):
        sub = temporal_tokens_root / f"{freq}hz"
        temporal_base_paths.append(sub / f"baseline_0ms_bw{bw}_tokens.npy")
        for offset_ms in sorted(temporal_freq_data[freq].keys()):
            temporal_var_paths.append(sub / f"offset_{int(offset_ms):03d}ms_bw{bw}_tokens.npy")

    amp_base_sum, amp_base_cnt = _add_mean_perps(amp_base_paths, n_cb)
    amp_var_sum, amp_var_cnt = _add_mean_perps(amp_var_paths, n_cb)
    phase_base_sum, phase_base_cnt = _add_mean_perps(phase_base_paths, n_cb)
    phase_var_sum, phase_var_cnt = _add_mean_perps(phase_var_paths, n_cb)
    temp_base_sum, temp_base_cnt = _add_mean_perps(temporal_base_paths, n_cb)
    temp_var_sum, temp_var_cnt = _add_mean_perps(temporal_var_paths, n_cb)

    def _safe_div(sums: np.ndarray | None, counts: np.ndarray | None) -> np.ndarray:
        if sums is None or counts is None:
            return np.full(n_cb, np.nan, dtype=np.float64)
        out = np.full(n_cb, np.nan, dtype=np.float64)
        valid = counts > 0
        out[valid] = sums[valid] / counts[valid]
        return out

    amp_base = _safe_div(amp_base_sum, amp_base_cnt)
    amp_var = _safe_div(amp_var_sum, amp_var_cnt)
    phase_base = _safe_div(phase_base_sum, phase_base_cnt)
    phase_var = _safe_div(phase_var_sum, phase_var_cnt)
    temp_base = _safe_div(temp_base_sum, temp_base_cnt)
    temp_var = _safe_div(temp_var_sum, temp_var_cnt)

    rows = []
    for cb in range(n_cb):
        rows.append([
            _get_unit_label(cb, is_q2d2=is_q2d2, model_type=model_type),
            f"{amp_base[cb]:.1f}" if np.isfinite(amp_base[cb]) else "-",
            f"{amp_var[cb]:.1f}" if np.isfinite(amp_var[cb]) else "-",
            f"{phase_base[cb]:.1f}" if np.isfinite(phase_base[cb]) else "-",
            f"{phase_var[cb]:.1f}" if np.isfinite(phase_var[cb]) else "-",
            f"{temp_base[cb]:.1f}" if np.isfinite(temp_base[cb]) else "-",
            f"{temp_var[cb]:.1f}" if np.isfinite(temp_var[cb]) else "-",
        ])

    fig, ax = plt.subplots(figsize=(14, 11))
    ax.axis("off")
    col_labels = [
        "Unit",
        "Amp\nBaseline",
        "Amp\nVariants Mean",
        "Phase\nBaseline",
        "Phase\nVariants Mean",
        "Temporal\nBaseline",
        "Temporal\nVariants Mean",
    ]
    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.2)
    for j in range(len(col_labels)):
        header = tbl[0, j]
        header.set_facecolor("#2c3e50")
        header.set_text_props(color="white", fontweight="bold")

    summary_text = (
        f"Mean over units — Amp baseline: {_mean_or_nan(amp_base):.1f}, Amp variants: {_mean_or_nan(amp_var):.1f} | "
        f"Phase baseline: {_mean_or_nan(phase_base):.1f}, Phase variants: {_mean_or_nan(phase_var):.1f} | "
        f"Temporal baseline: {_mean_or_nan(temp_base):.1f}, Temporal variants: {_mean_or_nan(temp_var):.1f}"
    )
    fig.suptitle(
        "Per-Unit Perplexity Summary Table\n"
        f"Perplexity = exp(H), vocab={vocab_size}. Higher means broader token usage.",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(0.5, 0.04, summary_text, ha="center", va="center", fontsize=9)
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.94])
    return fig


# ---------------------------------------------------------------------------
# Distance figure builders (mirror of flip-rate grids; y-axis = L2 distance)
# ---------------------------------------------------------------------------

def _per_codebook_amplitude_distance_figure(
    all_stats: list[dict],
    amp_tokens_root: Path,
    bw: str,
    codebook_weights: list[np.ndarray],
    random_baselines: np.ndarray,
    is_q2d2: bool = False,
) -> plt.Figure:
    """4×8 per-codebook grid (codebooks 1–32 or grid pairs 1–16) — mean L2 codebook distance vs amplitude level."""
    if not all_stats or not codebook_weights:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return fig

    n_cb    = min(all_stats[0]["n_codebooks"], len(codebook_weights))
    n_freqs = len(all_stats)

    variant_vals_raw = all_stats[0]["variant_values"]
    x_levels = [0.0] + list(variant_vals_raw)
    x_labels  = ["0 dBFS"] + [
        AMP_TAG_TO_LABEL.get(int(v), f"−{int(v)} dBFS") for v in variant_vals_raw
    ]
    n_x = len(x_levels)

    data = np.full((n_x, n_freqs, n_cb), np.nan)
    for fi, stats in enumerate(all_stats):
        data[0, fi, :n_cb] = 0.0  # baseline: distance to itself = 0
        signal   = stats["signal"]
        sub      = amp_tokens_root / signal
        base_path = sub / f"{signal}_baseline_bw{bw}_tokens.npy"
        if not base_path.exists():
            continue
        base_tok = np.load(str(base_path)).astype(np.int64)
        for xi, (_, tag) in enumerate(
            zip(stats["variant_values"], stats["variant_tags"]), start=1
        ):
            var_path = sub / f"{signal}_var_{tag}_bw{bw}_tokens.npy"
            if not var_path.exists():
                continue
            var_tok = np.load(str(var_path)).astype(np.int64)
            dists = _l2_dist_per_codebook(base_tok, var_tok, codebook_weights)
            data[xi, fi, :len(dists)] = dists

    mean_data = np.nanmean(data, axis=1)  # [n_x, n_cb]
    std_data  = np.nanstd(data,  axis=1)
    var_data  = np.nanvar(data,  axis=1)

    n_cols = 8; n_rows = max(4, (n_cb + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, max(12, n_rows * 3)), sharey=True)
    fig.suptitle(
        f"Amplitude Per Codebook — Mean L2 Codebook Distance\n"
        f"(mean ± std across {n_freqs} frequencies, sine waves, bw={bw} kbps)",
        fontsize=12,
    )

    for cb in range(n_cb):
        row, col = divmod(cb, 8)
        ax = axes[row][col]
        y_mean = mean_data[:, cb]
        y_std  = std_data[:, cb]
        y_var  = var_data[:, cb]

        ax.fill_between(
            range(n_x), y_mean - y_std, y_mean + y_std,
            alpha=0.25, color="tomato", label="±1 std"
        )
        ax.plot(range(n_x), y_mean, color="tomato", linewidth=1.8,
                marker="o", markersize=4, label="mean")

        stats_text = (
            f"mean={np.nanmean(y_mean[1:]):.3f}\n"
            f"std ={np.nanmean(y_std[1:]):.3f}\n"
            f"var ={np.nanmean(y_var[1:]):.3f}"
        )
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                fontsize=6, va="top", ha="left", color="#333",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

        ax.set_title(_get_unit_label(cb, is_q2d2), fontsize=9)
        ax.set_xticks(range(n_x))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=6)
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("Mean L2 Distance", fontsize=8)
        if row == 3:
            ax.set_xlabel("Amplitude (dBFS)", fontsize=8)

    for cb in range(n_cb, 32):
        row, col = divmod(cb, 8)
        axes[row][col].set_visible(False)

    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def _per_codebook_phase_distance_figure(
    stats_0dB: list[dict],
    phase_tokens_root: Path,
    bw: str,
    codebook_weights: list[np.ndarray],
    random_baselines: np.ndarray,
    is_q2d2: bool = False,
) -> plt.Figure:
    """4×8 per-codebook grid (codebooks 1–32 or grid pairs 1–16) — mean L2 codebook distance vs phase offset."""
    if not stats_0dB or not codebook_weights:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no 0 dBFS data", ha="center", va="center")
        return fig

    n_cb    = min(stats_0dB[0]["n_codebooks"], len(codebook_weights))
    n_freqs = len(stats_0dB)
    phase_vals = stats_0dB[0]["variant_values"]
    phase_tags = stats_0dB[0]["variant_tags"]
    n_x = len(phase_vals)

    data = np.full((n_x, n_freqs, n_cb), np.nan)
    for fi, stats in enumerate(stats_0dB):
        signal    = stats["signal"]
        sub       = phase_tokens_root / signal
        base_path = sub / f"{signal}_baseline_bw{bw}_tokens.npy"
        if not base_path.exists():
            continue
        base_tok = np.load(str(base_path)).astype(np.int64)
        for xi, tag in enumerate(phase_tags):
            var_path = sub / f"{signal}_var_{tag}_bw{bw}_tokens.npy"
            if not var_path.exists():
                continue
            var_tok = np.load(str(var_path)).astype(np.int64)
            dists = _l2_dist_per_codebook(base_tok, var_tok, codebook_weights)
            data[xi, fi, :len(dists)] = dists

    mean_data = np.nanmean(data, axis=1)
    std_data  = np.nanstd(data,  axis=1)
    var_data  = np.nanvar(data,  axis=1)

    # Prepend 0° baseline (distance = 0 by definition)
    zero_row  = np.zeros((1, n_cb))
    mean_data = np.vstack([zero_row, mean_data])
    std_data  = np.vstack([zero_row, std_data])
    var_data  = np.vstack([zero_row, var_data])

    x = [0.0] + list(phase_vals)
    n_x_plot = len(x)

    n_cols = 8; n_rows = max(4, (n_cb + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 12), sharey=True)
    fig.suptitle(
        f"Phase Per Codebook — Mean L2 Codebook Distance\n"
        f"(mean ± std across {n_freqs} frequencies at 0 dBFS, sine waves, bw={bw} kbps)",
        fontsize=12,
    )

    for cb in range(n_cb):
        row, col = divmod(cb, 8)
        ax = axes[row][col]
        y_mean = mean_data[:, cb]
        y_std  = std_data[:, cb]
        y_var  = var_data[:, cb]

        ax.fill_between(
            x, y_mean - y_std, y_mean + y_std,
            alpha=0.25, color="mediumpurple", label="±1 std"
        )
        ax.plot(x, y_mean, color="mediumpurple", linewidth=1.8,
                marker="o", markersize=3, label="mean")

        stats_text = (
            f"mean={np.nanmean(y_mean[1:]):.3f}\n"
            f"std ={np.nanmean(y_std[1:]):.3f}\n"
            f"var ={np.nanmean(y_var[1:]):.3f}"
        )
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                fontsize=6, va="top", ha="left", color="#333",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

        ax.set_title(_get_unit_label(cb, is_q2d2), fontsize=9)
        tick_x = [v for v in x if v == 0.0 or (int(v) % 90 == 0)]
        ax.set_xticks(tick_x)
        ax.set_xticklabels([f"{int(v)}°" for v in tick_x], fontsize=7)
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("Mean L2 Distance", fontsize=8)
        if row == 3:
            ax.set_xlabel("Phase Offset (degrees)", fontsize=8)

    for cb in range(n_cb, 32):
        row, col = divmod(cb, 8)
        axes[row][col].set_visible(False)

    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def _per_codebook_temporal_distance_figure(
    all_freq_data: dict,
    temporal_tokens_root: Path,
    bw: str,
    codebook_weights: list[np.ndarray],
    random_baselines: np.ndarray,
    is_q2d2: bool = False,
) -> plt.Figure:
    """4×8 per-codebook grid (codebooks 1–32 or grid pairs 1–16) — mean L2 codebook distance vs time offset."""
    from analyze_sine_temporal import ALL_OFFSETS_MS  # noqa: F401

    if not all_freq_data or not codebook_weights:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return fig

    freqs   = sorted(all_freq_data.keys())
    n_freqs = len(freqs)
    n_cb    = min(
        next(iter(next(iter(all_freq_data.values())).values())).shape[0],
        len(codebook_weights),
    )

    offset_sets    = [set(v.keys()) for v in all_freq_data.values() if v]
    common_offsets = sorted(
        offset_sets[0].intersection(*offset_sets[1:])
        if len(offset_sets) > 1 else offset_sets[0]
    ) if offset_sets else ALL_OFFSETS_MS
    n_x = len(common_offsets)

    data = np.full((n_x, n_freqs, n_cb), np.nan)
    for fi, freq in enumerate(freqs):
        subdir    = temporal_tokens_root / f"{freq}hz"
        base_path = subdir / f"baseline_0ms_bw{bw}_tokens.npy"
        if not base_path.exists():
            continue
        base_tok = np.load(str(base_path)).astype(np.int64)
        for xi, offset_ms in enumerate(common_offsets):
            var_path = subdir / f"offset_{offset_ms:03d}ms_bw{bw}_tokens.npy"
            if not var_path.exists():
                continue
            var_tok = np.load(str(var_path)).astype(np.int64)
            dists = _l2_dist_per_codebook(base_tok, var_tok, codebook_weights)
            data[xi, fi, :len(dists)] = dists

    mean_data = np.nanmean(data, axis=1)
    std_data  = np.nanstd(data,  axis=1)
    var_data  = np.nanvar(data,  axis=1)

    # Prepend 0 ms baseline (distance = 0 by definition)
    zero_row  = np.zeros((1, n_cb))
    mean_data = np.vstack([zero_row, mean_data])
    std_data  = np.vstack([zero_row, std_data])
    var_data  = np.vstack([zero_row, var_data])

    display_offsets = [0] + list(common_offsets)
    n_x_plot = len(display_offsets)
    x_labels = [
        str(v) if v == 0 or (v <= 20 and v % 5 == 0) or v > 20 else ""
        for v in display_offsets
    ]

    n_cols = 8; n_rows = max(4, (n_cb + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, max(12, n_rows * 3)), sharey=True)
    fig.suptitle(
        f"Time Delay Per Codebook — Mean L2 Codebook Distance\n"
        f"(mean ± std across {n_freqs} sine frequencies, 0 dBFS, bw={bw} kbps)",
        fontsize=12,
    )

    for cb in range(n_cb):
        row, col = divmod(cb, 8)
        ax = axes[row][col]
        y_mean = mean_data[:, cb]
        y_std  = std_data[:, cb]
        y_var  = var_data[:, cb]

        ax.fill_between(
            range(n_x_plot), y_mean - y_std, y_mean + y_std,
            alpha=0.25, color="goldenrod", label="±1 std"
        )
        ax.plot(range(n_x_plot), y_mean, color="goldenrod", linewidth=1.8,
                marker="o", markersize=3, label="mean")

        stats_text = (
            f"mean={np.nanmean(y_mean[1:]):.3f}\n"
            f"std ={np.nanmean(y_std[1:]):.3f}\n"
            f"var ={np.nanmean(y_var[1:]):.3f}"
        )
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                fontsize=6, va="top", ha="left", color="#333",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

        ax.set_title(_get_unit_label(cb, is_q2d2), fontsize=9)
        ax.set_xticks(range(n_x_plot))
        ax.set_xticklabels(x_labels, fontsize=6)
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("Mean L2 Distance", fontsize=8)
        if row == 3:
            ax.set_xlabel("Time Offset (ms)", fontsize=8)

    for cb in range(n_cb, 32):
        row, col = divmod(cb, 8)
        axes[row][col].set_visible(False)

    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ---------------------------------------------------------------------------
# Cosine similarity figure builders (pages 8–10)
# ---------------------------------------------------------------------------

def _per_codebook_amplitude_cosine_figure(
    all_stats: list[dict],
    amp_tokens_root: Path,
    bw: str,
    codebook_weights: list[np.ndarray],
    random_baselines: np.ndarray,
    is_q2d2: bool = False,
) -> plt.Figure:
    """4×8 per-codebook grid (codebooks 1–32 or grid pairs 1–16) — mean cosine similarity vs amplitude level."""
    if not all_stats or not codebook_weights:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return fig

    n_cb    = min(all_stats[0]["n_codebooks"], len(codebook_weights))
    n_freqs = len(all_stats)

    variant_vals_raw = all_stats[0]["variant_values"]
    x_levels = [0.0] + list(variant_vals_raw)
    x_labels  = ["0 dBFS"] + [
        AMP_TAG_TO_LABEL.get(int(v), f"−{int(v)} dBFS") for v in variant_vals_raw
    ]
    n_x = len(x_levels)

    data = np.full((n_x, n_freqs, n_cb), np.nan)
    for fi, stats in enumerate(all_stats):
        data[0, fi, :n_cb] = 1.0  # baseline: cos sim of vector with itself = 1
        signal   = stats["signal"]
        sub      = amp_tokens_root / signal
        base_path = sub / f"{signal}_baseline_bw{bw}_tokens.npy"
        if not base_path.exists():
            continue
        base_tok = np.load(str(base_path)).astype(np.int64)
        for xi, (_, tag) in enumerate(
            zip(stats["variant_values"], stats["variant_tags"]), start=1
        ):
            var_path = sub / f"{signal}_var_{tag}_bw{bw}_tokens.npy"
            if not var_path.exists():
                continue
            var_tok = np.load(str(var_path)).astype(np.int64)
            sims = _cosine_sim_per_codebook(base_tok, var_tok, codebook_weights)
            data[xi, fi, :len(sims)] = sims

    mean_data = np.nanmean(data, axis=1)
    std_data  = np.nanstd(data,  axis=1)
    var_data  = np.nanvar(data,  axis=1)

    n_cols = 8; n_rows = max(4, (n_cb + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, max(12, n_rows * 3)), sharey=True)
    fig.suptitle(
        f"Amplitude Per Codebook — Mean Cosine Similarity\n"
        f"(mean ± std across {n_freqs} frequencies, sine waves, bw={bw} kbps)",
        fontsize=12,
    )

    for cb in range(n_cb):
        row, col = divmod(cb, 8)
        ax = axes[row][col]
        y_mean = mean_data[:, cb]
        y_std  = std_data[:, cb]
        y_var  = var_data[:, cb]

        ax.fill_between(
            range(n_x), y_mean - y_std, y_mean + y_std,
            alpha=0.25, color="steelblue", label="±1 std"
        )
        ax.plot(range(n_x), y_mean, color="steelblue", linewidth=1.8,
                marker="o", markersize=4, label="mean")

        stats_text = (
            f"mean={np.nanmean(y_mean[1:]):.3f}\n"
            f"std ={np.nanmean(y_std[1:]):.3f}\n"
            f"var ={np.nanmean(y_var[1:]):.3f}"
        )
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                fontsize=6, va="top", ha="left", color="#333",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

        ax.set_title(_get_unit_label(cb, is_q2d2), fontsize=9)
        ax.set_ylim(-0.6, 1.05)
        ax.set_xticks(range(n_x))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=6)
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("Mean Cosine Similarity", fontsize=8)
        if row == 3:
            ax.set_xlabel("Amplitude (dBFS)", fontsize=8)

    for cb in range(n_cb, 32):
        row, col = divmod(cb, 8)
        axes[row][col].set_visible(False)

    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def _per_codebook_phase_cosine_figure(
    stats_0dB: list[dict],
    phase_tokens_root: Path,
    bw: str,
    codebook_weights: list[np.ndarray],
    random_baselines: np.ndarray,
    is_q2d2: bool = False,
) -> plt.Figure:
    """4×8 per-codebook grid (codebooks 1–32 or grid pairs 1–16) — mean cosine similarity vs phase offset."""
    if not stats_0dB or not codebook_weights:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no 0 dBFS data", ha="center", va="center")
        return fig

    n_cb    = min(stats_0dB[0]["n_codebooks"], len(codebook_weights))
    n_freqs = len(stats_0dB)
    phase_vals = stats_0dB[0]["variant_values"]
    phase_tags = stats_0dB[0]["variant_tags"]
    n_x = len(phase_vals)

    data = np.full((n_x, n_freqs, n_cb), np.nan)
    for fi, stats in enumerate(stats_0dB):
        signal    = stats["signal"]
        sub       = phase_tokens_root / signal
        base_path = sub / f"{signal}_baseline_bw{bw}_tokens.npy"
        if not base_path.exists():
            continue
        base_tok = np.load(str(base_path)).astype(np.int64)
        for xi, tag in enumerate(phase_tags):
            var_path = sub / f"{signal}_var_{tag}_bw{bw}_tokens.npy"
            if not var_path.exists():
                continue
            var_tok = np.load(str(var_path)).astype(np.int64)
            sims = _cosine_sim_per_codebook(base_tok, var_tok, codebook_weights)
            data[xi, fi, :len(sims)] = sims

    mean_data = np.nanmean(data, axis=1)
    std_data  = np.nanstd(data,  axis=1)
    var_data  = np.nanvar(data,  axis=1)

    # Prepend 0° baseline (cos sim = 1 by definition)
    one_row   = np.ones((1, n_cb))
    zero_row  = np.zeros((1, n_cb))
    mean_data = np.vstack([one_row,  mean_data])
    std_data  = np.vstack([zero_row, std_data])
    var_data  = np.vstack([zero_row, var_data])

    x = [0.0] + list(phase_vals)
    n_x_plot = len(x)

    n_cols = 8; n_rows = max(4, (n_cb + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 12), sharey=True)
    fig.suptitle(
        f"Phase Per Codebook — Mean Cosine Similarity\n"
        f"(mean ± std across {n_freqs} frequencies at 0 dBFS, sine waves, bw={bw} kbps)",
        fontsize=12,
    )

    for cb in range(n_cb):
        row, col = divmod(cb, 8)
        ax = axes[row][col]
        y_mean = mean_data[:, cb]
        y_std  = std_data[:, cb]
        y_var  = var_data[:, cb]

        ax.fill_between(
            x, y_mean - y_std, y_mean + y_std,
            alpha=0.25, color="mediumseagreen", label="±1 std"
        )
        ax.plot(x, y_mean, color="mediumseagreen", linewidth=1.8,
                marker="o", markersize=3, label="mean")

        stats_text = (
            f"mean={np.nanmean(y_mean[1:]):.3f}\n"
            f"std ={np.nanmean(y_std[1:]):.3f}\n"
            f"var ={np.nanmean(y_var[1:]):.3f}"
        )
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                fontsize=6, va="top", ha="left", color="#333",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

        ax.set_title(_get_unit_label(cb, is_q2d2), fontsize=9)
        ax.set_ylim(-0.6, 1.05)
        tick_x = [v for v in x if v == 0.0 or (int(v) % 90 == 0)]
        ax.set_xticks(tick_x)
        ax.set_xticklabels([f"{int(v)}°" for v in tick_x], fontsize=7)
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("Mean Cosine Similarity", fontsize=8)
        if row == 3:
            ax.set_xlabel("Phase Offset (degrees)", fontsize=8)

    for cb in range(n_cb, 32):
        row, col = divmod(cb, 8)
        axes[row][col].set_visible(False)

    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def _per_codebook_temporal_cosine_figure(
    all_freq_data: dict,
    temporal_tokens_root: Path,
    bw: str,
    codebook_weights: list[np.ndarray],
    random_baselines: np.ndarray,
    is_q2d2: bool = False,
) -> plt.Figure:
    """4×8 per-codebook grid (codebooks 1–32 or grid pairs 1–16) — mean cosine similarity vs time offset."""
    from analyze_sine_temporal import ALL_OFFSETS_MS  # noqa: F401

    if not all_freq_data or not codebook_weights:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return fig

    freqs   = sorted(all_freq_data.keys())
    n_freqs = len(freqs)
    n_cb    = min(
        next(iter(next(iter(all_freq_data.values())).values())).shape[0],
        len(codebook_weights),
    )

    offset_sets    = [set(v.keys()) for v in all_freq_data.values() if v]
    common_offsets = sorted(
        offset_sets[0].intersection(*offset_sets[1:])
        if len(offset_sets) > 1 else offset_sets[0]
    ) if offset_sets else ALL_OFFSETS_MS
    n_x = len(common_offsets)

    data = np.full((n_x, n_freqs, n_cb), np.nan)
    for fi, freq in enumerate(freqs):
        subdir    = temporal_tokens_root / f"{freq}hz"
        base_path = subdir / f"baseline_0ms_bw{bw}_tokens.npy"
        if not base_path.exists():
            continue
        base_tok = np.load(str(base_path)).astype(np.int64)
        for xi, offset_ms in enumerate(common_offsets):
            var_path = subdir / f"offset_{offset_ms:03d}ms_bw{bw}_tokens.npy"
            if not var_path.exists():
                continue
            var_tok = np.load(str(var_path)).astype(np.int64)
            sims = _cosine_sim_per_codebook(base_tok, var_tok, codebook_weights)
            data[xi, fi, :len(sims)] = sims

    mean_data = np.nanmean(data, axis=1)
    std_data  = np.nanstd(data,  axis=1)
    var_data  = np.nanvar(data,  axis=1)

    # Prepend 0 ms baseline (cos sim = 1 by definition)
    one_row   = np.ones((1, n_cb))
    zero_row  = np.zeros((1, n_cb))
    mean_data = np.vstack([one_row,  mean_data])
    std_data  = np.vstack([zero_row, std_data])
    var_data  = np.vstack([zero_row, var_data])

    display_offsets = [0] + list(common_offsets)
    n_x_plot = len(display_offsets)
    x_labels = [
        str(v) if v == 0 or (v <= 20 and v % 5 == 0) or v > 20 else ""
        for v in display_offsets
    ]

    n_cols = 8; n_rows = max(4, (n_cb + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, max(12, n_rows * 3)), sharey=True)
    fig.suptitle(
        f"Time Delay Per Codebook — Mean Cosine Similarity\n"
        f"(mean ± std across {n_freqs} sine frequencies, 0 dBFS, bw={bw} kbps)",
        fontsize=12,
    )

    for cb in range(n_cb):
        row, col = divmod(cb, 8)
        ax = axes[row][col]
        y_mean = mean_data[:, cb]
        y_std  = std_data[:, cb]
        y_var  = var_data[:, cb]

        ax.fill_between(
            range(n_x_plot), y_mean - y_std, y_mean + y_std,
            alpha=0.25, color="darkorange", label="±1 std"
        )
        ax.plot(range(n_x_plot), y_mean, color="darkorange", linewidth=1.8,
                marker="o", markersize=3, label="mean")

        stats_text = (
            f"mean={np.nanmean(y_mean[1:]):.3f}\n"
            f"std ={np.nanmean(y_std[1:]):.3f}\n"
            f"var ={np.nanmean(y_var[1:]):.3f}"
        )
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                fontsize=6, va="top", ha="left", color="#333",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

        ax.set_title(_get_unit_label(cb, is_q2d2), fontsize=9)
        ax.set_ylim(-0.6, 1.05)
        ax.set_xticks(range(n_x_plot))
        ax.set_xticklabels(x_labels, fontsize=6)
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("Mean Cosine Similarity", fontsize=8)
        if row == 3:
            ax.set_xlabel("Time Offset (ms)", fontsize=8)

    for cb in range(n_cb, 32):
        row, col = divmod(cb, 8)
        axes[row][col].set_visible(False)

    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ---------------------------------------------------------------------------
# Codebook magnitude figure builders (pages 11–15)
#
# magnitude of slot i in codebook cb = ‖W[cb][i]‖₂  (L2 norm of its 128-dim
# centroid vector).  The integer token *index* is a nominal Voronoi-cell label
# and carries no geometry, but the magnitude of the centroid it points to is a
# genuine geometric quantity.  Two complementary views:
#
#   Static  (pages 11–12, checkpoint only): the distribution of all 1024
#           centroid magnitudes per codebook — a property of the trained model,
#           independent of any input.  Answers "is it Gaussian?" and
#           "are early codebooks larger than late ones?".
#
#   Trend   (pages 13–15, token files): the mean magnitude of the centroids the
#           encoder actually SELECTS, plotted against the test condition
#           (dBFS / phase / time offset).  This is NOT a usage-weighted
#           histogram — it is the per-frame magnitude of the chosen centroid
#           averaged per condition, so it is robust on sparse sine signals.
#           Answers "do the perturbations push the encoder toward bigger or
#           smaller centroids?".
# ---------------------------------------------------------------------------

def _static_codebook_magnitudes(
    codebook_weights: list[np.ndarray],
) -> "list[np.ndarray]":
    """L2 magnitude of every centroid in every codebook → list of [codebook_size]."""
    return [np.linalg.norm(W, axis=1) for W in codebook_weights]


def _selected_centroid_magnitudes_per_codebook(
    tokens: np.ndarray,                  # [n_cb, n_frames] int indices
    codebook_weights: list[np.ndarray],  # list of [codebook_size, dim]
) -> "list[np.ndarray]":                 # list length n_cb of [n_frames] magnitudes
    """Per-codebook L2 magnitude of the centroid the encoder selected at each frame."""
    n_cb = min(tokens.shape[0], len(codebook_weights))
    out: list[np.ndarray] = []
    for cb in range(n_cb):
        W = codebook_weights[cb]
        out.append(np.linalg.norm(W[tokens[cb]], axis=1))
    return out


# ── Page 11: static codebook magnitude histograms ──────────────────────────

def _codebook_magnitude_hist_figure(
    codebook_weights: list[np.ndarray],
) -> plt.Figure:
    """4×8 per-codebook grid (codebooks 1–32) — histogram of the centroid magnitudes
    of all 1024 slots with a fitted Gaussian overlay and mean/std/var/skew plus a
    normality p-value annotated per subplot."""
    try:
        from scipy.stats import norm as _norm, normaltest as _normaltest, skew as _skew
        _HAS_SCIPY = True
    except Exception:
        _HAS_SCIPY = False

    mags = _static_codebook_magnitudes(codebook_weights)
    n_cb = len(mags)

    n_cols = 8; n_rows = max(4, (n_cb + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, max(12, n_rows * 3)))
    fig.suptitle(
        "Static Codebook — Centroid Magnitude Distribution per Codebook\n"
        "(all 1024 slots per codebook, model property, input-independent)",
        fontsize=12,
    )

    for cb in range(n_cb):
        row, col = divmod(cb, 8)
        ax = axes[row][col]
        m = mags[cb]
        m = m[np.isfinite(m)]
        if m.size == 0:
            ax.set_visible(False)
            continue

        ax.hist(m, bins=50, density=True, color="steelblue", alpha=0.7)
        mu  = float(np.mean(m))
        sd  = float(np.std(m))
        var = float(np.var(m))

        p = np.nan
        sk = np.nan
        if _HAS_SCIPY:
            if sd > 0:
                xs = np.linspace(float(m.min()), float(m.max()), 200)
                ax.plot(xs, _norm.pdf(xs, mu, sd), color="black",
                        linewidth=1.5, label="Gaussian fit")
            if m.size >= 8:
                try:
                    p = float(_normaltest(m).pvalue)
                except Exception:
                    p = np.nan
            try:
                sk = float(_skew(m))
            except Exception:
                sk = np.nan

        stats_text = (
            f"mean={mu:.2f}\n"
            f"std ={sd:.2f}\n"
            f"var ={var:.2f}\n"
            f"skew={sk:.2f}\n"
            f"p   ={p:.1e}"
        )
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                fontsize=6, va="top", ha="left", color="#333",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

        ax.set_title(_get_unit_label(cb, False), fontsize=9)
        ax.tick_params(labelsize=6)
        if col == 0:
            ax.set_ylabel("Density", fontsize=8)
        if row == 3:
            ax.set_xlabel("Centroid Magnitude (‖embedding‖₂)", fontsize=8)

    for cb in range(n_cb, 32):
        row, col = divmod(cb, 8)
        axes[row][col].set_visible(False)

    handles, lbls = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, lbls, loc="lower right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ── Page 12: cross-codebook magnitude summary ──────────────────────────────

def _codebook_magnitude_summary_figure(
    codebook_weights: list[np.ndarray],
    is_q2d2: bool = False,
) -> plt.Figure:
    """Two-panel summary across codebooks 1–32 or grid pairs 1–16:
      • top    — mean centroid magnitude ± 1 std (tests the early-vs-late hypothesis)
      • bottom — per-codebook/grid-pair skewness of the magnitude distribution."""
    try:
        from scipy.stats import skew as _skew
        _HAS_SCIPY = True
    except Exception:
        _HAS_SCIPY = False

    mags = _static_codebook_magnitudes(codebook_weights)
    n_cb = len(mags)
    cb_idx = np.arange(1, n_cb + 1)

    means = np.array([float(np.mean(m)) for m in mags])
    stds  = np.array([float(np.std(m))  for m in mags])
    skews = np.array(
        [float(_skew(m)) if _HAS_SCIPY else np.nan for m in mags]
    )

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(28, 12), sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )
    unit_range = f"Grid Pairs 1–{n_cb}" if is_q2d2 else f"Codebooks 1–{n_cb}"
    subtitle = "(Magnitude development across grid pairs)" if is_q2d2 else "(Magnitude development across RVQ stages)"
    fig.suptitle(
        f"Static Codebook — Centroid Magnitude Across {unit_range}\n"
        f"{subtitle}",
        fontsize=12,
    )

    ax_top.fill_between(cb_idx, means - stds, means + stds,
                        alpha=0.25, color="steelblue", label="±1 std")
    ax_top.plot(cb_idx, means, color="steelblue", linewidth=1.8,
                marker="o", markersize=4, label="mean magnitude")
    ax_top.set_ylabel("Mean Centroid Magnitude (‖embedding‖₂)", fontsize=9)
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(fontsize=8, loc="upper right")

    ax_bot.bar(cb_idx, skews, color="indianred", alpha=0.8)
    ax_bot.axhline(0.0, color="0.4", linewidth=0.8)
    ax_bot.set_ylabel("Skewness", fontsize=9)
    xlabel = "Grid Pair" if is_q2d2 else "Codebook (RVQ stage, 1 = coarsest)"
    ax_bot.set_xlabel(xlabel, fontsize=9)
    ax_bot.grid(True, alpha=0.25)

    ax_bot.set_xticks(cb_idx)
    ax_bot.set_xticklabels([str(i) for i in cb_idx], fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ── Pages 13–15: selected-centroid magnitude trend vs test condition ───────

def _per_codebook_amplitude_magnitude_figure(
    all_stats: list[dict],
    amp_tokens_root: Path,
    bw: str,
    codebook_weights: list[np.ndarray],
    is_q2d2: bool = False,
) -> plt.Figure:
    """4×8 per-codebook grid (codebooks 1–32 or grid pairs 1–16) — mean selected-centroid magnitude vs amplitude level."""
    if not all_stats or not codebook_weights:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return fig

    n_cb    = min(all_stats[0]["n_codebooks"], len(codebook_weights))
    n_freqs = len(all_stats)

    variant_vals_raw = all_stats[0]["variant_values"]
    x_labels  = ["0 dBFS"] + [
        AMP_TAG_TO_LABEL.get(int(v), f"−{int(v)} dBFS") for v in variant_vals_raw
    ]
    n_x = 1 + len(variant_vals_raw)

    data = np.full((n_x, n_freqs, n_cb), np.nan)
    for fi, stats in enumerate(all_stats):
        signal    = stats["signal"]
        sub       = amp_tokens_root / signal
        base_path = sub / f"{signal}_baseline_bw{bw}_tokens.npy"
        if base_path.exists():
            base_tok = np.load(str(base_path)).astype(np.int64)
            mags = _selected_centroid_magnitudes_per_codebook(base_tok, codebook_weights)
            for cb in range(len(mags)):
                data[0, fi, cb] = mags[cb].mean()
        for xi, tag in enumerate(stats["variant_tags"], start=1):
            var_path = sub / f"{signal}_var_{tag}_bw{bw}_tokens.npy"
            if not var_path.exists():
                continue
            var_tok = np.load(str(var_path)).astype(np.int64)
            mags = _selected_centroid_magnitudes_per_codebook(var_tok, codebook_weights)
            for cb in range(len(mags)):
                data[xi, fi, cb] = mags[cb].mean()

    mean_data = np.nanmean(data, axis=1)
    std_data  = np.nanstd(data,  axis=1)
    var_data  = np.nanvar(data,  axis=1)

    n_cols = 8; n_rows = max(4, (n_cb + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, max(12, n_rows * 3)), sharey=True)
    fig.suptitle(
        f"Amplitude Per Codebook — Mean Selected-Centroid Magnitude\n"
        f"(mean ± std across {n_freqs} frequencies, sine waves, bw={bw} kbps)",
        fontsize=12,
    )

    for cb in range(n_cb):
        row, col = divmod(cb, 8)
        ax = axes[row][col]
        y_mean = mean_data[:, cb]
        y_std  = std_data[:, cb]
        y_var  = var_data[:, cb]

        ax.fill_between(
            range(n_x), y_mean - y_std, y_mean + y_std,
            alpha=0.25, color="teal", label="±1 std"
        )
        ax.plot(range(n_x), y_mean, color="teal", linewidth=1.8,
                marker="o", markersize=4, label="mean")

        stats_text = (
            f"mean={np.nanmean(y_mean):.3f}\n"
            f"std ={np.nanmean(y_std):.3f}\n"
            f"var ={np.nanmean(y_var):.3f}"
        )
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                fontsize=6, va="top", ha="left", color="#333",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

        ax.set_title(_get_unit_label(cb, is_q2d2), fontsize=9)
        ax.set_xticks(range(n_x))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=6)
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("Mean Centroid Magnitude", fontsize=8)
        if row == 3:
            ax.set_xlabel("Amplitude (dBFS)", fontsize=8)

    for cb in range(n_cb, 32):
        row, col = divmod(cb, 8)
        axes[row][col].set_visible(False)

    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def _per_codebook_phase_magnitude_figure(
    stats_0dB: list[dict],
    phase_tokens_root: Path,
    bw: str,
    codebook_weights: list[np.ndarray],
    is_q2d2: bool = False,
) -> plt.Figure:
    """4×8 per-codebook grid (codebooks 1–32 or grid pairs 1–16) — mean selected-centroid magnitude vs phase offset."""
    if not stats_0dB or not codebook_weights:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no 0 dBFS data", ha="center", va="center")
        return fig

    n_cb    = min(stats_0dB[0]["n_codebooks"], len(codebook_weights))
    n_freqs = len(stats_0dB)
    phase_vals = stats_0dB[0]["variant_values"]
    phase_tags = stats_0dB[0]["variant_tags"]

    x = [0.0] + list(phase_vals)
    n_x_plot = len(x)

    data = np.full((n_x_plot, n_freqs, n_cb), np.nan)
    for fi, stats in enumerate(stats_0dB):
        signal    = stats["signal"]
        sub       = phase_tokens_root / signal
        base_path = sub / f"{signal}_baseline_bw{bw}_tokens.npy"
        if base_path.exists():
            base_tok = np.load(str(base_path)).astype(np.int64)
            mags = _selected_centroid_magnitudes_per_codebook(base_tok, codebook_weights)
            for cb in range(len(mags)):
                data[0, fi, cb] = mags[cb].mean()
        for xi, tag in enumerate(phase_tags, start=1):
            var_path = sub / f"{signal}_var_{tag}_bw{bw}_tokens.npy"
            if not var_path.exists():
                continue
            var_tok = np.load(str(var_path)).astype(np.int64)
            mags = _selected_centroid_magnitudes_per_codebook(var_tok, codebook_weights)
            for cb in range(len(mags)):
                data[xi, fi, cb] = mags[cb].mean()

    mean_data = np.nanmean(data, axis=1)
    std_data  = np.nanstd(data,  axis=1)
    var_data  = np.nanvar(data,  axis=1)

    n_cols = 8; n_rows = max(4, (n_cb + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 12), sharey=True)
    fig.suptitle(
        f"Phase Per Codebook — Mean Selected-Centroid Magnitude\n"
        f"(mean ± std across {n_freqs} frequencies at 0 dBFS, sine waves, bw={bw} kbps)",
        fontsize=12,
    )

    for cb in range(n_cb):
        row, col = divmod(cb, 8)
        ax = axes[row][col]
        y_mean = mean_data[:, cb]
        y_std  = std_data[:, cb]
        y_var  = var_data[:, cb]

        ax.fill_between(
            x, y_mean - y_std, y_mean + y_std,
            alpha=0.25, color="slateblue", label="±1 std"
        )
        ax.plot(x, y_mean, color="slateblue", linewidth=1.8,
                marker="o", markersize=3, label="mean")

        stats_text = (
            f"mean={np.nanmean(y_mean):.3f}\n"
            f"std ={np.nanmean(y_std):.3f}\n"
            f"var ={np.nanmean(y_var):.3f}"
        )
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                fontsize=6, va="top", ha="left", color="#333",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

        ax.set_title(_get_unit_label(cb, is_q2d2), fontsize=9)
        tick_x = [v for v in x if v == 0.0 or (int(v) % 90 == 0)]
        ax.set_xticks(tick_x)
        ax.set_xticklabels([f"{int(v)}°" for v in tick_x], fontsize=7)
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("Mean Centroid Magnitude", fontsize=8)
        if row == 3:
            ax.set_xlabel("Phase Offset (degrees)", fontsize=8)

    for cb in range(n_cb, 32):
        row, col = divmod(cb, 8)
        axes[row][col].set_visible(False)

    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def _per_codebook_temporal_magnitude_figure(
    all_freq_data: dict,
    temporal_tokens_root: Path,
    bw: str,
    codebook_weights: list[np.ndarray],
    is_q2d2: bool = False,
) -> plt.Figure:
    """4×8 per-codebook grid (codebooks 1–32 or grid pairs 1–16) — mean selected-centroid magnitude vs time offset."""
    from analyze_sine_temporal import ALL_OFFSETS_MS  # noqa: F401

    if not all_freq_data or not codebook_weights:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return fig

    freqs   = sorted(all_freq_data.keys())
    n_freqs = len(freqs)
    n_cb    = min(
        next(iter(next(iter(all_freq_data.values())).values())).shape[0],
        len(codebook_weights),
    )

    offset_sets    = [set(v.keys()) for v in all_freq_data.values() if v]
    common_offsets = sorted(
        offset_sets[0].intersection(*offset_sets[1:])
        if len(offset_sets) > 1 else offset_sets[0]
    ) if offset_sets else ALL_OFFSETS_MS

    display_offsets = [0] + list(common_offsets)
    n_x_plot = len(display_offsets)
    x_labels = [
        str(v) if v == 0 or (v <= 20 and v % 5 == 0) or v > 20 else ""
        for v in display_offsets
    ]

    data = np.full((n_x_plot, n_freqs, n_cb), np.nan)
    for fi, freq in enumerate(freqs):
        subdir    = temporal_tokens_root / f"{freq}hz"
        base_path = subdir / f"baseline_0ms_bw{bw}_tokens.npy"
        if base_path.exists():
            base_tok = np.load(str(base_path)).astype(np.int64)
            mags = _selected_centroid_magnitudes_per_codebook(base_tok, codebook_weights)
            for cb in range(len(mags)):
                data[0, fi, cb] = mags[cb].mean()
        for xi, offset_ms in enumerate(common_offsets, start=1):
            var_path = subdir / f"offset_{offset_ms:03d}ms_bw{bw}_tokens.npy"
            if not var_path.exists():
                continue
            var_tok = np.load(str(var_path)).astype(np.int64)
            mags = _selected_centroid_magnitudes_per_codebook(var_tok, codebook_weights)
            for cb in range(len(mags)):
                data[xi, fi, cb] = mags[cb].mean()

    mean_data = np.nanmean(data, axis=1)
    std_data  = np.nanstd(data,  axis=1)
    var_data  = np.nanvar(data,  axis=1)

    n_cols = 8; n_rows = max(4, (n_cb + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, max(12, n_rows * 3)), sharey=True)
    fig.suptitle(
        f"Time Delay Per Codebook — Mean Selected-Centroid Magnitude\n"
        f"(mean ± std across {n_freqs} sine frequencies, 0 dBFS, bw={bw} kbps)",
        fontsize=12,
    )

    for cb in range(n_cb):
        row, col = divmod(cb, 8)
        ax = axes[row][col]
        y_mean = mean_data[:, cb]
        y_std  = std_data[:, cb]
        y_var  = var_data[:, cb]

        ax.fill_between(
            range(n_x_plot), y_mean - y_std, y_mean + y_std,
            alpha=0.25, color="chocolate", label="±1 std"
        )
        ax.plot(range(n_x_plot), y_mean, color="chocolate", linewidth=1.8,
                marker="o", markersize=3, label="mean")

        stats_text = (
            f"mean={np.nanmean(y_mean):.3f}\n"
            f"std ={np.nanmean(y_std):.3f}\n"
            f"var ={np.nanmean(y_var):.3f}"
        )
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                fontsize=6, va="top", ha="left", color="#333",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

        ax.set_title(_get_unit_label(cb, is_q2d2), fontsize=9)
        ax.set_xticks(range(n_x_plot))
        ax.set_xticklabels(x_labels, fontsize=6)
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("Mean Centroid Magnitude", fontsize=8)
        if row == 3:
            ax.set_xlabel("Time Offset (ms)", fontsize=8)

    for cb in range(n_cb, 32):
        row, col = divmod(cb, 8)
        axes[row][col].set_visible(False)

    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# ---------------------------------------------------------------------------
# Title page
# ---------------------------------------------------------------------------

def _title_page(bw: str, n_amp_freqs: int, n_phase_freqs: int, n_temporal_freqs: int,
                model: str = "multi_dataset_encodec",
                checkpoint: str = "",
                sample_rate: int = 24_000,
                channels: int = 1) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(28, 12))
    ax.axis("off")

    checkpoint_display = checkpoint if checkpoint else "(default / not specified)"
    # Accept either a model name or a model_type string for the title page
    _m2t = {"q2d2": "q2d2", "hificodec": "hificodec",
            "speechtokenizer": "speechtokenizer", "dac_fsq": "dac_fsq"}
    _model_type_tp = _m2t.get(model, "encodec")
    is_q2d2 = (_model_type_tp == "q2d2")
    _unit_base = _MODEL_UNIT_LABELS.get(_model_type_tp, "Codebook")
    unit_name  = _unit_base.lower()
    unit_range = f"{_unit_base.lower()}s 1–N"

    lines = [
        f"Bandwidth: {bw} kbps",
        f"Generated: {date.today().isoformat()}",
        "",
        "─" * 60,
        "",
        "MODEL CONFIGURATION",
        f"  Model       : {model}",
        f"  Checkpoint  : {checkpoint_display}",
        f"  Sample rate : {sample_rate} Hz",
        f"  Channels    : {channels} (mono)",
        "",
        "─" * 60,
        "",
        "AMPLITUDE TEST",
        f"  Signal type : sine waves — 5s file, full 5s signal, 0 dBFS baseline",
        f"  Frequencies : {n_amp_freqs} log-spaced (10 Hz – 20 kHz)",
        f"  Levels      : 0, −20, −40, −60, −80, −100, −120, −140 dBFS  (8 levels)",
        f"  Total files : {n_amp_freqs} frequencies × 8 levels = {n_amp_freqs * 8}",
        "",
        "PHASE TEST",
        f"  Signal type : sine waves — 5s file, full 5s signal, 0 dBFS",
        f"  Frequencies : {n_phase_freqs} log-spaced (10 Hz – 20 kHz)",
        f"  Phase steps : 15°, 30°, …, 360°  (24 steps)",
        f"  Total files : {n_phase_freqs} frequencies × 24 steps = {n_phase_freqs * 24}",
        "",
        "TEMPORAL TEST",
        f"  Signal type : sine waves — 2s file, 1s signal zero-padded, 0 dBFS",
        f"  Frequencies : {n_temporal_freqs} log-spaced (10 Hz \u2013 20 kHz)",
        f"  Offsets     : 1\u201320 ms (every 1 ms) + 40/60/80/100 ms  (25 offsets)",
        f"  Total files : {n_temporal_freqs} \u00d7 26 files = {n_temporal_freqs * 26}",
        f"  Note        : The signal is placed at the defined offset into a 2s buffer.",
        f"                RMS is constant across all offsets.",
        "",
        "─" * 60,
        "",
        f"Pages 2–4  : per-{unit_name} flip rate ({unit_range}).",
        f"Pages 5–7  : per-{unit_name} mean L2 {unit_name} distance (requires --checkpoint).",
        "Each subplot shows mean ± 1 std across all frequencies.",
        "Variance is annotated in the stats box at the top-left of each subplot.",
        "The 0 dBFS / 0° / 0 ms point is the baseline (flip rate = 0, distance = 0).",
        "",
        "─" * 60,
        "",
        f"Perplexity pages (always generated, no --checkpoint required):",
        f"  Amplitude (page 16), Phase (page 17), Temporal (page 18).",
        f"  Perplexity = exp(H), H = token-distribution entropy per {unit_name}.",
        f"  Max = vocab_size (all tokens used equally); min ≈ 1 (fully collapsed).",
        f"  A falling perplexity under the test condition indicates early token",
        f"  saturation — the codec underutilises its dictionary for that content.",
    ]

    ax.text(
        0.08, 0.90,
        "\n".join(lines),
        ha="left", va="top",
        family="monospace",
        fontsize=10.5,
        transform=ax.transAxes,
        linespacing=1.5,
    )
    
    _TITLE_MAP = {
        "q2d2":            "Q2D2 Grid Pair Sensitivity — Sine Waves",
        "hificodec":       "HiFiCodec Codebook Sensitivity — Sine Waves",
        "speechtokenizer": "SpeechTokenizer RVQ Codebook Sensitivity — Sine Waves",
        "dac_fsq":         "DAC-FSQ Stream Sensitivity — Sine Waves",
    }
    title_text = _TITLE_MAP.get(_model_type_tp, "EnCodec Codebook Sensitivity — Sine Waves")
    ax.text(
        0.5, 0.99,
        title_text,
        ha="center", va="top",
        fontsize=16, fontweight="bold",
        transform=ax.transAxes,
    )
    return fig


# ---------------------------------------------------------------------------
# SVD(ΔZ) EVR figure builders (Pages 16-19)
# ---------------------------------------------------------------------------

def _svd_collect_sinusoid_deltas(
    tokens_root: Path,
    signal_glob: str,
    bw: str,
    weights: list[np.ndarray],
) -> dict[int, list[np.ndarray]]:
    """Collect per-unit delta mean-embedding vectors for sinusoid-style tests."""
    n_cb = len(weights)
    cb_deltas: dict[int, list[np.ndarray]] = {cb: [] for cb in range(n_cb)}
    for sig_dir in sorted(tokens_root.glob(signal_glob)):
        signal = sig_dir.name
        base_path = sig_dir / f"{signal}_baseline_bw{bw}_tokens.npy"
        if not base_path.exists():
            continue
        baseline = np.load(str(base_path))
        if baseline.ndim != 2:
            continue
        for vp in sorted(sig_dir.glob(f"{signal}_var_*_bw{bw}_tokens.npy")):
            variant = np.load(str(vp))
            if variant.ndim != 2:
                continue
            n_frames = min(baseline.shape[1], variant.shape[1])
            for cb in range(min(n_cb, baseline.shape[0], variant.shape[0])):
                W = weights[cb]
                cb_deltas[cb].append(
                    W[variant[cb, :n_frames]].mean(axis=0)
                    - W[baseline[cb, :n_frames]].mean(axis=0)
                )
    return cb_deltas


def _svd_collect_temporal_deltas(
    tokens_root: Path,
    bw: str,
    weights: list[np.ndarray],
) -> dict[int, list[np.ndarray]]:
    """Collect per-unit delta mean-embedding vectors for the time-sine test."""
    n_cb = len(weights)
    cb_deltas: dict[int, list[np.ndarray]] = {cb: [] for cb in range(n_cb)}
    for freq_dir in sorted([d for d in tokens_root.iterdir() if d.is_dir()]):
        base_path = freq_dir / f"baseline_0ms_bw{bw}_tokens.npy"
        if not base_path.exists():
            continue
        baseline = np.load(str(base_path))
        if baseline.ndim != 2:
            continue
        for vp in sorted(freq_dir.glob(f"offset_*ms_bw{bw}_tokens.npy")):
            variant = np.load(str(vp))
            if variant.ndim != 2:
                continue
            n_frames = min(baseline.shape[1], variant.shape[1])
            for cb in range(min(n_cb, baseline.shape[0], variant.shape[0])):
                W = weights[cb]
                cb_deltas[cb].append(
                    W[variant[cb, :n_frames]].mean(axis=0)
                    - W[baseline[cb, :n_frames]].mean(axis=0)
                )
    return cb_deltas


def _svd_metrics_from_deltas(
    cb_deltas: dict[int, list[np.ndarray]],
    n_cb: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (evr2, evr1, eff_rank) each shape [n_cb]."""
    evr2 = np.full(n_cb, np.nan)
    evr1 = np.full(n_cb, np.nan)
    er   = np.full(n_cb, np.nan)
    for cb in range(n_cb):
        deltas = cb_deltas.get(cb, [])
        if len(deltas) < 2:
            continue
        _, s, _ = np.linalg.svd(np.stack(deltas), full_matrices=False)
        s2 = s ** 2
        total = s2.sum()
        if total < 1e-12:
            evr2[cb] = evr1[cb] = 0.0; er[cb] = 1.0
            continue
        p = s2 / total
        evr2[cb] = float(s2[:2].sum() / total)
        evr1[cb] = float(s2[0] / total)
        er[cb]   = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    return evr2, evr1, er


def _svd_evr_figure(
    evr_vals: np.ndarray,       # [n_cb] primary EVR (EVR₂ for phase, EVR₁ for amp/temporal)
    eff_rank: np.ndarray,       # [n_cb]
    title: str,
    evr_label: str,
    color: str,
    unit_name: str = "Codebook",
) -> plt.Figure:
    """Single-panel EVR bar/line chart across codebook units."""
    n_cb = len(evr_vals)
    xs   = np.arange(1, n_cb + 1)
    fig, ax1 = plt.subplots(figsize=(max(8, n_cb * 0.35 + 2), 5))

    valid_mask = ~np.isnan(evr_vals)
    ax1.bar(xs[valid_mask], evr_vals[valid_mask] * 100,
            color=color, alpha=0.75, label=evr_label)
    ax1.set_ylim(0, 105)
    ax1.set_ylabel(f"{evr_label} (%)", color=color, fontsize=11)
    ax1.tick_params(axis="y", labelcolor=color)

    # EffRank on secondary axis
    ax2 = ax1.twinx()
    er_valid = ~np.isnan(eff_rank)
    ax2.plot(xs[er_valid], eff_rank[er_valid], color="dimgray",
             marker=".", linestyle="--", linewidth=1.2, label="EffRank")
    ax2.set_ylabel("Effective Rank", color="dimgray", fontsize=10)
    ax2.tick_params(axis="y", labelcolor="dimgray")

    ax1.set_xlabel(unit_name, fontsize=11)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(xs, fontsize=7)
    ax1.set_title(title, fontsize=13, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _svd_amplitude_figure(
    amp_tokens_root: Path,
    bw: str,
    codebook_weights: list[np.ndarray],
    is_q2d2: bool = False,
) -> plt.Figure:
    """Page 16: EVR₁ per unit for amplitude perturbation (monotonic = 1-D ideal)."""
    unit_name = "Grid Pair" if is_q2d2 else "Codebook"
    cb_deltas = _svd_collect_sinusoid_deltas(amp_tokens_root, "self_amp_*", bw, codebook_weights)
    evr2, evr1, er = _svd_metrics_from_deltas(cb_deltas, len(codebook_weights))
    return _svd_evr_figure(
        evr_vals=evr1,
        eff_rank=er,
        title=f"SVD(\u0394Z) Amplitude — EVR\u2081 per {unit_name}\n"
              f"(top-1 SV / total variance; ideal \u2248 1 for monotonic response)",
        evr_label="EVR\u2081 Amplitude",
        color="steelblue",
        unit_name=unit_name,
    )


def _svd_phase_evr2_figure(
    phase_tokens_root: Path,
    bw: str,
    codebook_weights: list[np.ndarray],
    is_q2d2: bool = False,
) -> plt.Figure:
    """Page 17: EVR₂ + EffRank per unit for phase perturbation (circle = 2-D ideal)."""
    unit_name = "Grid Pair" if is_q2d2 else "Codebook"
    # 0 dBFS only — pure phase variation without amplitude noise
    cb_deltas = _svd_collect_sinusoid_deltas(phase_tokens_root, "self_phase_0dB_*", bw, codebook_weights)
    evr2, evr1, er = _svd_metrics_from_deltas(cb_deltas, len(codebook_weights))
    return _svd_evr_figure(
        evr_vals=evr2,
        eff_rank=er,
        title=f"SVD(\u0394Z) Phase — EVR\u2082 per {unit_name}\n"
              f"(top-2 SVs / total variance; ideal \u2248 1 for circular manifold, EffRank \u2248 2)",
        evr_label="EVR\u2082 Phase",
        color="darkorchid",
        unit_name=unit_name,
    )


def _svd_temporal_figure(
    temporal_tokens_root: Path,
    bw: str,
    codebook_weights: list[np.ndarray],
    is_q2d2: bool = False,
) -> plt.Figure:
    """Page 18: EVR₁ per unit for temporal perturbation (monotonic = 1-D ideal)."""
    unit_name = "Grid Pair" if is_q2d2 else "Codebook"
    cb_deltas = _svd_collect_temporal_deltas(temporal_tokens_root, bw, codebook_weights)
    evr2, evr1, er = _svd_metrics_from_deltas(cb_deltas, len(codebook_weights))
    return _svd_evr_figure(
        evr_vals=evr1,
        eff_rank=er,
        title=f"SVD(\u0394Z) Temporal — EVR\u2081 per {unit_name}\n"
              f"(top-1 SV / total variance; ideal \u2248 1 for monotonic offset response)",
        evr_label="EVR\u2081 Temporal",
        color="darkorange",
        unit_name=unit_name,
    )


def _svd_pca_scatter_figure(
    phase_tokens_root: Path,
    bw: str,
    codebook_weights: list[np.ndarray],
    is_q2d2: bool = False,
) -> plt.Figure:
    """Page 19: PC1 vs PC2 scatter of phase ΔZ at representative depths.

    Points are coloured by phase angle to reveal whether the circular manifold
    is preserved in the codebook's embedding space.
    """
    n_cb = len(codebook_weights)
    # Representative depths: first, ~quarter, ~half, last
    indices = sorted({0, n_cb // 4, n_cb // 2, n_cb - 1})
    unit_name = "Grid Pair" if is_q2d2 else "Codebook"

    # Build per-variant delta vectors (keyed by phase tag) for each signal dir
    # Each element: (phase_tag_float, delta[cb]) so we can colour by angle
    tagged_deltas: dict[int, list[tuple[float, np.ndarray]]] = {cb: [] for cb in indices}

    import re as _re
    _phase_re = _re.compile(r"_var_([^_]+)_bw")
    for sig_dir in sorted(phase_tokens_root.glob("self_phase_0dB_*")):
        signal = sig_dir.name
        base_path = sig_dir / f"{signal}_baseline_bw{bw}_tokens.npy"
        if not base_path.exists():
            continue
        baseline = np.load(str(base_path))
        if baseline.ndim != 2:
            continue
        for vp in sorted(sig_dir.glob(f"{signal}_var_*_bw{bw}_tokens.npy")):
            m = _phase_re.search(vp.name)
            if not m:
                continue
            try:
                angle = float(m.group(1))
            except ValueError:
                continue
            variant = np.load(str(vp))
            if variant.ndim != 2:
                continue
            n_frames = min(baseline.shape[1], variant.shape[1])
            for cb in indices:
                if cb >= baseline.shape[0]:
                    continue
                W = codebook_weights[cb]
                delta = (W[variant[cb, :n_frames]].mean(axis=0)
                         - W[baseline[cb, :n_frames]].mean(axis=0))
                tagged_deltas[cb].append((angle, delta))

    n_plots = len(indices)
    # Colorbar on the left (narrowed) + square PCA subplots
    fig = plt.figure(figsize=(28, 8))
    gs = fig.add_gridspec(1, n_plots + 1, width_ratios=[0.2] + [1] * n_plots,
                          wspace=0.35)
    cmap = plt.cm.hsv

    # Reserve colorbar axis on the far left
    cbar_ax = fig.add_subplot(gs[0, 0])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 360))
    sm.set_array([])
    cb_bar = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cb_bar.set_label("Phase angle (°)", fontsize=10)

    axes = [fig.add_subplot(gs[0, i + 1]) for i in range(n_plots)]

    for ax, cb in zip(axes, indices):
        items = tagged_deltas[cb]
        ax.set_box_aspect(1)
        if len(items) < 2:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{unit_name} {cb + 1}", fontsize=11, pad=14)
            continue

        angles  = np.array([a for a, _ in items])
        mat     = np.stack([d for _, d in items])   # [n_variants, dim]
        _, s, Vt = np.linalg.svd(mat - mat.mean(axis=0), full_matrices=False)
        pc1 = mat @ Vt[0]
        pc2 = mat @ Vt[1] if Vt.shape[0] > 1 else np.zeros(len(items))

        colours = cmap(angles / 360.0)
        ax.scatter(pc1, pc2, c=colours, s=28, alpha=0.7)
        s2 = s ** 2
        total = s2.sum()
        evr2_val = float(s2[:2].sum() / total) if total > 0 else 0.0
        ax.set_title(f"{unit_name} {cb + 1}\nEVR₂={evr2_val:.2f}", fontsize=11, pad=14)
        ax.set_xlabel("PC 1", fontsize=10)
        ax.set_ylabel("PC 2", fontsize=10)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_aspect("equal", "datalim")

    fig.suptitle(
        f"PCA of Phase \u0394Z — PC1 vs PC2 at representative depths\n"
        f"(colour = phase angle; ideal circle → points form a ring)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    _ds = _PROJ_ROOT / "datasets"
    ap.add_argument("--amp-tokens",
                    default=str(_ds / "audio_tokens" / "dsp_self_amp"),
                    help="Tokens root for amplitude test (dsp_self_amp)")
    ap.add_argument("--phase-tokens",
                    default=str(_ds / "audio_tokens" / "dsp_self_phase"),
                    help="Tokens root for phase test (dsp_self_phase)")
    ap.add_argument("--temporal-tokens",
                    default=str(_ds / "audio_tokens" / "time_sine"),
                    help="Tokens root for temporal sine test (time_sine)")
    ap.add_argument("--bandwidth", default="24.0",
                    help="EnCodec bandwidth string, e.g. 24.0")
    ap.add_argument("--model", default="multi_dataset_encodec",
                    help="Model name: multi_dataset_encodec, q2d2, hificodec, speechtokenizer, dac_fsq")
    ap.add_argument("--model-type", default="",
                    help="Explicit model type override: encodec|q2d2|hificodec|speechtokenizer|dac_fsq")
    ap.add_argument("--checkpoint", default="",
                    help="Checkpoint .pt path — required for distance pages (5–7)")
    ap.add_argument("--sample-rate", type=int, default=24_000,
                    help="Sample rate of test audio (Hz)")
    ap.add_argument("--output",
                    default=str(_ds / "analysis" / "combined_sensitivity_report_bw{bw}.pdf"),
                    help="Output PDF path (use {bw} as bandwidth placeholder)")
    args = ap.parse_args()

    bw = args.bandwidth
    output_path = Path(args.output.replace("{bw}", bw))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Amplitude ────────────────────────────────────────────────────────────
    print("\n[1/3] Loading amplitude test tokens …")
    amp_tokens_root = Path(args.amp_tokens)
    amp_all_stats: list[dict] = []
    for signal in AMP_SIGNALS:
        sub = amp_tokens_root / signal
        if not sub.is_dir():
            continue
        # Use a temp out_dir under the tokens root for stats JSONs
        tmp_out = amp_tokens_root.parent.parent / "analysis" / "dsp_self_amp" / signal
        stats = amp_analyze_signal(signal, sub, tmp_out, bw)
        if stats is not None:
            amp_all_stats.append(stats)
    print(f"  {len(amp_all_stats)} amplitude signals loaded")

    # ── Phase ─────────────────────────────────────────────────────────────────
    print("\n[2/3] Loading phase test tokens …")
    phase_tokens_root = Path(args.phase_tokens)
    phase_all_stats: list[dict] = []
    for signal in PHASE_SIGNALS:
        sub = phase_tokens_root / signal
        if not sub.is_dir():
            continue
        tmp_out = phase_tokens_root.parent.parent / "analysis" / "dsp_self_phase" / signal
        stats = phase_analyze_signal(signal, sub, tmp_out, bw)
        if stats is not None:
            phase_all_stats.append(stats)
    phase_0dB = [s for s in phase_all_stats if s["amp_label"] == "0dB"]
    print(f"  {len(phase_all_stats)} phase signals loaded ({len(phase_0dB)} at 0 dBFS)")

    # ── Temporal ─────────────────────────────────────────────────────────────
    print("\n[3/3] Loading temporal sine tokens …")
    temporal_tokens_root = Path(args.temporal_tokens)
    temporal_freq_data: dict[int, dict] = {}
    for freq in TEMPORAL_FREQ_TAGS:
        subdir = temporal_tokens_root / f"{freq}hz"
        if not subdir.is_dir():
            continue
        result = temporal_analyze_freq(freq, subdir, bw)
        if result is not None:
            temporal_freq_data[freq] = result
    print(f"  {len(temporal_freq_data)} temporal frequencies loaded")

    # ── Codebook weights (for distance pages 5–7) ─────────────────────────
    print("\n[4/4] Extracting codebook weights …")
    codebook_weights = _load_codebook_weights(args.checkpoint, args.model)
    random_baselines: np.ndarray | None = None
    if codebook_weights is not None:
        random_baselines = _random_baseline_distances(codebook_weights)
        print(f"  Random-pair baseline distances (first 4 codebooks): "
              f"{random_baselines[:4].round(4)}")
    else:
        print("  [SKIP] No codebook weights — pages 5–7 will be omitted")

    # ── Build PDF ─────────────────────────────────────────────────────────────
    # Derive model_type from --model-type override or --model name
    _model_name_to_type = {
        "q2d2":                  "q2d2",
        "hificodec":             "hificodec",
        "speechtokenizer":       "speechtokenizer",
        "dac_fsq":               "dac_fsq",
        "multi_dataset_encodec": "encodec",
        "encodec_24khz":         "encodec",
        "encodec_48khz":         "encodec",
        "my_encodec":            "encodec",
        "encodec_bw":            "encodec",
    }
    model_type: str = args.model_type or _model_name_to_type.get(args.model, "encodec")
    is_q2d2 = _derive_is_q2d2(model_type)
    print(f"  Model type : {model_type}")
    print(f"\nBuilding combined PDF → {output_path}")
    with PdfPages(str(output_path)) as pdf:
        # Page 1: title
        fig = _title_page(
            bw,
            n_amp_freqs=len(amp_all_stats),
            n_phase_freqs=len(phase_0dB),
            n_temporal_freqs=len(temporal_freq_data),
            model=args.model,
            checkpoint=args.checkpoint,
            sample_rate=args.sample_rate,
            channels=1,
        )
        pdf.savefig(fig)
        plt.close(fig)
        print("  [1/4] Title page")

        # Page 2: amplitude per-codebook
        fig = _per_codebook_amplitude_figure(amp_all_stats, bw, is_q2d2=is_q2d2)
        pdf.savefig(fig)
        plt.close(fig)
        print("  [2/4] Amplitude per-codebook grid")

        # Page 3: phase per-codebook (0 dBFS)
        fig = _per_codebook_phase_figure(phase_0dB, bw, is_q2d2=is_q2d2)
        pdf.savefig(fig)
        plt.close(fig)
        print("  [3/4] Phase per-codebook grid")

        # Page 4: temporal per-codebook (flip rate)
        fig = _per_codebook_temporal_figure(temporal_freq_data, bw, is_q2d2=is_q2d2)
        pdf.savefig(fig)
        plt.close(fig)
        print("  [4/10] Temporal per-codebook flip rate grid")

        # ── Distance pages (require checkpoint) ───────────────────────────
        # NOTE: Skip for Q2D2 — tokens are flattened multi-pair indices that need decoding
        if codebook_weights is not None and random_baselines is not None:
            # Page 5: amplitude codebook distances
            fig = _per_codebook_amplitude_distance_figure(
                amp_all_stats, amp_tokens_root, bw, codebook_weights, random_baselines, is_q2d2=is_q2d2
            )
            pdf.savefig(fig)
            plt.close(fig)
            print("  [5/10] Amplitude per-codebook L2 distance grid")

            # Page 6: phase codebook distances
            fig = _per_codebook_phase_distance_figure(
                phase_0dB, phase_tokens_root, bw, codebook_weights, random_baselines, is_q2d2=is_q2d2
            )
            pdf.savefig(fig)
            plt.close(fig)
            print("  [6/10] Phase per-codebook L2 distance grid")

            # Page 7: temporal codebook distances
            fig = _per_codebook_temporal_distance_figure(
                temporal_freq_data, temporal_tokens_root, bw, codebook_weights, random_baselines, is_q2d2=is_q2d2
            )
            pdf.savefig(fig)
            plt.close(fig)
            print("  [7/10] Temporal per-codebook L2 distance grid")

            # Page 8: amplitude cosine similarity
            fig = _per_codebook_amplitude_cosine_figure(
                amp_all_stats, amp_tokens_root, bw, codebook_weights, random_baselines, is_q2d2=is_q2d2
            )
            pdf.savefig(fig)
            plt.close(fig)
            print("  [8/10] Amplitude per-codebook cosine similarity grid")

            # Page 9: phase cosine similarity
            fig = _per_codebook_phase_cosine_figure(
                phase_0dB, phase_tokens_root, bw, codebook_weights, random_baselines, is_q2d2=is_q2d2
            )
            pdf.savefig(fig)
            plt.close(fig)
            print("  [9/10] Phase per-codebook cosine similarity grid")

            # Page 10: temporal cosine similarity
            fig = _per_codebook_temporal_cosine_figure(
                temporal_freq_data, temporal_tokens_root, bw, codebook_weights, random_baselines, is_q2d2=is_q2d2
            )
            pdf.savefig(fig)
            plt.close(fig)
            print("  [10/15] Temporal per-codebook cosine similarity grid")

            # Page 11: static codebook magnitude histograms
            fig = _codebook_magnitude_hist_figure(codebook_weights)
            pdf.savefig(fig)
            plt.close(fig)
            print("  [11/15] Static codebook magnitude histogram grid")

            # Page 12: cross-codebook magnitude summary
            fig = _codebook_magnitude_summary_figure(codebook_weights, is_q2d2=is_q2d2)
            pdf.savefig(fig)
            plt.close(fig)
            print("  [12/15] Cross-codebook magnitude summary")

            # Page 13: amplitude selected-centroid magnitude trend
            fig = _per_codebook_amplitude_magnitude_figure(
                amp_all_stats, amp_tokens_root, bw, codebook_weights, is_q2d2=is_q2d2
            )
            pdf.savefig(fig)
            plt.close(fig)
            print("  [13/15] Amplitude per-codebook selected-centroid magnitude grid")

            # Page 14: phase selected-centroid magnitude trend
            fig = _per_codebook_phase_magnitude_figure(
                phase_0dB, phase_tokens_root, bw, codebook_weights, is_q2d2=is_q2d2
            )
            pdf.savefig(fig)
            plt.close(fig)
            print("  [14/15] Phase per-codebook selected-centroid magnitude grid")

            # Page 15: temporal selected-centroid magnitude trend
            fig = _per_codebook_temporal_magnitude_figure(
                temporal_freq_data, temporal_tokens_root, bw, codebook_weights, is_q2d2=is_q2d2
            )
            pdf.savefig(fig)
            plt.close(fig)
            print("  [15/19] Temporal per-codebook selected-centroid magnitude grid")

            n_pages = 15
        else:
            if is_q2d2:
                print("\n  [INFO] Pages 5-15 skipped for Q2D2 — token decoding from flattened indices not yet implemented")
                print("  Flip rate pages (1-4) and SVD pages (16-19) are still available")
            n_pages = 4

        # ── Perplexity pages (always generated, no checkpoint needed) ─────────
        vocab_size = 1024  # EnCodec / Q2D2 default codebook size
        print("\n[Perplexity] Building per-codebook perplexity pages …")

        try:
            fig = _per_codebook_amplitude_perplexity_figure(
                amp_all_stats, amp_tokens_root, bw,
                vocab_size=vocab_size, is_q2d2=is_q2d2,
            )
            pdf.savefig(fig)
            plt.close(fig)
            n_pages += 1
            print(f"  [{n_pages}] Amplitude per-codebook perplexity grid")
        except Exception as _e:
            print(f"  [SKIP] Amplitude perplexity page: {_e}")

        try:
            fig = _per_codebook_phase_perplexity_figure(
                phase_0dB, phase_tokens_root, bw,
                vocab_size=vocab_size, is_q2d2=is_q2d2,
            )
            pdf.savefig(fig)
            plt.close(fig)
            n_pages += 1
            print(f"  [{n_pages}] Phase per-codebook perplexity grid")
        except Exception as _e:
            print(f"  [SKIP] Phase perplexity page: {_e}")

        try:
            fig = _per_codebook_temporal_perplexity_figure(
                temporal_freq_data, temporal_tokens_root, bw,
                vocab_size=vocab_size, is_q2d2=is_q2d2,
            )
            pdf.savefig(fig)
            plt.close(fig)
            n_pages += 1
            print(f"  [{n_pages}] Temporal per-codebook perplexity grid")
        except Exception as _e:
            print(f"  [SKIP] Temporal perplexity page: {_e}")

        try:
            fig = _perplexity_summary_table_figure(
                amp_all_stats=amp_all_stats,
                phase_0dB=phase_0dB,
                temporal_freq_data=temporal_freq_data,
                amp_tokens_root=amp_tokens_root,
                phase_tokens_root=phase_tokens_root,
                temporal_tokens_root=temporal_tokens_root,
                bw=bw,
                vocab_size=vocab_size,
                is_q2d2=is_q2d2,
                model_type=model_type,
            )
            pdf.savefig(fig)
            plt.close(fig)
            n_pages += 1
            print(f"  [{n_pages}] Per-unit perplexity summary table")
        except Exception as _e:
            print(f"  [SKIP] Perplexity summary table page: {_e}")

        # ── SVD(ΔZ) topology pages (16-19) — require checkpoint for weights ──
        if codebook_weights is not None:
            try:
                fig = _svd_amplitude_figure(
                    amp_tokens_root, bw, codebook_weights, is_q2d2=is_q2d2)
                pdf.savefig(fig); plt.close(fig)
                n_pages += 1
                print(f"  [{n_pages}] SVD(ΔZ) amplitude EVR₁ per unit")
            except Exception as _e:
                print(f"  [SKIP] SVD amplitude page: {_e}")

            try:
                fig = _svd_phase_evr2_figure(
                    phase_tokens_root, bw, codebook_weights, is_q2d2=is_q2d2)
                pdf.savefig(fig); plt.close(fig)
                n_pages += 1
                print(f"  [{n_pages}] SVD(ΔZ) phase EVR₂ per unit")
            except Exception as _e:
                print(f"  [SKIP] SVD phase page: {_e}")

            try:
                fig = _svd_temporal_figure(
                    temporal_tokens_root, bw, codebook_weights, is_q2d2=is_q2d2)
                pdf.savefig(fig); plt.close(fig)
                n_pages += 1
                print(f"  [{n_pages}] SVD(ΔZ) temporal EVR₁ per unit")
            except Exception as _e:
                print(f"  [SKIP] SVD temporal page: {_e}")

            try:
                fig = _svd_pca_scatter_figure(
                    phase_tokens_root, bw, codebook_weights, is_q2d2=is_q2d2)
                pdf.savefig(fig); plt.close(fig)
                n_pages += 1
                print(f"  [{n_pages}] SVD(ΔZ) phase PCA scatter")
            except Exception as _e:
                print(f"  [SKIP] SVD PCA scatter page: {_e}")
        else:
            print("\n  [INFO] No checkpoint weights — SVD pages 16-19 skipped")

    print(f"\nDone — {n_pages} pages written to {output_path}")


if __name__ == "__main__":
    main()

"""Publication figures for codec sensitivity analysis — 6 figures, 1 PDF.

Figure 1  — Phase sensitivity per codec at depths 1, 4, 8
            5 rows (codecs) × up to 3 cols (depth columns)
            x = phase degrees, y = normalised [0–1]
            TFR ○, cos-sim △ (+ variance shadow), rel-L₂ □

Figure 2  — EGFX response per codec per effect category
            5 rows (codecs) × 3 cols (modulation, time_based, distortion)
            x = quantisation depth, y = normalised [0–1]
            Same 3 metric lines as Figure 1

Figure 3  — Mean centroid magnitude vs depth
            Single plot, 5 codec lines (distinct markers)
            y normalised 0–1 per codec

Figure 4  — SVD EVR₂ (phase) vs depth
            Dual y-axis: left = EVR₂ (%), right = effective rank
            5 codec lines with distinct markers

Figure 5  — EnCodec PC1 vs PC2 scatter at representative depths
            Coloured by phase angle; narrow left colorbar (1/8 subplot width)

Figure 6  — Q2D2 PC1 vs PC2 scatter at representative depths
            Same layout as Figure 5

Companion CSVs are written to --csv-dir.

Usage (single command):
    cd /path/to/msc_proj
    python tok_analysis/6_fig_summary_plot.py

Override defaults:
    python tok_analysis/6_fig_summary_plot.py \\
        --encodec-ckpt Encodec/outputs/.../best_model.pt \\
        --q2d2-ckpt    Q2D2/outputs/.../Q2D2_best.ckpt \\
        --hificodec-ckpt hificodec/egs/hificodec_fsd50k \\
        --st-ckpt      results/speechtokenizer_fsd50k/SpeechTokenizer_best_dev.pt \\
        --dac-ckpt     descript-audio-codec/ckpt/fsd50k_fsq/best \\
        --output       results/paper_figures.pdf
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

_HERE = Path(__file__).resolve().parent
_PROJ = _HERE.parent

# ── path setup so existing modules are importable ──────────────────────────
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_PROJ / "Encodec"))
sys.path.insert(0, str(_PROJ / "hificodec"))
sys.path.insert(0, str(_PROJ / "SpeechTokenizer"))

from analyze_self_phase_test import (   # noqa: E402
    FREQ_TAGS  as PHASE_FREQ_TAGS,
    PHASE_TAGS,
)
from report_combined_sensitivity import (   # noqa: E402
    _load_codebook_weights,
    _svd_collect_sinusoid_deltas,
    _svd_metrics_from_deltas,
)

# ---------------------------------------------------------------------------
# IBM accessible colour palette
# ---------------------------------------------------------------------------
_IBM_BLUE       = "#0f62fe"   # Blue 60
_IBM_TEAL       = "#009d9a"   # Teal 50
_IBM_WARM       = "#726e6a"   # Warm Gray 60  (brown-grey)
_IBM_GRAY       = "#525252"   # Gray 70
_IBM_NEAR_BLACK = "#21272a"   # Cool Gray 90

# Metric colours used in Fig 1 & 2 (same on every subplot)
_METRIC_COLOR = {"tfr": _IBM_BLUE, "cosim": _IBM_TEAL, "rell2": _IBM_WARM}
_METRIC_MARKER = {"tfr": "o", "cosim": "^", "rell2": "s"}

# Codec display properties (colour + marker used in Fig 3 & 4)
_CODEC_STYLE: dict[str, dict] = {
    "encodec":         {"label": "EnCodec",        "color": _IBM_BLUE,       "marker": "o"},
    "q2d2":            {"label": "Q2D2",            "color": _IBM_TEAL,       "marker": "s"},
    "speechtokenizer": {"label": "SpeechTokenizer", "color": _IBM_GRAY,       "marker": "^"},
    "hificodec":       {"label": "HiFiCodec",       "color": _IBM_WARM,       "marker": "D"},
    "dac_fsq":         {"label": "DAC-FSQ",         "color": _IBM_NEAR_BLACK, "marker": "X"},
}

# Row order for multi-codec figures
_CODEC_ORDER = ["encodec", "q2d2", "speechtokenizer", "hificodec", "dac_fsq"]

# Bandwidth tags (used in token filename patterns)
_BW_TAG: dict[str, str] = {
    "encodec": "24.0", "q2d2": "9.8",
    "hificodec": "HFC", "speechtokenizer": "ST", "dac_fsq": "FSQ",
}

# model_name passed to _load_codebook_weights
_MODEL_NAME: dict[str, str] = {
    "encodec": "multi_dataset_encodec",
    "q2d2": "q2d2", "hificodec": "hificodec",
    "speechtokenizer": "speechtokenizer", "dac_fsq": "dac_fsq",
}

# Token subdir prefix (empty = flat layout used by encodec/q2d2)
_TOKEN_SUBDIR: dict[str, str] = {
    "encodec": "", "q2d2": "",
    "hificodec": "hificodec", "speechtokenizer": "speechtokenizer",
    "dac_fsq": "dac_fsq",
}

# Depth columns shown in Figure 1 (1-indexed)
_FIG1_DEPTHS = [1, 4, 8]

# EGFX category order for Figure 2
_EGFX_CATS = ["modulation", "time_based", "distortion"]
_EGFX_CAT_LABEL = {
    "modulation": "Modulation", "time_based": "Time-Based", "distortion": "Distortion"
}

# ---------------------------------------------------------------------------
# Token path helpers
# ---------------------------------------------------------------------------

def _phase_root(tokens_root: Path, codec: str) -> Path:
    sub = _TOKEN_SUBDIR.get(codec, codec)
    base = tokens_root / sub if sub else tokens_root
    return base / "dsp_self_phase"


def _egfx_root(tokens_root: Path, codec: str) -> Path:
    return tokens_root / "egfx" / codec

# ---------------------------------------------------------------------------
# Metric computation — Relative L2 and cosine similarity at flip sites
# ---------------------------------------------------------------------------

def _load_tok(p: Path) -> "np.ndarray | None":
    if not p.exists():
        return None
    try:
        arr = np.load(str(p))
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return arr
    except Exception:
        return None


def _rel_l2_at_flips(
    base: np.ndarray,            # [n_cb, T]  int token indices
    var:  np.ndarray,            # [n_cb, T]
    weights: list[np.ndarray],   # list of [vocab, dim]
) -> np.ndarray:                 # [n_cb]  mean Rel-L2 at flip sites
    """Mean ||z_var - z_base||₂ at flip sites, normalised by codebook mean magnitude."""
    n_cb = min(base.shape[0], len(weights))
    n    = min(base.shape[1], var.shape[1])
    out  = np.full(n_cb, np.nan)
    for cb in range(n_cb):
        mask = base[cb, :n] != var[cb, :n]
        if not mask.any():
            out[cb] = 0.0
            continue
        W        = weights[cb]
        zb       = W[base[cb, :n][mask].astype(np.intp)]
        zv       = W[var[cb,  :n][mask].astype(np.intp)]
        # constant per-codebook normaliser avoids bias from boundary-token magnitudes
        mag_norm = max(float(np.mean(np.linalg.norm(W, axis=1))), 1e-12)
        out[cb]  = float(np.linalg.norm(zv - zb, axis=1).mean() / mag_norm)
    return out


def _cosim_at_flips(
    base: np.ndarray,
    var:  np.ndarray,
    weights: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:  # (mean_cosim [n_cb], var_cosim [n_cb])
    """Mean cosine similarity and its variance at flip sites, per codebook."""
    n_cb = min(base.shape[0], len(weights))
    n    = min(base.shape[1], var.shape[1])
    mean_out = np.full(n_cb, np.nan)
    var_out  = np.full(n_cb, 0.0)
    for cb in range(n_cb):
        mask = base[cb, :n] != var[cb, :n]
        if not mask.any():
            mean_out[cb] = 1.0
            continue
        W  = weights[cb]
        zb = W[base[cb, :n][mask].astype(np.intp)]
        zv = W[var[cb,  :n][mask].astype(np.intp)]
        nb = np.linalg.norm(zb, axis=1, keepdims=True)
        nv = np.linalg.norm(zv, axis=1, keepdims=True)
        nb = np.where(nb < 1e-12, 1e-12, nb)
        nv = np.where(nv < 1e-12, 1e-12, nv)
        sims = np.clip((zb / nb * zv / nv).sum(axis=1), -1.0, 1.0)
        mean_out[cb] = float(sims.mean())
        var_out[cb]  = float(sims.var())
    return mean_out, var_out

# ---------------------------------------------------------------------------
# Phase data loader — per codec, averaged across 0-dBFS sine frequencies
# ---------------------------------------------------------------------------

def _load_phase_metrics(
    codec: str,
    tokens_root: Path,
    weights: list[np.ndarray],
) -> dict:
    """Return phase-sweep metrics averaged across all 0-dBFS sine frequencies.

    Keys: phases, n_cb, tfr [n_p, n_cb], cosim [n_p, n_cb],
          cosim_var [n_p, n_cb], rell2 [n_p, n_cb]  (rell2 is normalised 0-1).
    """
    bw   = _BW_TAG[codec]
    root = _phase_root(tokens_root, codec)
    n_cb = len(weights)

    phase_vals  = [float(p) for p in PHASE_TAGS]  # 15 … 360
    n_phases    = len(phase_vals)

    tfr_stack:   list[np.ndarray] = []  # each [n_phases, n_cb]
    cosim_stack: list[np.ndarray] = []
    rell2_stack: list[np.ndarray] = []

    for freq in PHASE_FREQ_TAGS:
        sig    = f"self_phase_0dB_{freq}hz"
        sd     = root / sig
        base_p = sd / f"{sig}_baseline_bw{bw}_tokens.npy"
        base   = _load_tok(base_p)
        if base is None:
            continue

        tfr_f   = np.full((n_phases, n_cb), np.nan)
        cosim_f = np.full((n_phases, n_cb), np.nan)
        rell2_f = np.full((n_phases, n_cb), np.nan)

        for pi, phase in enumerate(phase_vals):
            tag   = str(int(phase))
            var_p = sd / f"{sig}_var_{tag}_bw{bw}_tokens.npy"
            var   = _load_tok(var_p)
            if var is None:
                continue
            ncb = min(base.shape[0], n_cb)
            n   = min(base.shape[1], var.shape[1])
            tfr_f[pi, :ncb] = (base[:ncb, :n] != var[:ncb, :n]).mean(axis=1)
            cm, cv = _cosim_at_flips(base[:ncb], var[:ncb], weights[:ncb])
            cosim_f[pi, :ncb] = cm
            rell2_f[pi, :ncb] = _rel_l2_at_flips(base[:ncb], var[:ncb], weights[:ncb])

        tfr_stack.append(tfr_f)
        cosim_stack.append(cosim_f)
        rell2_stack.append(rell2_f)

    if not tfr_stack:
        return {}

    tfr_arr   = np.nanmean(np.stack(tfr_stack,   0), 0)  # [n_phases, n_cb]
    cosim_arr = np.nanmean(np.stack(cosim_stack, 0), 0)
    # cross-freq variance of cosim used for shadow
    cosim_var = np.nanvar( np.stack(cosim_stack, 0), 0)
    rell2_arr = np.nanmean(np.stack(rell2_stack, 0), 0)

    # Normalise Rel-L2 globally per codec (preserves cross-depth shape)
    rl2_max = float(np.nanmax(rell2_arr)) if np.any(np.isfinite(rell2_arr)) else 1.0
    rl2_max = max(rl2_max, 1e-12)

    return {
        "phases":    phase_vals,
        "n_cb":      n_cb,
        "tfr":       tfr_arr,
        "cosim":     cosim_arr,
        "cosim_var": cosim_var,
        "rell2":     rell2_arr / rl2_max,
        "rell2_raw": rell2_arr,
        "rell2_max": rl2_max,
    }

# ---------------------------------------------------------------------------
# EGFX data loader — per codec, averaged across clean/processed pairs
# ---------------------------------------------------------------------------

def _load_egfx_metrics(
    codec: str,
    tokens_root: Path,
    weights: list[np.ndarray],
) -> dict[str, dict]:
    """Per-category EGFX metrics averaged across all clean/processed pairs.

    TFR from token files; cosine similarity and Rel-L2 from saved embedding files
    (_emb_layer{i}.npy written by egfx_encode.py) — matching egfx_metrics.py.
    """
    root  = _egfx_root(tokens_root, codec)
    n_cb  = len(weights)
    raw:  dict[str, dict] = {}

    for cat in _EGFX_CATS:
        cat_dir = root / cat
        if not cat_dir.exists():
            continue

        tfr_list:   list[np.ndarray] = []
        cosim_list: list[np.ndarray] = []
        rell2_list: list[np.ndarray] = []

        for cp in sorted(cat_dir.glob("*_clean_tokens.npy")):
            pair_name = cp.stem.replace("_clean_tokens", "")
            pp = cat_dir / f"{pair_name}_processed_tokens.npy"
            if not pp.exists():
                continue
            clean = _load_tok(cp)
            proc  = _load_tok(pp)
            if clean is None or proc is None:
                continue
            ncb = min(clean.shape[0], n_cb)
            n   = min(clean.shape[1], proc.shape[1])

            tfr_row   = np.full(n_cb, np.nan)
            cosim_row = np.full(n_cb, np.nan)
            rell2_row = np.full(n_cb, np.nan)

            tfr_row[:ncb] = (clean[:ncb, :n] != proc[:ncb, :n]).mean(axis=1)

            # Cosine similarity and Rel-L2 from continuous encoder embeddings,
            # not codebook lookups — avoids packed-index issues (Q2D2, DAC-FSQ)
            for cb in range(ncb):
                ec_p = cat_dir / f"{pair_name}_clean_emb_layer{cb}.npy"
                ep_p = cat_dir / f"{pair_name}_processed_emb_layer{cb}.npy"
                if not ec_p.exists() or not ep_p.exists():
                    continue
                try:
                    ec = np.load(str(ec_p)).astype(np.float32)  # [D, T]
                    ep = np.load(str(ep_p)).astype(np.float32)  # [D, T]
                    if ec.ndim == 1: ec = ec[np.newaxis]
                    if ep.ndim == 1: ep = ep[np.newaxis]
                    # align on time axis (last dim); also cap to token array length
                    t = min(ec.shape[-1], ep.shape[-1], n)
                    W        = weights[cb]
                    mag_norm = max(float(np.mean(np.linalg.norm(W, axis=1))), 1e-12)
                    # Rel-L2 at flip sites only — matches sinusoid _rel_l2_at_flips
                    mask = clean[cb, :t] != proc[cb, :t]
                    if mask.any():
                        diff = ep[:, mask] - ec[:, mask]  # [D, n_flips]
                        rell2_row[cb] = float(np.linalg.norm(diff, axis=0).mean() / mag_norm)
                    else:
                        rell2_row[cb] = 0.0
                    ec_f = ec[:, :t].ravel(); ep_f = ep[:, :t].ravel()
                    nb = float(np.linalg.norm(ec_f))
                    nv = float(np.linalg.norm(ep_f))
                    if nb > 1e-12 and nv > 1e-12:
                        cosim_row[cb] = float(np.clip(np.dot(ec_f, ep_f) / (nb * nv), -1.0, 1.0))
                except Exception:
                    pass

            tfr_list.append(tfr_row)
            cosim_list.append(cosim_row)
            rell2_list.append(rell2_row)

        if not tfr_list:
            continue

        raw[cat] = {
            "n_cb":      n_cb,
            "tfr":       np.nanmean(np.stack(tfr_list,   0), 0),
            "cosim":     np.nanmean(np.stack(cosim_list, 0), 0),
            "cosim_var": np.nanvar( np.stack(cosim_list, 0), 0),
            "rell2_raw": np.nanmean(np.stack(rell2_list, 0), 0),
        }

    if not raw:
        return {}

    all_rl2 = np.concatenate([v["rell2_raw"] for v in raw.values()])
    rl2_max = float(np.nanmax(all_rl2)) if np.any(np.isfinite(all_rl2)) else 1.0
    rl2_max = max(rl2_max, 1e-12)
    for v in raw.values():
        v["rell2"] = v["rell2_raw"] / rl2_max

    return raw

# ---------------------------------------------------------------------------
# Shared subplot painter for the 3-metric line plot
# ---------------------------------------------------------------------------

def _paint_metrics(
    ax: plt.Axes,
    x: list[float],
    tfr: np.ndarray,
    cosim: np.ndarray,
    cosim_var: np.ndarray,
    rell2: np.ndarray,
    ms: float = 4,
) -> None:
    ax.plot(x, tfr,   color=_IBM_BLUE,  marker="o", ms=ms, lw=1.2, label="TFR")
    ax.plot(x, cosim, color=_IBM_TEAL,  marker="^", ms=ms, lw=1.2, label="Cos-sim")
    std_c = np.sqrt(np.clip(cosim_var, 0, None))
    # shadow unclamped — negative values indicate polarity flip
    ax.fill_between(x, cosim - std_c, cosim + std_c, color=_IBM_TEAL, alpha=0.15)
    ax.plot(x, rell2, color=_IBM_WARM,  marker="s", ms=ms, lw=1.2, label="Rel-L₂")

# ---------------------------------------------------------------------------
# Figure 1 — Phase sensitivity per codec at depths 1, 4, 8
# ---------------------------------------------------------------------------

def _fig1(all_phase: dict[str, dict]) -> plt.Figure:
    codecs = _CODEC_ORDER
    n_rows = len(codecs)
    n_cols = len(_FIG1_DEPTHS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7.16, 1.55 * n_rows + 0.9),
        squeeze=False,
    )
    fig.suptitle(
        "Figure 1 — Phase Sensitivity: TFR, Cosine Similarity, and Rel-L₂ by Depth",
        fontsize=9, fontweight="bold",
    )

    for ri, codec in enumerate(codecs):
        data  = all_phase.get(codec, {})
        label = _CODEC_STYLE[codec]["label"]

        for ci, depth in enumerate(_FIG1_DEPTHS):
            ax    = axes[ri][ci]
            cb_i  = depth - 1  # 0-indexed codebook

            if not data or cb_i >= data.get("n_cb", 0):
                ax.axis("off")
                if ri == 0:
                    ax.set_title(f"Depth {depth}", fontsize=8, pad=4)
                if ci == 0:
                    ax.set_ylabel(label, fontsize=7, fontweight="bold", labelpad=4)
                continue

            phases = data["phases"]
            x      = [0.0] + phases
            tfr    = np.concatenate([[0.0], data["tfr"][:, cb_i]])
            cosim  = np.concatenate([[1.0], data["cosim"][:, cb_i]])
            cv     = np.concatenate([[0.0], data["cosim_var"][:, cb_i]])
            rl2    = np.concatenate([[0.0], data["rell2"][:, cb_i]])

            _paint_metrics(ax, x, tfr, cosim, cv, rl2, ms=3)

            ax.set_xlim(-5, 375)
            ax.set_ylim(-0.5, 1.10)
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.18, lw=0.5)

            if ri == 0:
                ax.set_title(f"Depth {depth}", fontsize=8, pad=4)
            if ci == 0:
                ax.set_ylabel(label, fontsize=7, fontweight="bold", labelpad=4)
            if ri == n_rows - 1:
                ax.set_xlabel("Phase (°)", fontsize=7)

    # Shared legend
    handles = [
        mlines.Line2D([], [], color=_IBM_BLUE,  marker="o", ms=5, label="TFR"),
        mlines.Line2D([], [], color=_IBM_TEAL,  marker="^", ms=5, label="Cos-sim"),
        mlines.Line2D([], [], color=_IBM_WARM,  marker="s", ms=5, label="Rel-L₂ (norm.)"),
    ]
    fig.legend(handles=handles, loc="lower center", fontsize=7,
               ncol=3, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    return fig

# ---------------------------------------------------------------------------
# Figure 2 — EGFX response per codec per category
# ---------------------------------------------------------------------------

def _fig2(all_egfx: dict[str, dict[str, dict]]) -> plt.Figure:
    codecs = _CODEC_ORDER
    n_rows = len(codecs)
    n_cols = len(_EGFX_CATS)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(7.16, 1.55 * n_rows + 0.9),
        squeeze=False,
    )
    fig.suptitle(
        "Figure 2 — EGFX Response: TFR, Cosine Similarity, and Rel-L₂ by Depth",
        fontsize=9, fontweight="bold",
    )

    for ri, codec in enumerate(codecs):
        cat_data = all_egfx.get(codec, {})
        label    = _CODEC_STYLE[codec]["label"]

        for ci, cat in enumerate(_EGFX_CATS):
            ax   = axes[ri][ci]
            data = cat_data.get(cat, {})

            if not data:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7, color="#8d8375")
                ax.set_facecolor("#f4f4f4")
                for spine in ax.spines.values():
                    spine.set_edgecolor("#d0d0d0")
                ax.set_xticks([]); ax.set_yticks([])
                if ri == 0:
                    ax.set_title(_EGFX_CAT_LABEL[cat], fontsize=8, pad=4)
                if ci == 0:
                    ax.set_ylabel(label, fontsize=7, fontweight="bold", labelpad=4)
                continue

            n_cb   = int(data["n_cb"])
            depths = list(range(1, n_cb + 1))
            tfr    = data["tfr"][:n_cb]
            cosim  = data["cosim"][:n_cb]
            cv     = data["cosim_var"][:n_cb]
            rl2    = data["rell2"][:n_cb]

            _paint_metrics(ax, depths, tfr, cosim, cv, rl2, ms=3)

            ax.set_xlim(0.5, n_cb + 0.5)
            ax.set_ylim(-0.5, 1.10)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.18, lw=0.5)

            if ri == 0:
                ax.set_title(_EGFX_CAT_LABEL[cat], fontsize=8, pad=4)
            if ci == 0:
                ax.set_ylabel(label, fontsize=7, fontweight="bold", labelpad=4)
            if ri == n_rows - 1:
                ax.set_xlabel("Quantisation Depth", fontsize=7)

    handles = [
        mlines.Line2D([], [], color=_IBM_BLUE, marker="o", ms=5, label="TFR"),
        mlines.Line2D([], [], color=_IBM_TEAL, marker="^", ms=5, label="Cos-sim"),
        mlines.Line2D([], [], color=_IBM_WARM, marker="s", ms=5, label="Rel-L₂ (norm.)"),
    ]
    fig.legend(handles=handles, loc="lower center", fontsize=7,
               ncol=3, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    return fig

# ---------------------------------------------------------------------------
# Figure 3 — Centroid magnitude vs depth (normalised per codec)
# ---------------------------------------------------------------------------

def _fig3(all_weights: dict[str, list[np.ndarray]]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.16, 4.5))
    fig.suptitle("Figure 3 — Mean Centroid Magnitude by Depth",
                 fontsize=9, fontweight="bold")

    for codec in _CODEC_ORDER:
        W_list = all_weights.get(codec, [])
        if not W_list:
            continue
        props = _CODEC_STYLE[codec]
        mags  = np.array([float(np.mean(np.linalg.norm(W, axis=1))) for W in W_list])
        mx    = float(np.nanmax(mags))
        # zero-anchored: 0 = truly no magnitude; min-max would map the minimum to 0
        norm  = mags / mx if mx > 1e-12 else np.zeros(len(mags))
        depths = np.arange(1, len(mags) + 1)
        ax.plot(depths, norm, color=props["color"], marker=props["marker"],
                ms=5, lw=1.4, label=props["label"])

    ax.set_xlabel("Quantisation Depth", fontsize=9)
    ax.set_ylabel("Normalised Centroid Magnitude", fontsize=9)
    ax.set_ylim(-0.05, 1.10)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.25, lw=0.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

# ---------------------------------------------------------------------------
# Figure 4 — SVD EVR₂ (phase) + effective rank vs depth
# ---------------------------------------------------------------------------

def _fig4(
    all_weights: dict[str, list[np.ndarray]],
    all_phase_roots: dict[str, Path],
) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(7.16, 4.5))
    fig.suptitle("Figure 4 — SVD EVR₂ (Phase) and Effective Rank by Depth",
                 fontsize=9, fontweight="bold")
    ax2 = ax1.twinx()

    any_data = False
    for codec in _CODEC_ORDER:
        W_list = all_weights.get(codec, [])
        if not W_list:
            continue
        props      = _CODEC_STYLE[codec]
        bw         = _BW_TAG[codec]
        phase_root = all_phase_roots.get(codec)
        if phase_root is None or not phase_root.exists():
            continue
        try:
            cb_deltas = _svd_collect_sinusoid_deltas(
                phase_root, "self_phase_0dB_*", bw, W_list)
            evr2, _, eff_rank = _svd_metrics_from_deltas(cb_deltas, len(W_list))
        except Exception as exc:
            print(f"  [SKIP] SVD Fig4 {codec}: {exc}")
            continue

        depths = np.arange(1, len(W_list) + 1)
        valid  = ~np.isnan(evr2)
        ax1.plot(depths[valid], evr2[valid] * 100,
                 color=props["color"], marker=props["marker"],
                 ms=5, lw=1.4, label=props["label"])
        ax2.plot(depths[valid], eff_rank[valid],
                 color=props["color"], marker=props["marker"],
                 ms=3, lw=0.9, linestyle="--", alpha=0.55)
        any_data = True

    ax1.set_xlabel("Quantisation Depth", fontsize=9)
    ax1.set_ylabel("EVR₂ Phase (%)", fontsize=9, color=_IBM_BLUE)
    ax1.set_ylim(0, 108)
    ax1.tick_params(axis="y", labelcolor=_IBM_BLUE, labelsize=7)
    ax1.tick_params(axis="x", labelsize=7)
    ax2.set_ylabel("Effective Rank  (dashed)", fontsize=8, color=_IBM_GRAY)
    ax2.tick_params(axis="y", labelcolor=_IBM_GRAY, labelsize=7)
    ax1.grid(True, alpha=0.25, lw=0.5)
    ax1.legend(fontsize=7, loc="lower left")
    if not any_data:
        ax1.text(0.5, 0.5, "no token data found",
                 ha="center", va="center", transform=ax1.transAxes, color="gray")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig

# ---------------------------------------------------------------------------
# Figures 5 & 6 — PCA scatter of phase ΔZ (smaller colorbar)
# ---------------------------------------------------------------------------

_PHASE_RE = re.compile(r"_var_([^_]+)_bw")


def _pca_scatter_fig(
    phase_root: Path,
    bw: str,
    weights: list[np.ndarray],
    codec_label: str,
    fig_num: int,
) -> plt.Figure:
    """PC1 vs PC2 scatter at representative depths.  Colorbar = 1/8 subplot width."""
    n_cb    = len(weights)
    indices = sorted({0, n_cb // 4, n_cb // 2, n_cb - 1})
    n_plots = len(indices)
    is_q2d2 = (n_cb <= 16 and weights[0].shape[1] == 2)
    unit    = "Grid Pair" if is_q2d2 else "Codebook"

    tagged: dict[int, list[tuple[float, np.ndarray]]] = {cb: [] for cb in indices}

    for sd in sorted(phase_root.glob("self_phase_0dB_*")):
        sig    = sd.name
        base_p = sd / f"{sig}_baseline_bw{bw}_tokens.npy"
        base   = _load_tok(base_p)
        if base is None or base.ndim != 2:
            continue
        for vp in sorted(sd.glob(f"{sig}_var_*_bw{bw}_tokens.npy")):
            m = _PHASE_RE.search(vp.name)
            if not m:
                continue
            try:
                angle = float(m.group(1))
            except ValueError:
                continue
            var = _load_tok(vp)
            if var is None or var.ndim != 2:
                continue
            nf = min(base.shape[1], var.shape[1])
            for cb in indices:
                if cb >= base.shape[0]:
                    continue
                W = weights[cb]
                delta = (W[var[cb, :nf].astype(np.intp)].mean(0)
                         - W[base[cb, :nf].astype(np.intp)].mean(0))
                tagged[cb].append((angle, delta))

    cmap  = plt.cm.hsv
    fig_w = 7.16
    fig_h = fig_w / n_plots + 0.9

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 360))
    sm.set_array([])

    # Colorbar is added after layout so its height matches the square subplots exactly
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = fig.add_gridspec(1, n_plots, wspace=0.65, left=0.27)
    plot_axes = [fig.add_subplot(gs[0, i]) for i in range(n_plots)]

    for ax, cb in zip(plot_axes, indices):
        ax.set_box_aspect(1)
        items = tagged[cb]
        if len(items) < 2:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_title(f"{unit} {cb+1}", fontsize=8)
            continue
        angles = np.array([a for a, _ in items])
        mat    = np.stack([d for _, d in items])
        centre = mat.mean(0)
        _, s, Vt = np.linalg.svd(mat - centre, full_matrices=False)
        pc1 = (mat - centre) @ Vt[0]
        pc2 = (mat - centre) @ Vt[1] if Vt.shape[0] > 1 else np.zeros(len(items))
        s2  = s ** 2; total = s2.sum()
        evr2 = float(s2[:2].sum() / total) if total > 0 else 0.0

        ax.scatter(pc1, pc2, c=cmap(angles / 360.0), s=14, alpha=0.75)
        
        # added for easier visualization before the aspect ratio locks
        ax.margins(0.15) 
        
        ax.set_title(f"{unit} {cb+1}\nEVR₂={evr2:.2f}", fontsize=8, pad=6)
        ax.set_xlabel("PC 1", fontsize=7)
        ax.set_ylabel("PC 2", fontsize=7)
        ax.axhline(0, color="#8d8375", lw=0.5)
        ax.axvline(0, color="#8d8375", lw=0.5)
        ax.set_aspect("equal", "datalim")
        ax.tick_params(labelsize=6)

    style_label = "Isotropic" if is_q2d2 else "Anisotropic"
    fig.suptitle(
        f"Figure {fig_num} — {codec_label} Phase ΔZ PCA  ({style_label} Response)\n"
        "colour = phase angle °;  ideal circle → points form a closed ring",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.87])

    # Place colorbar after layout so its height matches the rendered subplot squares
    fig.canvas.draw()
    pos0 = plot_axes[0].get_position()
    cbar_ax = fig.add_axes([pos0.x0 - 0.15, pos0.y0, 0.015, pos0.height])
    cb_obj = fig.colorbar(sm, cax=cbar_ax, orientation="vertical")
    cb_obj.set_label("Phase (°)", fontsize=8)
    cb_obj.ax.tick_params(labelsize=7)

    return fig

# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def _export_csvs(
    out_dir: Path,
    all_phase: dict[str, dict],
    all_egfx:  dict[str, dict[str, dict]],
    all_weights: dict[str, list[np.ndarray]],
    all_svd:   dict[str, tuple[np.ndarray, np.ndarray]],  # codec → (evr2, eff_rank)
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fig 1
    with open(out_dir / "fig1_phase_sensitivity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["codec", "depth_col", "phase_deg",
                    "tfr", "cosim", "cosim_var", "rell2_norm"])
        for codec in _CODEC_ORDER:
            d = all_phase.get(codec, {})
            if not d:
                continue
            for depth in _FIG1_DEPTHS:
                cb_i = depth - 1
                if cb_i >= d["n_cb"]:
                    continue
                x     = [0.0] + d["phases"]
                tfr   = np.concatenate([[0.0], d["tfr"][:,   cb_i]])
                cosim = np.concatenate([[1.0], d["cosim"][:,  cb_i]])
                cv    = np.concatenate([[0.0], d["cosim_var"][:,cb_i]])
                rl2   = np.concatenate([[0.0], d["rell2"][:,  cb_i]])
                for i, ph in enumerate(x):
                    w.writerow([codec, depth, ph,
                                f"{tfr[i]:.6f}", f"{cosim[i]:.6f}",
                                f"{cv[i]:.6f}",  f"{rl2[i]:.6f}"])

    # Fig 2
    with open(out_dir / "fig2_egfx.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["codec", "effect_category", "depth",
                    "tfr", "cosim", "cosim_var", "rell2_norm"])
        for codec in _CODEC_ORDER:
            for cat in _EGFX_CATS:
                d = all_egfx.get(codec, {}).get(cat, {})
                if not d:
                    continue
                for cb in range(d["n_cb"]):
                    w.writerow([codec, cat, cb + 1,
                                f"{d['tfr'][cb]:.6f}",
                                f"{d['cosim'][cb]:.6f}",
                                f"{d['cosim_var'][cb]:.6f}",
                                f"{d['rell2'][cb]:.6f}"])

    # Fig 3
    with open(out_dir / "fig3_centroid_magnitude.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["codec", "depth", "mean_mag_raw", "mean_mag_norm"])
        for codec in _CODEC_ORDER:
            W_list = all_weights.get(codec, [])
            if not W_list:
                continue
            mags = np.array([float(np.mean(np.linalg.norm(W, axis=1))) for W in W_list])
            mx   = float(np.nanmax(mags))
            norm = mags / mx if mx > 1e-12 else np.zeros(len(mags))
            for cb, (m, n) in enumerate(zip(mags, norm)):
                w.writerow([codec, cb + 1, f"{m:.6f}", f"{n:.6f}"])

    # Fig 4
    with open(out_dir / "fig4_svd_evr2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["codec", "depth", "evr2_phase_pct", "eff_rank"])
        for codec in _CODEC_ORDER:
            if codec not in all_svd:
                continue
            evr2, eff_rank = all_svd[codec]
            for cb, (e, r) in enumerate(zip(evr2, eff_rank)):
                w.writerow([codec, cb + 1,
                             f"{e*100:.4f}" if np.isfinite(e) else "",
                             f"{r:.4f}"     if np.isfinite(r) else ""])

    print(f"  CSVs written → {out_dir}")

# ---------------------------------------------------------------------------
# Checkpoint auto-discovery (mirrors run_analysis.py)
# ---------------------------------------------------------------------------

def _autodiscover_checkpoints() -> dict[str, str]:
    ckpts: dict[str, str] = {}

    # EnCodec
    candidates = [
        p for p in (_PROJ / "Encodec").glob(
            "outputs/**/checkpoints_multi_dataset/*.pt")
        if "_disc" not in p.stem
    ]
    preferred = [p for p in candidates if "best" in p.stem] or candidates
    if preferred:
        ckpts["encodec"] = str(sorted(preferred, key=lambda p: p.stat().st_mtime)[-1])

    # Q2D2
    q2_matches = sorted(
        (_PROJ / "Q2D2").glob(
            "outputs/lightning_logs/version_*/checkpoints/Q2D2_best.ckpt"),
        key=lambda p: p.stat().st_mtime,
    )
    if q2_matches:
        ckpts["q2d2"] = str(q2_matches[-1])

    # HiFiCodec (directory)
    hfc = _PROJ / "hificodec" / "egs" / "hificodec_fsd50k"
    if hfc.exists():
        ckpts["hificodec"] = str(hfc)

    # SpeechTokenizer
    st = _PROJ / "results" / "speechtokenizer_fsd50k" / "SpeechTokenizer_best_dev.pt"
    if st.exists():
        ckpts["speechtokenizer"] = str(st)

    # DAC-FSQ (no checkpoint needed but pass dir for completeness)
    dac = _PROJ / "descript-audio-codec" / "ckpt" / "fsd50k_fsq" / "best"
    if dac.exists():
        ckpts["dac_fsq"] = str(dac)

    return ckpts

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    _ds = _PROJ / "datasets"
    ap  = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens-root", default=str(_ds / "audio_tokens"),
                    help="Root directory containing all codec token .npy files")
    ap.add_argument("--encodec-ckpt",    default="", help="EnCodec checkpoint .pt")
    ap.add_argument("--q2d2-ckpt",       default="", help="Q2D2 checkpoint .ckpt")
    ap.add_argument("--hificodec-ckpt",  default="", help="HiFiCodec checkpoint dir")
    ap.add_argument("--st-ckpt",         default="", help="SpeechTokenizer checkpoint .pt")
    ap.add_argument("--dac-ckpt",        default="", help="DAC-FSQ checkpoint dir (optional)")
    ap.add_argument("--output",  default=str(_PROJ / "results" / "paper_figures.pdf"),
                    help="Output PDF path")
    ap.add_argument("--csv-dir", default=str(_PROJ / "results" / "paper_figures_csv"),
                    help="Directory for CSV exports")
    args = ap.parse_args()

    tokens_root = Path(args.tokens_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Fill missing checkpoints via auto-discovery
    auto = _autodiscover_checkpoints()
    ckpt_map = {
        "encodec":         args.encodec_ckpt   or auto.get("encodec", ""),
        "q2d2":            args.q2d2_ckpt      or auto.get("q2d2", ""),
        "hificodec":       args.hificodec_ckpt or auto.get("hificodec", ""),
        "speechtokenizer": args.st_ckpt        or auto.get("speechtokenizer", ""),
        "dac_fsq":         args.dac_ckpt       or auto.get("dac_fsq", ""),
    }

    # ── 1. Load codebook weights ───────────────────────────────────────────
    print("\n[1/5] Loading codebook weights …")
    all_weights: dict[str, list[np.ndarray]] = {}
    for codec in _CODEC_ORDER:
        w = _load_codebook_weights(ckpt_map[codec], _MODEL_NAME[codec])
        if w is None:
            print(f"  [WARN] {codec}: no weights — omitted from Figs 3/4/5/6")
            all_weights[codec] = []
        else:
            all_weights[codec] = w
            print(f"  {codec}: {len(w)} units, first shape {w[0].shape}")

    # ── 2. Phase metrics ───────────────────────────────────────────────────
    print("\n[2/5] Computing phase metrics (Figs 1 & 4) …")
    all_phase: dict[str, dict] = {}
    for codec in _CODEC_ORDER:
        W = all_weights.get(codec, [])
        if not W:
            all_phase[codec] = {}
            continue
        print(f"  {codec} …", end=" ", flush=True)
        data = _load_phase_metrics(codec, tokens_root, W)
        all_phase[codec] = data
        print(f"found {len(data.get('phases', []))} phase steps, "
              f"{data.get('n_cb', 0)} units" if data else "no token files")

    # ── 3. EGFX metrics ────────────────────────────────────────────────────
    print("\n[3/5] Computing EGFX metrics (Fig 2) …")
    all_egfx: dict[str, dict[str, dict]] = {}
    for codec in _CODEC_ORDER:
        W = all_weights.get(codec, [])
        if not W:
            all_egfx[codec] = {}
            continue
        print(f"  {codec} …", end=" ", flush=True)
        data = _load_egfx_metrics(codec, tokens_root, W)
        all_egfx[codec] = data
        print(list(data.keys()) if data else "no EGFX token files")

    # ── 4. SVD EVR₂ (Fig 4) ───────────────────────────────────────────────
    print("\n[4/5] Computing SVD EVR₂ (Fig 4) …")
    phase_roots = {c: _phase_root(tokens_root, c) for c in _CODEC_ORDER}
    all_svd: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for codec in _CODEC_ORDER:
        W = all_weights.get(codec, [])
        if not W:
            continue
        pr = phase_roots[codec]
        if not pr.exists():
            print(f"  [SKIP] {codec}: phase root not found ({pr})")
            continue
        try:
            print(f"  {codec} …", end=" ", flush=True)
            cb_deltas = _svd_collect_sinusoid_deltas(
                pr, "self_phase_0dB_*", _BW_TAG[codec], W)
            evr2, _, eff_rank = _svd_metrics_from_deltas(cb_deltas, len(W))
            all_svd[codec] = (evr2, eff_rank)
            valid = int(np.sum(np.isfinite(evr2)))
            print(f"{valid}/{len(W)} units with valid EVR₂")
        except Exception as exc:
            print(f"[SKIP] {exc}")

    # ── 5. Build PDF ───────────────────────────────────────────────────────
    print(f"\n[5/5] Building PDF → {output_path}")
    with PdfPages(str(output_path)) as pdf:
        print("  Fig 1 — Phase sensitivity …")
        pdf.savefig(_fig1(all_phase), bbox_inches="tight"); plt.close("all")

        print("  Fig 2 — EGFX response …")
        pdf.savefig(_fig2(all_egfx), bbox_inches="tight"); plt.close("all")

        print("  Fig 3 — Centroid magnitude …")
        pdf.savefig(_fig3(all_weights), bbox_inches="tight"); plt.close("all")

        print("  Fig 4 — SVD EVR₂ …")
        fig4 = _fig4(all_weights, phase_roots)
        pdf.savefig(fig4, bbox_inches="tight"); plt.close("all")

        for fig_num, codec in [(5, "encodec"), (6, "q2d2")]:
            W = all_weights.get(codec, [])
            if not W:
                print(f"  Fig {fig_num} — skipped (no {codec} weights)")
                continue
            pr = phase_roots[codec]
            if not pr.exists():
                print(f"  Fig {fig_num} — skipped ({pr} not found)")
                continue
            print(f"  Fig {fig_num} — PCA scatter ({_CODEC_STYLE[codec]['label']}) …")
            try:
                fig = _pca_scatter_fig(pr, _BW_TAG[codec], W,
                                       _CODEC_STYLE[codec]["label"], fig_num)
                pdf.savefig(fig, bbox_inches="tight"); plt.close("all")
            except Exception as exc:
                print(f"    [SKIP] {exc}")

    _export_csvs(Path(args.csv_dir), all_phase, all_egfx, all_weights, all_svd)
    print(f"\nDone.\n  PDF: {output_path}\n  CSVs: {args.csv_dir}")


if __name__ == "__main__":
    main()
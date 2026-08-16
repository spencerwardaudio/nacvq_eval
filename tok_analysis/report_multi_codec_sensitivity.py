"""Multi-codec combined sensitivity report.

Runs the amplitude / phase / temporal token flip-rate and perplexity
analyses for each requested codec and assembles everything into a single
PDF, followed by cross-codec summary pages that plot mean flip rate and
perplexity side-by-side, normalised by quantizer depth position.

Codec configurations
--------------------
Codec         BW tag   n_cb  Quantizer type
-----------   ------   ----  ---------------
encodec       24.0     32    RVQ (sequential)
q2d2          9.8      16    2D-Grid FSQ (parallel)
hificodec     HFC       4    GRVQ (grouped)
speechtoken   ST        8    RVQ (sequential)
dac_fsq       FSQ       1    FSQ (flat single stream)

Usage (from project root):
    python tok_analysis/report_multi_codec_sensitivity.py \\
        --codecs encodec q2d2 hificodec speechtokenizer dac_fsq \\
        [--tokens-root  datasets/audio_tokens] \\
        [--encodec-bw   24.0] \\
        [--q2d2-bw      9.8] \\
        [--hificodec-bw-tag HFC] \\
        [--speechtokenizer-bw-tag ST] \\
        [--dac-fsq-bw-tag FSQ] \\
        [--output       datasets/analysis/multi_codec_sensitivity.pdf]

Per-codec token files must already exist.  Run the encode scripts first:
    python tok_analysis/encode_hificodec_tokens.py  --checkpoint ...
    python tok_analysis/encode_speechtokenizer_tokens.py --checkpoint ...
    python tok_analysis/encode_dac_fsq_tokens.py --checkpoint ...

Encodec and Q2D2 tokens are produced by the existing batch_encode_*.py
scripts in tok_analysis/.
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
_DS = _PROJ_ROOT / "datasets"

sys.path.insert(0, str(_HERE))
if str(_ENCODEC_DIR) not in sys.path:
    sys.path.insert(0, str(_ENCODEC_DIR))

from analyze_self_amp_test import (
    SIGNALS as AMP_SIGNALS,
    _TAG_TO_LABEL as AMP_TAG_TO_LABEL,
    analyze_signal as amp_analyze_signal,
    _per_codebook_amplitude_figure,
)
from analyze_self_phase_test import (
    SIGNALS as PHASE_SIGNALS,
    analyze_signal as phase_analyze_signal,
    _per_codebook_phase_figure,
)
from analyze_sine_temporal import (
    FREQ_TAGS as TEMPORAL_FREQ_TAGS,
    analyze_freq as temporal_analyze_freq,
    _per_codebook_temporal_figure,
)
from report_combined_sensitivity import (
    _get_unit_label,
    _derive_is_q2d2,
    _MODEL_UNIT_LABELS,
    _codebook_perplexity,
    _per_codebook_amplitude_perplexity_figure,
    _per_codebook_phase_perplexity_figure,
    _per_codebook_temporal_perplexity_figure,
)


# ---------------------------------------------------------------------------
# Codec registry
# ---------------------------------------------------------------------------

CODEC_CONFIGS: dict[str, dict] = {
    "encodec": {
        "label":    "EnCodec",
        "color":    "#2196F3",   # blue
        "n_cb_max": 32,
        "quantizer": "RVQ (sequential)",
        "vocab_size": 1024,
    },
    "q2d2": {
        "label":    "Q2D2",
        "color":    "#4CAF50",   # green
        "n_cb_max": 16,
        "quantizer": "2D-Grid FSQ (parallel)",
        "vocab_size": [81]*5 + [49]*11,  # 9.8 kbps: streams 1-5 = 9×9, 6-16 = 7×7
    },
    "hificodec": {
        "label":    "HiFiCodec",
        "color":    "#FF9800",   # orange
        "n_cb_max": 4,
        "quantizer": "GRVQ (grouped residual)",
        "vocab_size": 1024,
    },
    "speechtokenizer": {
        "label":    "SpeechTokenizer",
        "color":    "#E91E63",   # pink
        "n_cb_max": 8,
        "quantizer": "RVQ (sequential)",
        "vocab_size": 1024,
    },
    "dac_fsq": {
        "label":    "DAC-FSQ",
        "color":    "#9C27B0",   # purple
        "n_cb_max": 1,
        "quantizer": "FSQ (flat single stream)",
        "vocab_size": 2_560_000,  # [8,8,8,8,5,5,5,5] mixed-radix flat index
    },
}

# encodec/q2d2 tokens live in the flat layout (no codec subdir prefix)
_CODEC_TOKEN_SUBDIR: dict[str, str] = {
    "encodec":         "",
    "q2d2":            "",
    "hificodec":       "hificodec",
    "speechtokenizer": "speechtokenizer",
    "dac_fsq":         "dac_fsq",
}


def _codec_root(tokens_root: Path, codec: str) -> Path:
    sub = _CODEC_TOKEN_SUBDIR.get(codec, codec)
    return tokens_root / sub if sub else tokens_root


# ---------------------------------------------------------------------------
# Token loading helpers
# ---------------------------------------------------------------------------

def _bw_tag(codec: str, bw_tags: dict[str, str]) -> str:
    return bw_tags.get(codec, codec.upper()[:3])


def _load_amp_stats(codec: str, tokens_root: Path, bw: str) -> list[dict]:
    amp_root = _codec_root(tokens_root, codec) / "dsp_self_amp"
    if not amp_root.exists():
        return []
    stats = []
    for signal in AMP_SIGNALS:
        sub = amp_root / signal
        if not sub.is_dir():
            continue
        tmp = amp_root.parent.parent / "analysis" / codec / "dsp_self_amp" / signal
        s = amp_analyze_signal(signal, sub, tmp, bw)
        if s is not None:
            stats.append(s)
    return stats


def _load_phase_stats(codec: str, tokens_root: Path, bw: str) -> tuple[list[dict], list[dict]]:
    phase_root = _codec_root(tokens_root, codec) / "dsp_self_phase"
    if not phase_root.exists():
        return [], []
    all_stats = []
    for signal in PHASE_SIGNALS:
        sub = phase_root / signal
        if not sub.is_dir():
            continue
        tmp = phase_root.parent.parent / "analysis" / codec / "dsp_self_phase" / signal
        s = phase_analyze_signal(signal, sub, tmp, bw)
        if s is not None:
            all_stats.append(s)
    stats_0dB = [s for s in all_stats if s.get("amp_label") == "0dB"]
    return all_stats, stats_0dB


def _load_temporal_stats(codec: str, tokens_root: Path, bw: str) -> dict:
    temp_root = _codec_root(tokens_root, codec) / "time_sine"
    if not temp_root.exists():
        return {}
    freq_data = {}
    for freq in TEMPORAL_FREQ_TAGS:
        sub = temp_root / f"{freq}hz"
        if not sub.is_dir():
            continue
        r = temporal_analyze_freq(freq, sub, bw)
        if r is not None:
            freq_data[freq] = r
    return freq_data


# ---------------------------------------------------------------------------
# Title pages
# ---------------------------------------------------------------------------

def _global_title_page(codecs: list[str], bw_tags: dict[str, str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(28, 14))
    ax.axis("off")

    rows = [
        f"Generated : {date.today().isoformat()}",
        "",
        "─" * 70,
        "",
        "CODEC COMPARISON",
        "",
    ]
    for codec in codecs:
        cfg = CODEC_CONFIGS.get(codec, {})
        bw  = bw_tags.get(codec, "?")
        rows.append(
            f"  {cfg.get('label', codec):<16}  BW tag={bw:<8}  "
            f"n_units={cfg.get('n_cb_max','?'):<4}  {cfg.get('quantizer','')}"
        )

    rows += [
        "",
        "─" * 70,
        "",
        "REPORT STRUCTURE",
        "  • Per-codec sections: flip-rate grids (amp / phase / temporal)",
        "    and perplexity grids — one page set per codec.",
        "  • Cross-codec summary (final section):",
        "    – Mean flip rate vs quantizer depth (position 1=coarsest to N=finest)",
        "    – Mean perplexity vs quantizer depth",
        "    – Both amplitude and temporal conditions overlaid.",
        "",
        "─" * 70,
        "",
        "NOTE ON DEPTH NORMALISATION",
        "  Each codec's N quantization units are mapped to positions 1..N.",
        "  RVQ (EnCodec, SpeechTokenizer): coarse-to-fine residual cascade.",
        "  GRVQ (HiFiCodec): 2 residual layers × 2 groups = 4 codebooks.",
        "  FSQ flat (DAC-FSQ): single stream — only position 1 exists.",
        "  Grid FSQ (Q2D2): parallel 2D grids — no strict ordering.",
    ]

    ax.text(0.05, 0.95, "\n".join(rows),
            ha="left", va="top", family="monospace", fontsize=10,
            transform=ax.transAxes, linespacing=1.55)
    ax.text(0.5, 0.99,
            "Multi-Codec Sensitivity Report — Sine Waves",
            ha="center", va="top", fontsize=18, fontweight="bold",
            transform=ax.transAxes)
    return fig


def _codec_section_title(codec: str, bw: str, n_amp: int,
                          n_phase: int, n_temporal: int) -> plt.Figure:
    cfg = CODEC_CONFIGS.get(codec, {})
    fig, ax = plt.subplots(figsize=(28, 6))
    ax.axis("off")
    color = cfg.get("color", "#333")
    ax.add_patch(plt.Rectangle((0, 0.7), 1, 0.3,
                                transform=ax.transAxes,
                                color=color, alpha=0.15))
    ax.text(0.5, 0.85, cfg.get("label", codec),
            ha="center", va="center", fontsize=28, fontweight="bold",
            transform=ax.transAxes, color=color)
    info = (
        f"Quantizer: {cfg.get('quantizer', '')}   |   "
        f"BW tag: {bw}   |   "
        f"Amp signals: {n_amp}   |   Phase signals: {n_phase}   |   "
        f"Temporal freqs: {n_temporal}"
    )
    ax.text(0.5, 0.50, info,
            ha="center", va="center", fontsize=13,
            transform=ax.transAxes, family="monospace")
    return fig


# ---------------------------------------------------------------------------
# Cross-codec summary figures
# ---------------------------------------------------------------------------

def _cross_codec_flip_rate_by_depth(
    codec_amp_stats: dict[str, list[dict]],
    bw_tags: dict[str, str],
) -> plt.Figure:
    """Line plot: mean flip rate vs normalised depth position (1=coarsest)."""
    fig, axes = plt.subplots(1, 2, figsize=(24, 8))
    fig.suptitle(
        "Cross-Codec: Mean Flip Rate vs Quantizer Depth (Amplitude Test)\n"
        "Position 1 = coarsest / lowest-index unit; normalised to [1, N]",
        fontsize=14,
    )

    for ax_idx, (condition, x_label) in enumerate([
        ("baseline vs −20 dBFS", "Variant: −20 dBFS"),
        ("baseline vs −80 dBFS", "Variant: −80 dBFS"),
    ]):
        ax = axes[ax_idx]
        ax.set_title(condition, fontsize=12)

        for codec, all_stats in codec_amp_stats.items():
            if not all_stats:
                continue
            cfg   = CODEC_CONFIGS.get(codec, {})
            color = cfg.get("color", "#888")
            label = cfg.get("label", codec)
            n_cb  = all_stats[0]["n_codebooks"]

            # Target variant index: 0 = −20 dBFS, 3 = −80 dBFS
            var_idx = 0 if ax_idx == 0 else 3

            flip_by_depth = np.full(n_cb, np.nan)
            counts = np.zeros(n_cb)

            for stats in all_stats:
                bw  = bw_tags.get(codec, "")
                base_tok = None
                var_tok  = None

                # Load tokens directly for flip rate calculation
                signal = stats["signal"]
                tokens_root = _codec_root(_DS / "audio_tokens", codec) / "dsp_self_amp" / signal
                base_path = tokens_root / f"{signal}_baseline_bw{bw}_tokens.npy"
                if not base_path.exists():
                    continue
                base_tok = np.load(str(base_path)).astype(np.int64)

                tags = stats.get("variant_tags", [])
                if var_idx >= len(tags):
                    continue
                tag = tags[var_idx]
                var_path = tokens_root / f"{signal}_var_{tag}_bw{bw}_tokens.npy"
                if not var_path.exists():
                    continue
                var_tok = np.load(str(var_path)).astype(np.int64)

                n = min(base_tok.shape[1], var_tok.shape[1])
                flips = (base_tok[:, :n] != var_tok[:, :n]).mean(axis=1)
                for cb in range(min(n_cb, len(flips))):
                    if not np.isnan(flips[cb]):
                        flip_by_depth[cb] = (
                            flip_by_depth[cb] * counts[cb] + flips[cb]
                        ) / (counts[cb] + 1) if not np.isnan(flip_by_depth[cb]) else flips[cb]
                        counts[cb] += 1

            valid = ~np.isnan(flip_by_depth)
            if not valid.any():
                continue
            x = np.arange(1, n_cb + 1)[valid]
            y = flip_by_depth[valid]
            ax.plot(x, y, marker="o", color=color, linewidth=2.2,
                    markersize=6, label=label)

        ax.set_xlabel("Quantizer depth position (1 = coarsest)", fontsize=11)
        ax.set_ylabel("Mean token flip rate", fontsize=11)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def _cross_codec_perplexity_by_depth(
    codec_amp_stats: dict[str, list[dict]],
    bw_tags: dict[str, str],
) -> plt.Figure:
    """Line plot: baseline perplexity vs quantizer depth for all codecs."""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(
        "Cross-Codec: Baseline Codebook Perplexity vs Quantizer Depth\n"
        "(mean across amplitude test frequencies, baseline = 0 dBFS signal)",
        fontsize=14,
    )

    for codec, all_stats in codec_amp_stats.items():
        if not all_stats:
            continue
        cfg   = CODEC_CONFIGS.get(codec, {})
        color = cfg.get("color", "#888")
        label = cfg.get("label", codec)
        n_cb  = all_stats[0]["n_codebooks"]
        bw    = bw_tags.get(codec, "")

        perp_by_depth = np.zeros(n_cb)
        counts = np.zeros(n_cb)

        for stats in all_stats:
            signal = stats["signal"]
            tokens_root = _codec_root(_DS / "audio_tokens", codec) / "dsp_self_amp" / signal
            base_path = tokens_root / f"{signal}_baseline_bw{bw}_tokens.npy"
            if not base_path.exists():
                continue
            base_tok = np.load(str(base_path)).astype(np.int64)
            perps = _codebook_perplexity(base_tok, vocab_size=cfg.get("vocab_size", 1024))
            for cb in range(min(n_cb, len(perps))):
                perp_by_depth[cb] += perps[cb]
                counts[cb] += 1

        valid = counts > 0
        if not valid.any():
            continue
        y = np.where(valid, perp_by_depth / np.where(valid, counts, 1), np.nan)
        x = np.arange(1, n_cb + 1)
        ax.plot(x[valid], y[valid], marker="o", color=color,
                linewidth=2.2, markersize=7, label=label)

    ax.set_xlabel("Quantizer depth position (1 = coarsest)", fontsize=12)
    ax.set_ylabel("Mean baseline perplexity  exp(H)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def _cross_codec_perplexity_table_figure(
    codecs: list[str],
    codec_amp_stats: dict[str, list[dict]],
    codec_phase_0dB: dict[str, list[dict]],
    codec_temporal: dict[str, dict],
    bw_tags: dict[str, str],
) -> plt.Figure:
    """One-page cross-codec perplexity summary table."""

    def _avg_perplexity_from_paths(paths: list[Path], vs) -> tuple[float, float, float]:
        vals = []
        for p in paths:
            if not p.exists():
                continue
            tokens = np.load(str(p)).astype(np.int64)
            perps = _codebook_perplexity(tokens, vocab_size=vs)
            vals.extend(perps.tolist())
        if not vals:
            return float("nan"), float("nan"), float("nan")
        arr = np.array(vals, dtype=np.float64)
        return float(np.mean(arr)), float(np.min(arr)), float(np.max(arr))

    table_rows = []
    for codec in codecs:
        cfg = CODEC_CONFIGS.get(codec, {})
        label = cfg.get("label", codec)
        unit_label = _MODEL_UNIT_LABELS.get(codec, "Codebook")
        bw = bw_tags.get(codec, "")
        n_units = cfg.get("n_cb_max", "?")
        vs = cfg.get("vocab_size", 1024)

        amp_base_paths: list[Path] = []
        amp_var_paths: list[Path] = []
        for stats in codec_amp_stats.get(codec, []):
            signal = stats["signal"]
            root = _codec_root(_DS / "audio_tokens", codec) / "dsp_self_amp" / signal
            amp_base_paths.append(root / f"{signal}_baseline_bw{bw}_tokens.npy")
            for tag in stats.get("variant_tags", []):
                amp_var_paths.append(root / f"{signal}_var_{tag}_bw{bw}_tokens.npy")

        phase_base_paths: list[Path] = []
        phase_var_paths: list[Path] = []
        for stats in codec_phase_0dB.get(codec, []):
            signal = stats["signal"]
            root = _codec_root(_DS / "audio_tokens", codec) / "dsp_self_phase" / signal
            phase_base_paths.append(root / f"{signal}_baseline_bw{bw}_tokens.npy")
            for tag in stats.get("variant_tags", []):
                phase_var_paths.append(root / f"{signal}_var_{tag}_bw{bw}_tokens.npy")

        temp_base_paths: list[Path] = []
        temp_var_paths: list[Path] = []
        for freq in sorted(codec_temporal.get(codec, {}).keys()):
            root = _codec_root(_DS / "audio_tokens", codec) / "time_sine" / f"{freq}hz"
            temp_base_paths.append(root / f"baseline_0ms_bw{bw}_tokens.npy")
            for offset_ms in sorted(codec_temporal[codec][freq].keys()):
                temp_var_paths.append(root / f"offset_{int(offset_ms):03d}ms_bw{bw}_tokens.npy")

        amp_base_mean, amp_base_min, amp_base_max = _avg_perplexity_from_paths(amp_base_paths, vs)
        amp_var_mean, _, _ = _avg_perplexity_from_paths(amp_var_paths, vs)
        phase_base_mean, _, _ = _avg_perplexity_from_paths(phase_base_paths, vs)
        phase_var_mean, _, _ = _avg_perplexity_from_paths(phase_var_paths, vs)
        temp_base_mean, _, _ = _avg_perplexity_from_paths(temp_base_paths, vs)
        temp_var_mean, _, _ = _avg_perplexity_from_paths(temp_var_paths, vs)

        table_rows.append([
            label,
            unit_label,
            str(n_units),
            f"{amp_base_mean:.1f}" if np.isfinite(amp_base_mean) else "-",
            f"{amp_var_mean:.1f}" if np.isfinite(amp_var_mean) else "-",
            f"{phase_base_mean:.1f}" if np.isfinite(phase_base_mean) else "-",
            f"{phase_var_mean:.1f}" if np.isfinite(phase_var_mean) else "-",
            f"{temp_base_mean:.1f}" if np.isfinite(temp_base_mean) else "-",
            f"{temp_var_mean:.1f}" if np.isfinite(temp_var_mean) else "-",
            (
                f"{amp_base_min:.1f} / {amp_base_max:.1f}"
                if np.isfinite(amp_base_min) and np.isfinite(amp_base_max)
                else "-"
            ),
        ])

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.axis("off")
    col_labels = [
        "Codec",
        "Unit Type",
        "N Units",
        "Amp Base\nMean",
        "Amp Var\nMean",
        "Phase Base\nMean",
        "Phase Var\nMean",
        "Temp Base\nMean",
        "Temp Var\nMean",
        "Amp Base\nMin / Max",
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 1.4)
    for j in range(len(col_labels)):
        header = table[0, j]
        header.set_facecolor("#2c3e50")
        header.set_text_props(color="white", fontweight="bold")

    fig.suptitle(
        "Cross-Codec Perplexity Summary Table\n"
        f"Perplexity = exp(H), per-codec vocab; values aggregated across all available units and signals.",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.92])
    return fig


def _cross_codec_amplitude_sensitivity_bar(
    codec_amp_stats: dict[str, list[dict]],
    bw_tags: dict[str, str],
) -> plt.Figure:
    """Bar chart: mean flip rate across all codebooks/units for each codec,
    for each amplitude level.  Summarises sensitivity in a single figure."""
    codecs = [c for c in codec_amp_stats if codec_amp_stats[c]]
    if not codecs:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return fig

    first_stats = codec_amp_stats[codecs[0]]
    n_variants  = len(first_stats[0].get("variant_tags", [])) if first_stats else 0
    x_labels    = ["0 dBFS"] + [
        AMP_TAG_TO_LABEL.get(int(v), f"−{int(v)} dBFS")
        for v in first_stats[0].get("variant_values", [])
    ] if first_stats else []

    n_x = 1 + n_variants
    x   = np.arange(n_x)
    width = 0.8 / max(len(codecs), 1)

    fig, ax = plt.subplots(figsize=(20, 8))
    fig.suptitle(
        "Cross-Codec: Mean Token Flip Rate vs Amplitude Level\n"
        "(averaged across all codebook/grid units and all frequencies)",
        fontsize=14,
    )

    for ci, codec in enumerate(codecs):
        cfg    = CODEC_CONFIGS.get(codec, {})
        color  = cfg.get("color", "#888")
        label  = cfg.get("label", codec)
        bw     = bw_tags.get(codec, "")
        all_stats = codec_amp_stats[codec]

        mean_flips = np.full(n_x, np.nan)
        cumsum     = np.zeros(n_x)
        cnt        = np.zeros(n_x)

        for stats in all_stats:
            signal      = stats["signal"]
            tokens_root = _codec_root(_DS / "audio_tokens", codec) / "dsp_self_amp" / signal
            base_path   = tokens_root / f"{signal}_baseline_bw{bw}_tokens.npy"
            if not base_path.exists():
                continue
            base_tok = np.load(str(base_path)).astype(np.int64)
            # baseline: 0 flip rate
            cumsum[0] += 0.0
            cnt[0] += 1

            for xi, tag in enumerate(stats.get("variant_tags", []), start=1):
                if xi >= n_x:
                    break
                var_path = tokens_root / f"{signal}_var_{tag}_bw{bw}_tokens.npy"
                if not var_path.exists():
                    continue
                var_tok = np.load(str(var_path)).astype(np.int64)
                n = min(base_tok.shape[1], var_tok.shape[1])
                overall = (base_tok[:, :n] != var_tok[:, :n]).mean()
                cumsum[xi] += float(overall)
                cnt[xi] += 1

        valid = cnt > 0
        mean_flips = np.where(valid, cumsum / np.where(valid, cnt, 1), np.nan)
        offset = (ci - len(codecs) / 2.0 + 0.5) * width
        bars = ax.bar(x + offset, np.nan_to_num(mean_flips), width,
                      label=label, color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlabel("Amplitude level (dBFS)", fontsize=12)
    ax.set_ylabel("Mean token flip rate (all units)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codecs", nargs="+",
                    default=["encodec", "q2d2", "hificodec", "speechtokenizer", "dac_fsq"],
                    choices=list(CODEC_CONFIGS.keys()),
                    help="Codecs to include in the report")
    ap.add_argument("--tokens-root",
                    default=str(_DS / "audio_tokens"),
                    help="Root dir containing codec-specific token subdirs")
    # Per-codec bandwidth tags
    ap.add_argument("--encodec-bw",           default="24.0")
    ap.add_argument("--q2d2-bw",              default="9.8")
    ap.add_argument("--hificodec-bw-tag",         default="HFC")
    ap.add_argument("--speechtokenizer-bw-tag",    default="ST")
    ap.add_argument("--dac-fsq-bw-tag",            default="FSQ")
    ap.add_argument("--output",
                    default=str(_DS / "analysis" / "multi_codec_sensitivity.pdf"),
                    help="Output PDF path")
    args = ap.parse_args()

    tokens_root = Path(args.tokens_root)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bw_tags: dict[str, str] = {
        "encodec":          args.encodec_bw,
        "q2d2":             args.q2d2_bw,
        "hificodec":        args.hificodec_bw_tag,
        "speechtokenizer":  args.speechtokenizer_bw_tag,
        "dac_fsq":          args.dac_fsq_bw_tag,
    }

    codecs = args.codecs
    print(f"\n[Multi-Codec Sensitivity Report]")
    print(f"  Codecs : {codecs}")
    print(f"  Output : {output_path}")

    # Pre-load all stats (so we can cross-reference for summary pages)
    codec_amp_stats:      dict[str, list[dict]] = {}
    codec_phase_0dB:      dict[str, list[dict]] = {}
    codec_temporal:       dict[str, dict]       = {}
    codec_tokens_roots:   dict[str, dict]       = {}

    for codec in codecs:
        bw = bw_tags[codec]
        print(f"\n  Loading {codec} (bw={bw}) …")
        amp_stats = _load_amp_stats(codec, tokens_root, bw)
        _, phase_0dB = _load_phase_stats(codec, tokens_root, bw)
        temp_data = _load_temporal_stats(codec, tokens_root, bw)
        codec_amp_stats[codec]    = amp_stats
        codec_phase_0dB[codec]    = phase_0dB
        codec_temporal[codec]     = temp_data
        codec_tokens_roots[codec] = {
            "amp":      _codec_root(tokens_root, codec) / "dsp_self_amp",
            "phase":    _codec_root(tokens_root, codec) / "dsp_self_phase",
            "temporal": _codec_root(tokens_root, codec) / "time_sine",
        }
        print(f"    amp={len(amp_stats)}  phase_0dB={len(phase_0dB)}  temporal={len(temp_data)}")

    # Build PDF
    n_pages = 0
    print(f"\nBuilding PDF → {output_path}")
    with PdfPages(str(output_path)) as pdf:

        # ── Global title page ──────────────────────────────────────────────
        fig = _global_title_page(codecs, bw_tags)
        pdf.savefig(fig); plt.close(fig); n_pages += 1
        print(f"  [{n_pages}] Global title")

        # ── Per-codec sections ─────────────────────────────────────────────
        for codec in codecs:
            bw          = bw_tags[codec]
            is_q2d2     = _derive_is_q2d2(codec)
            amp_stats   = codec_amp_stats[codec]
            phase_0dB   = codec_phase_0dB[codec]
            temp_data   = codec_temporal[codec]
            amp_root    = codec_tokens_roots[codec]["amp"]
            phase_root  = codec_tokens_roots[codec]["phase"]
            temp_root   = codec_tokens_roots[codec]["temporal"]

            if not amp_stats and not phase_0dB and not temp_data:
                print(f"\n  [{codec}] SKIP — no token data found under {tokens_root/codec}")
                continue

            print(f"\n  [{codec}] Building pages …")

            # Section title page
            fig = _codec_section_title(codec, bw,
                                        len(amp_stats), len(phase_0dB), len(temp_data))
            pdf.savefig(fig); plt.close(fig); n_pages += 1

            # Flip-rate grids
            if amp_stats:
                fig = _per_codebook_amplitude_figure(amp_stats, bw, is_q2d2=is_q2d2)
                pdf.savefig(fig); plt.close(fig); n_pages += 1
                print(f"    [{n_pages}] {codec} amplitude flip rate")

            if phase_0dB:
                fig = _per_codebook_phase_figure(phase_0dB, bw, is_q2d2=is_q2d2)
                pdf.savefig(fig); plt.close(fig); n_pages += 1
                print(f"    [{n_pages}] {codec} phase flip rate")

            if temp_data:
                fig = _per_codebook_temporal_figure(temp_data, bw, is_q2d2=is_q2d2)
                pdf.savefig(fig); plt.close(fig); n_pages += 1
                print(f"    [{n_pages}] {codec} temporal flip rate")

            # Perplexity grids
            vocab_size = CODEC_CONFIGS[codec].get("vocab_size", 1024)

            if amp_stats:
                try:
                    fig = _per_codebook_amplitude_perplexity_figure(
                        amp_stats, amp_root, bw,
                        vocab_size=vocab_size, is_q2d2=is_q2d2)
                    pdf.savefig(fig); plt.close(fig); n_pages += 1
                    print(f"    [{n_pages}] {codec} amplitude perplexity")
                except Exception as e:
                    print(f"    [SKIP] {codec} amplitude perplexity: {e}")

            if phase_0dB:
                try:
                    fig = _per_codebook_phase_perplexity_figure(
                        phase_0dB, phase_root, bw,
                        vocab_size=vocab_size, is_q2d2=is_q2d2)
                    pdf.savefig(fig); plt.close(fig); n_pages += 1
                    print(f"    [{n_pages}] {codec} phase perplexity")
                except Exception as e:
                    print(f"    [SKIP] {codec} phase perplexity: {e}")

            if temp_data:
                try:
                    fig = _per_codebook_temporal_perplexity_figure(
                        temp_data, temp_root, bw,
                        vocab_size=vocab_size, is_q2d2=is_q2d2)
                    pdf.savefig(fig); plt.close(fig); n_pages += 1
                    print(f"    [{n_pages}] {codec} temporal perplexity")
                except Exception as e:
                    print(f"    [SKIP] {codec} temporal perplexity: {e}")

        # ── Cross-codec summary pages ──────────────────────────────────────
        print(f"\n  Building cross-codec summary pages …")

        try:
            fig = _cross_codec_amplitude_sensitivity_bar(codec_amp_stats, bw_tags)
            pdf.savefig(fig); plt.close(fig); n_pages += 1
            print(f"  [{n_pages}] Cross-codec amplitude sensitivity bar chart")
        except Exception as e:
            print(f"  [SKIP] Cross-codec bar chart: {e}")

        try:
            fig = _cross_codec_flip_rate_by_depth(codec_amp_stats, bw_tags)
            pdf.savefig(fig); plt.close(fig); n_pages += 1
            print(f"  [{n_pages}] Cross-codec flip rate by depth")
        except Exception as e:
            print(f"  [SKIP] Cross-codec flip-rate depth: {e}")

        try:
            fig = _cross_codec_perplexity_by_depth(codec_amp_stats, bw_tags)
            pdf.savefig(fig); plt.close(fig); n_pages += 1
            print(f"  [{n_pages}] Cross-codec perplexity by depth")
        except Exception as e:
            print(f"  [SKIP] Cross-codec perplexity depth: {e}")

        try:
            fig = _cross_codec_perplexity_table_figure(
                codecs=codecs,
                codec_amp_stats=codec_amp_stats,
                codec_phase_0dB=codec_phase_0dB,
                codec_temporal=codec_temporal,
                bw_tags=bw_tags,
            )
            pdf.savefig(fig); plt.close(fig); n_pages += 1
            print(f"  [{n_pages}] Cross-codec perplexity summary table")
        except Exception as e:
            print(f"  [SKIP] Cross-codec perplexity summary table: {e}")

    print(f"\nDone — {n_pages} pages written to {output_path}")


if __name__ == "__main__":
    main()

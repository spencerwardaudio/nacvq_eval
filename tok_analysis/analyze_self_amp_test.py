"""Analyze EnCodec token sensitivity to amplitude changes per-frequency.

For each of 20 log-spaced frequencies, the baseline is that frequency at
0 dBFS, and variants are the same frequency at −20 / −40 / −60 / −80 / −100
/ −120 / −140 dBFS.

Produces 3 pages per signal (heatmap, lineplot, summary) PLUS:
  • A cross-frequency overlay page (all 20 frequency curves on one axis)
  • 32 per-codebook subplots (4×8 grid) showing flip rate vs dBFS level (codebooks 1–32)
    with mean ± std across all 20 frequencies, one plot per codebook.

Usage:
    python analyze_self_amp_test.py [tokens_root] [out_root] [bandwidth]
    
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent

FREQ_TAGS: list[int] = sorted({int(round(f)) for f in np.geomspace(10, 20_000, 20)})
SIGNALS: list[str] = [f"self_amp_{f}hz" for f in FREQ_TAGS]  # 20 subdirs, one per log-spaced frequency

# Tag (attenuation magnitude) → display label on x-axis
_BASELINE_ATTEN_DB: int = 0  # reference level
_TAG_TO_LABEL: dict[int, str] = {
    20:  "−20 dBFS",
    40:  "−40 dBFS",
    60:  "−60 dBFS",
    80:  "−80 dBFS",
    100: "−100 dBFS",
    120: "−120 dBFS",
    140: "−140 dBFS",
}

_VAR_RE = re.compile(r"_var_([^_]+)_bw")


# ---------------------------------------------------------------------------
# Shared data helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(str(path)).astype(np.int64, copy=False)


def _flip_rate_per_codebook(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
    n = min(ref.shape[1], other.shape[1])
    return (ref[:, :n] != other[:, :n]).mean(axis=1).astype(np.float64)


def _discover_variants(
    tokens_dir: Path, signal: str, bw: str
) -> list[tuple[float, str, Path]]:
    items: list[tuple[float, str, Path]] = []
    for p in tokens_dir.glob(f"{signal}_var_*_bw{bw}_tokens.npy"):
        m = _VAR_RE.search(p.name)
        if not m:
            continue
        tag = m.group(1)
        try:
            val = float(tag)
        except ValueError:
            continue
        items.append((val, tag, p))
    items.sort(key=lambda t: t[0])
    return items


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_signal(
    signal: str, 
    tokens_dir: Path,
    out_dir: Path,
    bw: str,
) -> dict | None:
    base = tokens_dir / f"{signal}_baseline_bw{bw}_tokens.npy"
    baseline = _load(base)
    if baseline is None:
        print(f"  [SKIP] {signal}: missing baseline")
        return None

    n_cb = baseline.shape[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    repeats: dict[str, dict] = {}
    for r in (1, 2):
        rt = _load(tokens_dir / f"{signal}_repeat_{r}_bw{bw}_tokens.npy")
        if rt is None:
            repeats[f"repeat_{r}"] = {"present": False}
            continue
        n = min(baseline.shape[1], rt.shape[1])
        diff = baseline[:, :n] != rt[:, :n]
        repeats[f"repeat_{r}"] = {
            "present": True,
            "exact": bool(diff.sum() == 0),
            "mismatch_count": int(diff.sum()),
            "per_codebook_flip_rate": diff.mean(axis=1).astype(float).tolist(),
        }

    variants = _discover_variants(tokens_dir, signal, bw)
    flip_by_var: dict[str, list[float]] = {}
    mean_flip_by_var: dict[str, float] = {}
    var_values: list[float] = []
    var_tags: list[str] = []
    for val, tag, p in variants:
        et = _load(p)
        if et is None:
            continue
        per_cb = _flip_rate_per_codebook(baseline, et)
        flip_by_var[tag] = per_cb.tolist()
        mean_flip_by_var[tag] = float(per_cb.mean())
        var_values.append(val)
        var_tags.append(tag)

    # Extract frequency from signal name (e.g. "self_amp_1000hz" → 1000)
    freq_hz = int(signal.split("_")[-1].replace("hz", ""))

    stats = {
        "signal": signal,
        "freq_hz": freq_hz,
        "bandwidth": bw,
        "baseline_atten_db": _BASELINE_ATTEN_DB,
        "n_codebooks": int(n_cb),
        "n_frames": int(baseline.shape[1]),
        "axis_label": "attenuation (dBFS absolute)",
        "variant_values": var_values,
        "variant_tags": var_tags,
        "determinism": repeats,
        "flip_rate_by_variant": flip_by_var,
        "mean_flip_rate_by_variant": mean_flip_by_var,
    }
    
    with (out_dir / f"stats_bw{bw}.json").open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"  [OK] {signal}  freq={freq_hz} Hz  variants={len(var_tags)}")
    return stats


# ---------------------------------------------------------------------------
# Per-signal figures
# ---------------------------------------------------------------------------

def _heatmap_figure(stats: dict) -> plt.Figure:
    n_cb = stats["n_codebooks"]
    rows, labels = [], []
    for r in (1, 2):
        d = stats["determinism"].get(f"repeat_{r}", {})
        if d.get("present"):
            rows.append(np.asarray(d["per_codebook_flip_rate"], dtype=float))
            labels.append(f"repeat_{r}")
    for tag in stats["variant_tags"]:
        rows.append(np.asarray(stats["flip_rate_by_variant"][tag], dtype=float))
        atten = int(float(tag))
        labels.append(_TAG_TO_LABEL.get(atten, f"−{tag} dBFS"))
    if not rows:
        rows = [np.zeros(n_cb)]
        labels = ["(no data)"]
    matrix = np.vstack(rows)
    display = np.where(matrix > 0, matrix, 1e-6)

    fig, ax = plt.subplots(figsize=(max(8.0, n_cb * 0.35), 0.6 * len(labels) + 2.5))
    im = ax.imshow(display, aspect="auto", cmap="magma",
                   norm=LogNorm(vmin=1e-4, vmax=1.0))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(n_cb))
    ax.set_xticklabels([str(i+1) for i in range(n_cb)], fontsize=7)
    ax.set_xlabel("Codebook index")
    freq = stats["freq_hz"]
    ax.set_title(
        f"self_amp — {freq} Hz  per-codebook flip rate vs baseline "
        f"(baseline=0 dBFS, bw={stats['bandwidth']})"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("flip rate (frames differing)")
    fig.tight_layout()
    return fig


def _lineplot_figure(stats: dict) -> plt.Figure:
    n_cb = stats["n_codebooks"]
    var_vals = stats["variant_values"]
    tags = stats["variant_tags"]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    if not var_vals:
        ax.text(0.5, 0.5, "no variant data", ha="center", va="center")
        return fig

    rate_mat = np.vstack(
        [np.asarray(stats["flip_rate_by_variant"][t], dtype=float) for t in tags]
    )
    highlighted = sorted({0, n_cb // 4, n_cb // 2, 3 * n_cb // 4, n_cb - 1})
    for cb in range(n_cb):
        ax.plot(var_vals, rate_mat[:, cb], color="0.85", linewidth=0.7, zorder=1)
    cmap = plt.get_cmap("viridis")
    for i, cb in enumerate(highlighted):
        ax.plot(var_vals, rate_mat[:, cb], marker="o", linewidth=1.8,
                color=cmap(i / max(1, len(highlighted) - 1)),
                label=f"cb {cb+1}", zorder=3)
    ax.plot(var_vals, rate_mat.mean(axis=1), "k--", linewidth=2.0,
            label="mean (all cb)", zorder=4)

    ax.set_xticks(var_vals)
    ax.set_xticklabels(
        [_TAG_TO_LABEL.get(int(v), f"−{int(v)} dBFS") for v in var_vals], fontsize=8
    )
    ax.set_xlabel("amplitude (dBFS)")
    ax.set_ylabel("flip rate")
    freq = stats["freq_hz"]
    ax.set_title(
        f"self_amp — {freq} Hz  flip rate vs amplitude (bw={stats['bandwidth']})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def _summary_figure(stats: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.axis("off")
    freq = stats["freq_hz"]
    det = stats["determinism"]
    lines = [
        f"Signal: {stats['signal']}    bandwidth: {stats['bandwidth']}",
        f"Frequency: {freq} Hz    Baseline: 0 dBFS",
        f"axis: {stats['axis_label']}",
        f"n_codebooks={stats['n_codebooks']}   n_frames={stats['n_frames']}",
        "",
        "Determinism (identical-input repeats vs baseline):",
    ]
    for r in (1, 2):
        d = det.get(f"repeat_{r}", {})
        verdict = ("EXACT MATCH" if d.get("exact")
                   else f"DIFFERS ({d.get('mismatch_count', '?')} mismatches)"
                   if d.get("present") else "MISSING")
        lines.append(f"  repeat_{r}: {verdict}")
    lines += ["", "Flip rate per amplitude level (mean over codebooks; cb0):"]
    lines.append(f"  {'amplitude':>12}   {'mean':>10}   {'cb0':>10}")
    for tag in stats["variant_tags"]:
        atten = int(float(tag))
        label = _TAG_TO_LABEL.get(atten, f"−{atten} dBFS")
        m = stats["mean_flip_rate_by_variant"].get(tag, float("nan"))
        cb0 = stats["flip_rate_by_variant"][tag][0]
        lines.append(f"  {label:>12}   {m:>10.6f}   {cb0:>10.6f}")
    ax.text(0.02, 0.98, "\n".join(lines), ha="left", va="top",
            family="monospace", fontsize=10, transform=ax.transAxes)
    return fig


# ---------------------------------------------------------------------------
# Cross-frequency overlay (final page)
# ---------------------------------------------------------------------------

def _frequency_overlay_figure(all_stats: list[dict], bw: str) -> plt.Figure:
    """One line per frequency: mean flip rate vs attenuation across all 20 freqs."""
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    cmap = plt.get_cmap("plasma")
    n = len(all_stats)

    for i, stats in enumerate(all_stats):
        var_vals = stats["variant_values"]
        means = [stats["mean_flip_rate_by_variant"][t] for t in stats["variant_tags"]]
        freq = stats["freq_hz"]
        color = cmap(i / max(1, n - 1))
        ax.plot(var_vals, means, marker="o", linewidth=1.5, color=color,
                label=f"{freq} Hz")

    # Add example x-tick labels
    if all_stats:
        var_vals = all_stats[0]["variant_values"]
        ax.set_xticks(var_vals)
        ax.set_xticklabels(
            [_TAG_TO_LABEL.get(int(v), f"−{int(v)} dBFS") for v in var_vals],
            fontsize=8,
        )

    ax.set_xlabel("amplitude (dBFS)")
    ax.set_ylabel("mean flip rate (all codebooks)")
    ax.set_title(
        f"Self-amplitude test — flip rate vs level, all frequencies (bw={bw})\n"
        "Each line = one frequency; reveals if high vs low freqs differ in amplitude tolerance"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(
        loc="upper left", fontsize=6, ncol=2, title="Frequency",
        title_fontsize=7,
    )
    fig.tight_layout()
    return fig


def _per_codebook_amplitude_figure(all_stats: list[dict], bw: str, is_q2d2: bool = False) -> plt.Figure:
    """4×8 grid of subplots, one per codebook (1–32) or grid pair (1–16).

    For each codebook/grid pair:
      x-axis: 0 dBFS (baseline, flip=0) + 7 variant levels (−20 to −140 dBFS)
      y-axis: flip rate 0–1
      line: mean across 20 frequencies, shaded ±1 std band
    """
    if not all_stats:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return fig

    n_cb = all_stats[0]["n_codebooks"]

    # Build x-axis: 0 dBFS (baseline) followed by variant levels
    variant_vals_raw = all_stats[0]["variant_values"]   # e.g. [20, 40, ..., 140]
    x_levels = [0.0] + list(variant_vals_raw)            # positive attenuation magnitudes
    x_labels = ["0 dBFS"] + [
        _TAG_TO_LABEL.get(int(v), f"−{int(v)} dBFS") for v in variant_vals_raw
    ]

    # For each (codebook, x-point): collect flip rates across 20 frequencies
    # Shape will be [n_x_points, n_frequencies]
    n_x = len(x_levels)
    n_freqs = len(all_stats)
    data = np.full((n_x, n_freqs, n_cb), np.nan)
    for fi, stats in enumerate(all_stats):
        # x=0 is baseline: flip rate is 0 by definition (compared to itself)
        data[0, fi, :] = 0.0
        for xi, (val, tag) in enumerate(
            zip(stats["variant_values"], stats["variant_tags"]), start=1
        ):
            rates = stats["flip_rate_by_variant"].get(tag)
            if rates is not None:
                cb_len = min(len(rates), n_cb)
                data[xi, fi, :cb_len] = rates[:cb_len]

    # mean / std across frequencies axis (axis=1)
    mean_data = np.nanmean(data, axis=1)   # [n_x, n_cb]
    std_data  = np.nanstd(data,  axis=1)   # [n_x, n_cb]
    var_data  = np.nanvar(data,  axis=1)   # [n_x, n_cb]

    fig, axes = plt.subplots(4, 8, figsize=(28, 12), sharey=True)
    fig.suptitle(
        f"Amplitude Per Codebook Flip Rate\n"
        f"(mean ± std across {n_freqs} frequencies, sine waves, bw={bw} kbps)",
        fontsize=13,
    )

    for cb in range(min(n_cb, 32)):
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

        # Stats box inside subplot
        stats_text = (
            f"mean={y_mean.mean():.3f}\n"
            f"std ={y_std.mean():.3f}\n"
            f"var ={y_var.mean():.3f}"
        )
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                fontsize=6, va="top", ha="left", color="#333",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

        unit_label = f"Grid Pair {cb+1}" if is_q2d2 else f"Codebook {cb+1}"
        ax.set_title(unit_label, fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(range(n_x))
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=6)
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("Flip Rate", fontsize=8)
        if row == 3:
            ax.set_xlabel("Amplitude (dBFS)", fontsize=8)

    # Hide unused subplots if n_cb < 32
    for cb in range(n_cb, 32):
        row, col = divmod(cb, 8)
        axes[row][col].set_visible(False)

    handles, lbls = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower right", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze token sensitivity to amplitude changes",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("tokens_root", type=Path, nargs="?",
                        default=_PROJ_ROOT / "datasets" / "audio_tokens" / "dsp_self_amp",
                        help="Root directory containing token files")
    parser.add_argument("out_root", type=Path, nargs="?",
                        default=_PROJ_ROOT / "datasets" / "analysis" / "dsp_self_amp",
                        help="Output directory for analysis results")
    parser.add_argument("--bandwidth", default="24.0",
                        help="Bandwidth identifier (e.g., 24.0)")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    print(f"Self-amplitude analysis — tokens={args.tokens_root}  out={args.out_root}  bw={args.bandwidth}")

    all_stats: list[dict] = []
    for signal in SIGNALS:
        sub = args.tokens_root / signal
        if not sub.is_dir():
            print(f"  [SKIP] {signal}: directory missing ({sub})")
            continue
        stats = analyze_signal(
            signal, sub, args.out_root / signal, args.bandwidth,
        )
        if stats is not None:
            all_stats.append(stats)

    if not all_stats:
        print("No signals analyzed; nothing to write.")
        sys.exit(1)

    pdf_path = args.out_root / f"results_bw{args.bandwidth}.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(pdf_path)) as pdf:
        for stats in all_stats:
            for builder in (_heatmap_figure, _lineplot_figure, _summary_figure):
                fig = builder(stats)
                pdf.savefig(fig)
                plt.close(fig)
        # Cross-frequency overlay page
        fig = _frequency_overlay_figure(all_stats, args.bandwidth)
        pdf.savefig(fig)
        plt.close(fig)
        # Per-codebook 4x4 grid page
        fig = _per_codebook_amplitude_figure(all_stats, args.bandwidth)
        pdf.savefig(fig)
        plt.close(fig)

    print(f"  [PDF] wrote {len(all_stats) * 3 + 2} pages → {pdf_path}")

    print("\n--- Summary ---")
    for s in all_stats:
        det = s["determinism"]
        det_ok = (det.get("repeat_1", {}).get("exact")
                  and det.get("repeat_2", {}).get("exact"))
        cb0_max = max(
            (s["flip_rate_by_variant"][t][0] for t in s["variant_tags"]),
            default=float("nan"),
        )
        print(f"  {s['signal']:30s}  "
              f"determinism={'OK' if det_ok else 'FAIL':<5}  "
              f"cb0_max_flip={cb0_max:.4f}")


if __name__ == "__main__":
    main()

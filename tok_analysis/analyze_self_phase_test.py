"""Analyze EnCodec token sensitivity to phase changes per-frequency.

For each of 20 log-spaced frequencies at two amplitude levels (0 dBFS and
−70 dBFS), the baseline is 0° phase and variants are 24 phase offsets:
15°, 30°, 45°, …, 360° (every 15°).

Produces:
  • 3 pages per signal (heatmap, lineplot, summary)   — 40 signals × 3 = 120 pages
  • 1 combined overlay per amplitude level (all 20 freqs on one axis)  — 2 pages
  • 1 comparison overlay: 0 dBFS vs −70 dBFS at each representative frequency — 1 page
  • 32 per-codebook subplots (4×8 grid) at 0 dBFS, mean ± std across 20 frequencies (codebooks 1–32)

Usage:
    python analyze_self_phase_test.py [tokens_root] [out_root] [bandwidth]
    
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
PHASE_TAGS: list[int] = list(range(15, 361, 15))  # 15° to 360°; 360° == 0° is a sanity check (should give ~0 flips)

# Two amplitude groups
AMP_GROUPS: list[tuple[str, str]] = [
    ("0dB",  "0 dBFS"),
    ("70dB", "−70 dBFS"),
]

# All 40 signals (0dB first, then 70dB)
SIGNALS: list[str] = [
    f"self_phase_{amp}_{f}hz"
    for amp, _ in AMP_GROUPS
    for f in FREQ_TAGS
]

_VAR_RE = re.compile(r"_var_([^_]+)_bw")


# ---------------------------------------------------------------------------
# Shared helpers
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

    # Parse signal name: self_phase_<amp>_<freq>hz
    parts = signal.split("_")  # ["self", "phase", "0dB", "1000hz"]
    amp_label = parts[2]       # "0dB" or "70dB"
    freq_hz = int(parts[3].replace("hz", ""))

    stats = {
        "signal": signal,
        "freq_hz": freq_hz,
        "amp_label": amp_label,
        "bandwidth": bw,
        "n_codebooks": int(n_cb),
        "n_frames": int(baseline.shape[1]),
        "axis_label": "phase offset (degrees)",
        "variant_values": var_values,
        "variant_tags": var_tags,
        "determinism": repeats,
        "flip_rate_by_variant": flip_by_var,
        "mean_flip_rate_by_variant": mean_flip_by_var,
    }
    
    with (out_dir / f"stats_bw{bw}.json").open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"  [OK] {signal}  {freq_hz} Hz @ {amp_label}  variants={len(var_tags)}")
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
        labels.append(f"{int(float(tag))}°")
    if not rows:
        rows = [np.zeros(n_cb)]
        labels = ["(no data)"]
    matrix = np.vstack(rows)
    display = np.where(matrix > 0, matrix, 1e-6)

    fig, ax = plt.subplots(figsize=(max(8.0, n_cb * 0.35), 0.55 * len(labels) + 2.5))
    im = ax.imshow(display, aspect="auto", cmap="magma",
                   norm=LogNorm(vmin=1e-4, vmax=1.0))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(n_cb))
    ax.set_xticklabels([str(i+1) for i in range(n_cb)], fontsize=7)
    ax.set_xlabel("Codebook index")
    freq = stats["freq_hz"]
    amp = stats["amp_label"]
    ax.set_title(
        f"self_phase — {freq} Hz [{amp}]  per-codebook flip rate vs 0° baseline\n"
        f"(rows = phase offset, bw={stats['bandwidth']})"
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
    ax.set_xticklabels([f"{int(v)}°" for v in var_vals], fontsize=8)
    ax.set_xlabel("phase offset (degrees)")
    ax.set_ylabel("flip rate")
    freq = stats["freq_hz"]
    amp = stats["amp_label"]
    ax.set_title(
        f"self_phase — {freq} Hz [{amp}]  flip rate vs phase offset "
        f"(bw={stats['bandwidth']})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def _summary_figure(stats: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.axis("off")
    det = stats["determinism"]
    lines = [
        f"Signal: {stats['signal']}    bandwidth: {stats['bandwidth']}",
        f"Frequency: {stats['freq_hz']} Hz    Amplitude: {stats['amp_label']}",
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
    lines += ["", "Flip rate per phase offset (mean over codebooks; cb0):"]
    lines.append(f"  {'phase':>8}   {'mean':>10}   {'cb0':>10}")
    for tag in stats["variant_tags"]:
        deg = int(float(tag))
        m = stats["mean_flip_rate_by_variant"].get(tag, float("nan"))
        cb0 = stats["flip_rate_by_variant"][tag][0]
        lines.append(f"  {deg:>7}°   {m:>10.6f}   {cb0:>10.6f}")
    ax.text(0.02, 0.98, "\n".join(lines), ha="left", va="top",
            family="monospace", fontsize=10, transform=ax.transAxes)
    return fig


# ---------------------------------------------------------------------------
# Overlay figures
# ---------------------------------------------------------------------------

def _amp_group_overlay(group_stats: list[dict], amp_label: str, amp_display: str, bw: str) -> plt.Figure:
    """All frequencies overlaid for one amplitude level."""
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    cmap = plt.get_cmap("plasma")
    n = len(group_stats)
    for i, stats in enumerate(group_stats):
        var_vals = stats["variant_values"]
        means = [stats["mean_flip_rate_by_variant"][t] for t in stats["variant_tags"]]
        freq = stats["freq_hz"]
        ax.plot(var_vals, means, marker="o", linewidth=1.5,
                color=cmap(i / max(1, n - 1)), label=f"{freq} Hz")

    if group_stats:
        var_vals = group_stats[0]["variant_values"]
        ax.set_xticks(var_vals)
        ax.set_xticklabels([f"{int(v)}°" for v in var_vals])

    ax.set_xlabel("phase offset (degrees)")
    ax.set_ylabel("mean flip rate (all codebooks)")
    ax.set_title(
        f"Self-phase test [{amp_display}] — flip rate vs phase, all frequencies (bw={bw})\n"
        "Each line = one frequency"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=6, ncol=2, title="Frequency", title_fontsize=7)
    fig.tight_layout()
    return fig


def _amplitude_comparison_overlay(
    stats_0dB: list[dict], stats_70dB: list[dict], bw: str
) -> plt.Figure:
    """Compare 0 dBFS vs −70 dBFS at 4 representative frequencies."""
    # Pick 4 representative frequencies: lowest, ~250 Hz, ~1 kHz, ~8 kHz, highest
    target_freqs = [FREQ_TAGS[0], FREQ_TAGS[5], FREQ_TAGS[12], FREQ_TAGS[18], FREQ_TAGS[-1]]
    lookup_0 = {s["freq_hz"]: s for s in stats_0dB}
    lookup_70 = {s["freq_hz"]: s for s in stats_70dB}

    fig, axes = plt.subplots(1, len(target_freqs), figsize=(3.5 * len(target_freqs), 4.5),
                              sharey=True)
    if len(target_freqs) == 1:
        axes = [axes]

    for ax, freq in zip(axes, target_freqs):
        s0 = lookup_0.get(freq)
        s70 = lookup_70.get(freq)
        for stats, label, color in [
            (s0, "0 dBFS", "steelblue"),
            (s70, "−70 dBFS", "tomato"),
        ]:
            if stats is None:
                continue
            var_vals = stats["variant_values"]
            means = [stats["mean_flip_rate_by_variant"][t] for t in stats["variant_tags"]]
            ax.plot(var_vals, means, marker="o", linewidth=1.5, color=color, label=label)
        ax.set_title(f"{freq} Hz", fontsize=9)
        ax.set_xticks([15, 90, 180, 270, 360])
        ax.set_xticklabels(["15°", "90°", "180°", "270°", "360°"], fontsize=7)
        ax.grid(True, alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("mean flip rate")

    handles, labels_leg = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_leg, loc="upper right", fontsize=8)
    fig.suptitle(
        f"Phase sensitivity: 0 dBFS vs −70 dBFS at representative frequencies (bw={bw})\n"
        "Does amplitude affect phase sensitivity?"
    )
    fig.tight_layout()
    return fig


def _per_codebook_phase_figure(stats_0dB: list[dict], bw: str, is_q2d2: bool = False) -> plt.Figure:
    """4×8 grid of subplots, one per codebook (1–32) or grid pair (1–16).

    Uses 0 dBFS signals only.  For each codebook/grid pair:
      x-axis: 15°, 30°, …, 360° (24 phase values)
      y-axis: flip rate 0–1
      line:  mean across 20 frequencies, shaded ±1 std band + variance annotation
    """
    if not stats_0dB:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no 0 dBFS data", ha="center", va="center")
        return fig

    n_cb = stats_0dB[0]["n_codebooks"]
    phase_vals = stats_0dB[0]["variant_values"]   # [15.0, 30.0, ..., 360.0]
    phase_tags = stats_0dB[0]["variant_tags"]     # ["15", "30", ..., "360"]
    n_x = len(phase_vals)
    n_freqs = len(stats_0dB)

    data = np.full((n_x, n_freqs, n_cb), np.nan)
    for fi, stats in enumerate(stats_0dB):
        for xi, tag in enumerate(phase_tags):
            rates = stats["flip_rate_by_variant"].get(tag)
            if rates is not None:
                cb_len = min(len(rates), n_cb)
                data[xi, fi, :cb_len] = rates[:cb_len]

    mean_data = np.nanmean(data, axis=1)
    std_data  = np.nanstd(data,  axis=1)
    var_data  = np.nanvar(data,  axis=1)

    # Prepend baseline point: 0° phase → flip rate = 0 by definition
    zero_row = np.zeros((1, n_cb))
    mean_data = np.vstack([zero_row, mean_data])
    std_data  = np.vstack([zero_row, std_data])
    var_data  = np.vstack([zero_row, var_data])

    n_cols = 8; n_rows = max(4, (n_cb + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, max(12, n_rows * 3)), sharey=True)
    fig.suptitle(
        f"Phase Per Codebook Flip Rate\n"
        f"(mean ± std across {n_freqs} frequencies at 0 dBFS, sine waves, bw={bw} kbps)",
        fontsize=13,
    )

    x = [0.0] + list(phase_vals)  # prepend 0° baseline
    for cb in range(n_cb):
        row, col = divmod(cb, 8)
        ax = axes[row][col]
        y_mean = mean_data[:, cb]
        y_std  = std_data[:, cb]
        y_var  = var_data[:, cb]

        ax.fill_between(
            x, y_mean - y_std, y_mean + y_std,
            alpha=0.25, color="darkorange", label="±1 std"
        )
        ax.plot(x, y_mean, color="darkorange", linewidth=1.8,
                marker="o", markersize=3, label="mean")

        # Stats exclude the synthetic 0° point (it is always 0)
        stats_text = (
            f"mean={y_mean[1:].mean():.3f}\n"
            f"std ={y_std[1:].mean():.3f}\n"
            f"var ={y_var[1:].mean():.3f}"
        )
        ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
                fontsize=6, va="top", ha="left", color="#333",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

        unit_label = f"Grid Pair {cb+1}" if is_q2d2 else f"Codebook {cb+1}"
        ax.set_title(unit_label, fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        tick_x = [v for v in x if v == 0.0 or (int(v) % 90 == 0)]
        ax.set_xticks(tick_x)
        ax.set_xticklabels([f"{int(v)}°" for v in tick_x], fontsize=7)
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("Flip Rate", fontsize=8)
        if row == n_rows - 1:
            ax.set_xlabel("Phase Offset (degrees)", fontsize=8)

    for cb in range(n_cb, n_rows * n_cols):
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
        description="Analyze token sensitivity to phase changes",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("tokens_root", type=Path, nargs="?",
                        default=_PROJ_ROOT / "datasets" / "audio_tokens" / "dsp_self_phase",
                        help="Root directory containing token files")
    parser.add_argument("out_root", type=Path, nargs="?",
                        default=_PROJ_ROOT / "datasets" / "analysis" / "dsp_self_phase",
                        help="Output directory for analysis results")
    parser.add_argument("--bandwidth", default="24.0",
                        help="Bandwidth identifier (e.g., 24.0)")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    print(f"Self-phase analysis — tokens={args.tokens_root}  out={args.out_root}  bw={args.bandwidth}")

    all_stats: list[dict] = []
    for signal in SIGNALS:
        sub = args.tokens_root / signal
        if not sub.is_dir():
            print(f"  [SKIP] {signal}: directory missing")
            continue
        stats = analyze_signal(
            signal, sub, args.out_root / signal, args.bandwidth,
        )
        if stats is not None:
            all_stats.append(stats)

    if not all_stats:
        print("No signals analyzed; nothing to write.")
        sys.exit(1)

    # Split by amplitude group
    stats_0dB = [s for s in all_stats if s["amp_label"] == "0dB"]
    stats_70dB = [s for s in all_stats if s["amp_label"] == "70dB"]

    pdf_path = args.out_root / f"results_bw{args.bandwidth}.pdf"
    with PdfPages(str(pdf_path)) as pdf:
        for stats in all_stats:
            for builder in (_heatmap_figure, _lineplot_figure, _summary_figure):
                fig = builder(stats)
                pdf.savefig(fig)
                plt.close(fig)
        # Overlay: all freqs at 0 dBFS
        if stats_0dB:
            fig = _amp_group_overlay(stats_0dB, "0dB", "0 dBFS", args.bandwidth)
            pdf.savefig(fig)
            plt.close(fig)
        # Overlay: all freqs at −70 dBFS
        if stats_70dB:
            fig = _amp_group_overlay(stats_70dB, "70dB", "−70 dBFS", args.bandwidth)
            pdf.savefig(fig)
            plt.close(fig)
        # Comparison: 0dB vs 70dB
        if stats_0dB and stats_70dB:
            fig = _amplitude_comparison_overlay(stats_0dB, stats_70dB, args.bandwidth)
            pdf.savefig(fig)
            plt.close(fig)
        # Per-codebook 4×4 grid (0 dBFS)
        if stats_0dB:
            fig = _per_codebook_phase_figure(stats_0dB, args.bandwidth)
            pdf.savefig(fig)
            plt.close(fig)

    extra = (1 if stats_0dB else 0) + (1 if stats_70dB else 0) + (1 if stats_0dB and stats_70dB else 0) + (1 if stats_0dB else 0)
    n_pages = len(all_stats) * 3 + extra
    print(f"  [PDF] wrote {n_pages} pages → {pdf_path}")

    print("\n--- Summary ---")
    for s in all_stats:
        det = s["determinism"]
        det_ok = (det.get("repeat_1", {}).get("exact")
                  and det.get("repeat_2", {}).get("exact"))
        cb0_max = max(
            (s["flip_rate_by_variant"][t][0] for t in s["variant_tags"]),
            default=float("nan"),
        )
        print(f"  {s['signal']:38s}  "
              f"determinism={'OK' if det_ok else 'FAIL':<5}  "
              f"cb0_max_flip={cb0_max:.4f}")


if __name__ == "__main__":
    main()

"""Analyze EnCodec token sensitivity to additive white noise (SNR sweep).

For each of 4 shapes × 20 log-spaced frequencies, the baseline is a clean
signal at −6 dBFS.  Variants add white noise at 5 SNR levels:
  tag 100 → noise 100 dB below signal  (≈ clean)
  tag  80 → noise  80 dB below signal
  tag  60 → noise  60 dB below signal
  tag  40 → noise  40 dB below signal
  tag  20 → noise  20 dB below signal  (heavy noise)

Rather than 80 × 3 = 240 per-signal pages, this analyzer produces concise
summary pages that directly answer the key questions:
  • Per-frequency page: flip rate vs SNR for all 4 shapes overlaid (20 pages)
  • Per-shape page: flip rate vs SNR for 4 representative freqs overlaid (4 pages)
  • SNR threshold heatmap: shape × frequency grid showing at what SNR flip > 5% (1 page)
  • Per-signal detail on request (--detail flag)

Usage:
    python analyze_snr_test.py [tokens_root] [out_root] [bandwidth] [--detail]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Normalize

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent

FREQ_TAGS: list[int] = sorted({int(round(f)) for f in np.geomspace(10, 20_000, 20)})
SHAPES: list[str] = ["sine", "saw", "triangle", "square"]
SNR_TAGS: list[int] = [20, 40, 60, 80, 100]  # ascending SNR → descending noise; 20 dB = heavy noise, 100 dB ≈ clean

SIGNALS: list[str] = [
    f"snr_{shape}_{freq}hz"
    for shape in SHAPES
    for freq in FREQ_TAGS
]

# SNR at which we declare the encoding "degraded"
FLIP_THRESHOLD: float = 0.05  # 5 % flip rate — above this the codec token stream is meaningfully changed by noise

_VAR_RE = re.compile(r"_var_([^_]+)_bw")

# Shape display colors
_SHAPE_COLORS: dict[str, str] = {
    "sine": "steelblue",
    "saw": "tomato",
    "triangle": "forestgreen",
    "square": "darkorange",
}


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


def _snr_threshold(var_vals: list[float], means: list[float],
                   threshold: float = FLIP_THRESHOLD) -> float:
    """Return the highest SNR at which mean flip rate exceeds threshold.
    (Descending SNR = increasing noise, so we want the first breach
    from the high-SNR / clean end.)"""
    # Traverse from high SNR (clean) downward
    snr_threshold = float("nan")
    for snr, rate in sorted(zip(var_vals, means), reverse=True):
        if rate > threshold:
            snr_threshold = snr
    return snr_threshold


# ---------------------------------------------------------------------------
# Core per-signal analysis
# ---------------------------------------------------------------------------

def analyze_signal(
    signal: str, tokens_dir: Path, out_dir: Path, bw: str
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

    # Parse: snr_{shape}_{freq}hz
    parts = signal.split("_")  # ["snr", "sine", "1000hz"] or ["snr", "saw", "20000hz"]
    shape_name = parts[1]
    freq_hz = int(parts[2].replace("hz", ""))

    means = [mean_flip_by_var.get(t, float("nan")) for t in var_tags]
    threshold_snr = _snr_threshold(var_values, means)

    stats = {
        "signal": signal,
        "shape": shape_name,
        "freq_hz": freq_hz,
        "bandwidth": bw,
        "n_codebooks": int(n_cb),
        "n_frames": int(baseline.shape[1]),
        "axis_label": "SNR (dB, noise below signal)",
        "variant_values": var_values,
        "variant_tags": var_tags,
        "determinism": repeats,
        "flip_rate_by_variant": flip_by_var,
        "mean_flip_rate_by_variant": mean_flip_by_var,
        "snr_threshold_db": threshold_snr,   # highest SNR where flip > FLIP_THRESHOLD
    }
    with (out_dir / f"stats_bw{bw}.json").open("w") as f:
        json.dump(stats, f, indent=2)

    print(f"  [OK] {signal}  {shape_name} @ {freq_hz} Hz  "
          f"threshold_SNR={threshold_snr:.0f} dB  variants={len(var_tags)}")
    return stats


# ---------------------------------------------------------------------------
# Per-signal detail figures (optional)
# ---------------------------------------------------------------------------

def _detail_heatmap(stats: dict) -> plt.Figure:
    n_cb = stats["n_codebooks"]
    rows, labels = [], []
    for r in (1, 2):
        d = stats["determinism"].get(f"repeat_{r}", {})
        if d.get("present"):
            rows.append(np.asarray(d["per_codebook_flip_rate"], dtype=float))
            labels.append(f"repeat_{r}")
    for tag in stats["variant_tags"]:
        rows.append(np.asarray(stats["flip_rate_by_variant"][tag], dtype=float))
        labels.append(f"SNR {int(float(tag))} dB")
    if not rows:
        rows = [np.zeros(n_cb)]
        labels = ["(no data)"]
    matrix = np.vstack(rows)
    from matplotlib.colors import LogNorm
    display = np.where(matrix > 0, matrix, 1e-6)
    fig, ax = plt.subplots(figsize=(max(8.0, n_cb * 0.35), 0.55 * len(labels) + 2.5))
    im = ax.imshow(display, aspect="auto", cmap="magma",
                   norm=LogNorm(vmin=1e-4, vmax=1.0))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(n_cb))
    ax.set_xticklabels([str(i) for i in range(n_cb)], fontsize=7)
    ax.set_xlabel("Codebook index")
    ax.set_title(
        f"SNR test — {stats['shape']} @ {stats['freq_hz']} Hz\n"
        f"per-codebook flip rate vs clean baseline (bw={stats['bandwidth']})"
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("flip rate (frames differing)")
    fig.tight_layout()
    return fig


def _detail_lineplot(stats: dict) -> plt.Figure:
    n_cb = stats["n_codebooks"]
    var_vals = stats["variant_values"]   # ascending SNR (100→20 when sorted desc noise)
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
    cmap_v = plt.get_cmap("viridis")
    for i, cb in enumerate(highlighted):
        ax.plot(var_vals, rate_mat[:, cb], marker="o", linewidth=1.8,
                color=cmap_v(i / max(1, len(highlighted) - 1)), label=f"cb {cb}", zorder=3)
    means = rate_mat.mean(axis=1)
    ax.plot(var_vals, means, "k--", linewidth=2.0, label="mean (all cb)", zorder=4)
    ax.axhline(FLIP_THRESHOLD, color="red", linestyle=":", linewidth=1.2,
               label=f"{FLIP_THRESHOLD*100:.0f}% threshold")
    ax.set_xticks(var_vals)
    ax.set_xticklabels([f"{int(v)} dB" for v in var_vals], fontsize=8)
    ax.invert_xaxis()   # left = lowest SNR (noisiest) → right = highest SNR (cleanest)
    ax.set_xlabel("SNR (dB) — left = noisier")
    ax.set_ylabel("flip rate vs clean baseline")
    ax.set_title(
        f"SNR test — {stats['shape']} @ {stats['freq_hz']} Hz  "
        f"(bw={stats['bandwidth']})\n"
        f"threshold SNR = {stats['snr_threshold_db']:.0f} dB"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary figures
# ---------------------------------------------------------------------------

def _per_frequency_figure(
    freq_hz: int, shape_stats: dict[str, dict], bw: str
) -> plt.Figure:
    """Flip rate vs SNR for all 4 shapes at one frequency."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for shape in SHAPES:
        stats = shape_stats.get(shape)
        if stats is None:
            continue
        var_vals = stats["variant_values"]
        means = [stats["mean_flip_rate_by_variant"][t] for t in stats["variant_tags"]]
        ax.plot(var_vals, means, marker="o", linewidth=1.8,
                color=_SHAPE_COLORS[shape], label=shape)
    ax.axhline(FLIP_THRESHOLD, color="red", linestyle=":", linewidth=1.0,
               label=f"{FLIP_THRESHOLD*100:.0f}% threshold")
    ax.invert_xaxis()
    ax.set_xlabel("SNR (dB) — left = noisier, right = cleaner")
    ax.set_ylabel("mean flip rate vs clean baseline")
    ax.set_title(f"SNR tolerance — {freq_hz} Hz, all waveform shapes (bw={bw})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def _per_shape_figure(
    shape: str, freq_stats: dict[int, dict], bw: str,
    representative_freqs: list[int]
) -> plt.Figure:
    """Flip rate vs SNR for representative frequencies at one shape."""
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cmap = plt.get_cmap("plasma")
    valid = [f for f in representative_freqs if f in freq_stats]
    for i, freq in enumerate(valid):
        stats = freq_stats[freq]
        var_vals = stats["variant_values"]
        means = [stats["mean_flip_rate_by_variant"][t] for t in stats["variant_tags"]]
        ax.plot(var_vals, means, marker="o", linewidth=1.8,
                color=cmap(i / max(1, len(valid) - 1)), label=f"{freq} Hz")
    ax.axhline(FLIP_THRESHOLD, color="red", linestyle=":", linewidth=1.0,
               label=f"{FLIP_THRESHOLD*100:.0f}% threshold")
    ax.invert_xaxis()
    ax.set_xlabel("SNR (dB) — left = noisier, right = cleaner")
    ax.set_ylabel("mean flip rate vs clean baseline")
    ax.set_title(f"SNR tolerance — {shape} wave, representative frequencies (bw={bw})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def _threshold_heatmap_figure(
    threshold_grid: np.ndarray, bw: str
) -> plt.Figure:
    """Heatmap: shape × frequency, colour = SNR threshold in dB."""
    fig, ax = plt.subplots(figsize=(14.0, 3.5))
    # NaN where no threshold found (always < threshold)
    display = np.where(np.isnan(threshold_grid), 0.0, threshold_grid)
    im = ax.imshow(display, aspect="auto", cmap="RdYlGn",
                   norm=Normalize(vmin=min(SNR_TAGS), vmax=max(SNR_TAGS)),
                   origin="upper")
    ax.set_yticks(np.arange(len(SHAPES)))
    ax.set_yticklabels(SHAPES)
    ax.set_xticks(np.arange(len(FREQ_TAGS)))
    ax.set_xticklabels([str(f) for f in FREQ_TAGS], rotation=45, ha="right", fontsize=6)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_title(
        f"SNR threshold heatmap — colour = highest SNR (dB) where mean flip rate > "
        f"{FLIP_THRESHOLD*100:.0f}%  (bw={bw})\n"
        "Green = degrades only at low SNR (robust). Red = degrades even at high SNR (sensitive)."
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("SNR threshold (dB)")
    for r, shape in enumerate(SHAPES):
        for c, freq in enumerate(FREQ_TAGS):
            val = threshold_grid[r, c]
            label = f"{val:.0f}" if not np.isnan(val) else "—"
            ax.text(c, r, label, ha="center", va="center", fontsize=5, color="black")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    detail = "--detail" in sys.argv
    argv = [a for a in sys.argv[1:] if a != "--detail"]

    tokens_root = (
        Path(argv[0]) if len(argv) > 0
        else _PROJ_ROOT / "datasets" / "audio_tokens" / "dsp_snr"
    )
    out_root = (
        Path(argv[1]) if len(argv) > 1
        else _PROJ_ROOT / "datasets" / "analysis" / "dsp_snr"
    )
    bw = argv[2] if len(argv) > 2 else "24.0"

    out_root.mkdir(parents=True, exist_ok=True)
    print(f"SNR analysis — tokens={tokens_root}  out={out_root}  bw={bw}")

    all_stats: list[dict] = []
    for signal in SIGNALS:
        sub = tokens_root / signal
        if not sub.is_dir():
            print(f"  [SKIP] {signal}: directory missing")
            continue
        stats = analyze_signal(signal, sub, out_root / signal, bw)
        if stats is not None:
            all_stats.append(stats)

    if not all_stats:
        print("No signals analyzed; nothing to write.")
        sys.exit(1)

    # Build lookup structures
    # by_freq[freq][shape] = stats
    by_freq: dict[int, dict[str, dict]] = {f: {} for f in FREQ_TAGS}
    # by_shape[shape][freq] = stats
    by_shape: dict[str, dict[int, dict]] = {s: {} for s in SHAPES}
    for s in all_stats:
        by_freq[s["freq_hz"]][s["shape"]] = s
        by_shape[s["shape"]][s["freq_hz"]] = s

    # SNR threshold grid [n_shapes, n_freqs]
    threshold_grid = np.full((len(SHAPES), len(FREQ_TAGS)), float("nan"))
    for r, shape in enumerate(SHAPES):
        for c, freq in enumerate(FREQ_TAGS):
            s = by_shape[shape].get(freq)
            if s is not None:
                threshold_grid[r, c] = s["snr_threshold_db"]

    # Representative frequencies for per-shape plots
    rep_freq_indices = [0, 5, 10, 12, 15, 18, 19]
    rep_freqs = [FREQ_TAGS[i] for i in rep_freq_indices if i < len(FREQ_TAGS)]

    pdf_path = out_root / f"results_bw{bw}.pdf"
    with PdfPages(str(pdf_path)) as pdf:
        # Per-signal detail pages (optional)
        if detail:
            for stats in all_stats:
                for builder in (_detail_heatmap, _detail_lineplot):
                    fig = builder(stats)
                    pdf.savefig(fig)
                    plt.close(fig)

        # Per-frequency summary (all 4 shapes overlaid)
        for freq in FREQ_TAGS:
            shape_stats = by_freq.get(freq, {})
            if not shape_stats:
                continue
            fig = _per_frequency_figure(freq, shape_stats, bw)
            pdf.savefig(fig)
            plt.close(fig)

        # Per-shape summary (representative frequencies)
        for shape in SHAPES:
            freq_stats = by_shape.get(shape, {})
            if not freq_stats:
                continue
            fig = _per_shape_figure(shape, freq_stats, bw, rep_freqs)
            pdf.savefig(fig)
            plt.close(fig)

        # SNR threshold heatmap
        fig = _threshold_heatmap_figure(threshold_grid, bw)
        pdf.savefig(fig)
        plt.close(fig)

    total_pages = (
        (len(all_stats) * 2 if detail else 0)
        + len([f for f in FREQ_TAGS if by_freq.get(f)])
        + len([s for s in SHAPES if by_shape.get(s)])
        + 1
    )
    print(f"  [PDF] wrote ~{total_pages} pages → {pdf_path}")

    # Print threshold table
    print(f"\n--- SNR Threshold Table (highest SNR where mean flip > {FLIP_THRESHOLD*100:.0f}%) ---")
    header = f"{'shape':10s}" + "".join(f"{f:>8}" for f in FREQ_TAGS)
    print(header)
    for shape in SHAPES:
        row = f"{shape:10s}"
        for freq in FREQ_TAGS:
            s = by_shape[shape].get(freq)
            val = s["snr_threshold_db"] if s else float("nan")
            row += f"{'—':>8}" if np.isnan(val) else f"{val:>8.0f}"
        print(row)


if __name__ == "__main__":
    main()

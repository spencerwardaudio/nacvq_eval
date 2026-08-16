"""Analyze EnCodec token sensitivity to time offsets for pure sine wave sources.

For each of 20 log-spaced frequencies (10 Hz – 20 kHz) at 0 dBFS, computes
token flip rates at 25 time offsets (1–20 ms every 1 ms + 40/60/80/100 ms).

Produces:
  • 32 per-codebook subplots (4×8 grid) — flip rate vs time offset ms (codebooks 1–32)
    with mean ± std band across all 20 frequencies
  • results_sine_bw<bw>.pdf  containing the above

Usage:
    python analyze_sine_temporal.py [tokens_root] [out_dir] [bandwidth]

    tokens_root: directory containing <freq>hz/ subdirs, each with
                 baseline_0ms_bw<bw>_tokens.npy and offset_*ms_bw<bw>_tokens.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent

# Same frequency grid as all other tests
FREQ_TAGS: list[int] = sorted({int(round(f)) for f in np.geomspace(10, 20_000, 20)})

# Standard offset schedule (ms)
_LINEAR_OFFSETS = list(range(1, 21))  # 1 ms resolution in the perceptually sensitive 0–20 ms range
_LOG_OFFSETS    = [40, 60, 80, 100]  # coarser beyond 20 ms to extend the range without excessive files
ALL_OFFSETS_MS: list[int] = _LINEAR_OFFSETS + _LOG_OFFSETS


# ---------------------------------------------------------------------------
# Token loading helpers (mirror generate_timeoffsets.py naming)
# ---------------------------------------------------------------------------

def _load(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(str(path)).astype(np.int64, copy=False)


def _load_baseline(token_dir: Path, bw: str) -> np.ndarray | None:
    return _load(token_dir / f"baseline_0ms_bw{bw}_tokens.npy")


def _load_offset(token_dir: Path, offset_ms: int, bw: str) -> np.ndarray | None:
    return _load(token_dir / f"offset_{offset_ms:03d}ms_bw{bw}_tokens.npy")


def _flip_rate_per_codebook(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
    n = min(ref.shape[1], other.shape[1])
    return (ref[:, :n] != other[:, :n]).mean(axis=1).astype(np.float64)


# ---------------------------------------------------------------------------
# Per-frequency analysis
# ---------------------------------------------------------------------------

def analyze_freq(
    freq: int,
    token_dir: Path,
    bw: str,
) -> dict[int, np.ndarray] | None:
    """Return {offset_ms: per_codebook_flip_rate_vector} or None if missing."""
    baseline = _load_baseline(token_dir, bw)
    if baseline is None:
        print(f"  [SKIP] {freq} Hz: missing baseline in {token_dir}")
        return None

    result: dict[int, np.ndarray] = {}
    for offset_ms in ALL_OFFSETS_MS:
        tok = _load_offset(token_dir, offset_ms, bw)
        if tok is not None:
            result[offset_ms] = _flip_rate_per_codebook(baseline, tok)
        else:
            print(f"  [WARN] {freq} Hz: missing offset {offset_ms} ms")
    
    return result if result else None


# ---------------------------------------------------------------------------
# Per-codebook figure
# ---------------------------------------------------------------------------

def _per_codebook_temporal_figure(
    all_freq_data: dict[int, dict[int, np.ndarray]],
    bw: str,
    is_q2d2: bool = False,
) -> plt.Figure:
    """4×8 grid of subplots — flip rate vs time offset, one plot per codebook (1–32) or grid pair (1–16).

    For each codebook/grid pair:
      x-axis: time offset (ms) — 25 values (1–20 ms + 40/60/80/100 ms)
      y-axis: flip rate 0–1
      line:  mean across all available frequencies, shaded ±1 std band
      annotation: mean variance
    """
    freqs = sorted(all_freq_data.keys())
    n_freqs = len(freqs)
    if n_freqs == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return fig

    # Determine codebook count from first available datum
    n_cb = next(iter(next(iter(all_freq_data.values())).values())).shape[0]

    # Determine offset list from intersection of all available offsets
    offset_sets = [set(v.keys()) for v in all_freq_data.values() if v]
    common_offsets = sorted(offset_sets[0].intersection(*offset_sets[1:]))
    if not common_offsets:
        common_offsets = ALL_OFFSETS_MS

    n_x = len(common_offsets)

    # data[xi, fi, cb] = flip rate
    data = np.full((n_x, n_freqs, n_cb), np.nan)
    for fi, freq in enumerate(freqs):
        freq_data = all_freq_data[freq]
        if freq_data is None:
            continue
        for xi, offset_ms in enumerate(common_offsets):
            rates = freq_data.get(offset_ms)
            if rates is not None:
                cb_len = min(len(rates), n_cb)
                data[xi, fi, :cb_len] = rates[:cb_len]

    mean_data = np.nanmean(data, axis=1)   # [n_x, n_cb]
    std_data  = np.nanstd(data,  axis=1)
    var_data  = np.nanvar(data,  axis=1)

    # Prepend baseline point: 0 ms offset → flip rate = 0 by definition
    zero_row = np.zeros((1, n_cb))
    mean_data = np.vstack([zero_row, mean_data])
    std_data  = np.vstack([zero_row, std_data])
    var_data  = np.vstack([zero_row, var_data])

    display_offsets = [0] + list(common_offsets)  # prepend 0 ms baseline
    n_x_plot = len(display_offsets)

    n_cols = 8; n_rows = max(4, (n_cb + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, max(12, n_rows * 3)), sharey=True)
    fig.suptitle(
        f"Time Delay Per Codebook Flip Rate\n"
        f"(mean ± std across {n_freqs} sine frequencies, 0 dBFS, bw={bw} kbps)",
        fontsize=13,
    )

    x_labels = [str(v) if v == 0 or (v <= 20 and v % 5 == 0) or v > 20 else ""
                for v in display_offsets]

    for cb in range(n_cb):
        row, col = divmod(cb, 8)
        ax = axes[row][col]
        y_mean = mean_data[:, cb]
        y_std  = std_data[:, cb]
        y_var  = var_data[:, cb]

        ax.fill_between(
            range(n_x_plot), y_mean - y_std, y_mean + y_std,
            alpha=0.25, color="seagreen", label="±1 std"
        )
        ax.plot(range(n_x_plot), y_mean, color="seagreen", linewidth=1.8,
                marker="o", markersize=3, label="mean")

        # Stats exclude the synthetic 0 ms point (it is always 0)
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
        ax.set_xticks(range(n_x_plot))
        ax.set_xticklabels(x_labels, fontsize=6)
        ax.grid(True, alpha=0.25)
        if col == 0:
            ax.set_ylabel("Flip Rate", fontsize=8)
        if row == n_rows - 1:
            ax.set_xlabel("Time Offset (ms)", fontsize=8)

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
        description="Analyze token sensitivity to temporal offsets",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("tokens_root", type=Path, nargs="?",
                        default=_PROJ_ROOT / "datasets" / "audio_tokens" / "time_sine",
                        help="Root directory containing token files")
    parser.add_argument("out_dir", type=Path, nargs="?",
                        default=_PROJ_ROOT / "datasets" / "analysis" / "time_sine",
                        help="Output directory for analysis results")
    parser.add_argument("--bandwidth", default="24.0",
                        help="Bandwidth identifier (e.g., 24.0)")

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sine temporal analysis — tokens={args.tokens_root}  out={args.out_dir}  bw={args.bandwidth}")

    all_freq_data: dict[int, dict[int, np.ndarray] | None] = {}
    for freq in FREQ_TAGS:
        subdir = args.tokens_root / f"{freq}hz"
        if not subdir.is_dir():
            print(f"  [SKIP] {freq} Hz: directory missing ({subdir})")
            continue
        result = analyze_freq(
            freq, subdir, args.bandwidth,
        )
        if result is not None:
            all_freq_data[freq] = result
            print(f"  [OK] {freq} Hz  offsets={len(result)}")

    if not all_freq_data:
        print("No frequency data found; nothing to write.")
        sys.exit(1)

    # Save per-codebook PNG
    fig = _per_codebook_temporal_figure(all_freq_data, args.bandwidth)
    png_path = args.out_dir / f"per_codebook_temporal_sine_bw{args.bandwidth}.png"
    fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PNG] {png_path}")

    # Save PDF
    pdf_path = args.out_dir / f"results_sine_bw{args.bandwidth}.pdf"
    with PdfPages(str(pdf_path)) as pdf:
        fig = _per_codebook_temporal_figure(all_freq_data, args.bandwidth)
        pdf.savefig(fig)
        plt.close(fig)
    print(f"  [PDF] 1 page → {pdf_path}")


if __name__ == "__main__":
    main()

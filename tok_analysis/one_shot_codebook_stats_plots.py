"""Generate one-shot sensitivity proof plots from existing analysis JSON files.

Three error-bar (candlestick) figures that numerically prove model sensitivity:
1) Amplitude  — mean ± std flip rate vs frequency (all attenuation levels averaged).
2) Phase      — mean ± std flip rate vs frequency at two chosen phase angles
                (default: 15° and 180°). Two subplots stacked vertically.
3) Distance   — mean ± std flip rate vs time offset (ms) or RT60 (s).

Y-axis for all plots: flip rate averaged across codebooks, error bars = std across
codebooks.  This shows WHERE in frequency / phase / time the model is most sensitive,
independent of which specific codebook layer carries the effect.

Usage:
    python3 tok_analysis/one_shot_codebook_stats_plots.py
    python3 tok_analysis/one_shot_codebook_stats_plots.py --bandwidth 24.0
    python3 tok_analysis/one_shot_codebook_stats_plots.py --phase-angles 15 180
    python3 tok_analysis/one_shot_codebook_stats_plots.py --temporal-experiment room
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PROJ_ROOT = HERE.parent
DEFAULT_ANALYSIS_ROOT = PROJ_ROOT / "datasets" / "analysis"

# Standard box-and-whisker styling
_BOX_PROPS = dict(
    boxprops=dict(color="black", linewidth=1.2),
    whiskerprops=dict(color="black", linewidth=1.2),
    capprops=dict(color="black", linewidth=1.8),
    medianprops=dict(color="black", linewidth=2.0),
    meanprops=dict(marker="D", markeredgecolor="black", markerfacecolor="white",
                   markersize=5, zorder=4),
    flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor="#888888",
                    markersize=3, alpha=0.6),
    patch_artist=True,
    showmeans=True,
    meanline=False,
    widths=0.5,
    whis=(0, 100),  # whiskers to full min/max, so variance is fully visible
)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_stats(analysis_root: Path, experiment: str, bandwidth: str) -> list[dict]:
    exp_root = analysis_root / experiment
    if not exp_root.exists():
        return []
    stats_name = f"stats_bw{bandwidth}.json"
    out = []
    for stats_path in sorted(exp_root.rglob(stats_name)):
        payload = _load_json(stats_path)
        payload["_stats_path"] = str(stats_path)
        out.append(payload)
    return out


def _pick_temporal_experiment(analysis_root: Path, bandwidth: str, requested: str) -> str | None:
    if requested != "auto":
        return requested
    for exp in ["time", "room", "doa"]:
        if (analysis_root / exp).exists():
            if list((analysis_root / exp).rglob(f"stats_bw{bandwidth}.json")):
                return exp
    return None


# ---------------------------------------------------------------------------
# Shared axis styling
# ---------------------------------------------------------------------------

def _style_ax(ax: plt.Axes, ylabel: str = "Code Book Flip Rate") -> None:
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _boxplot(ax: plt.Axes, data_list: list[np.ndarray], positions: np.ndarray) -> None:
    """Draw a standard box-and-whisker for each position.

    data_list[i] is a 1-D array of per-codebook flip rates for position i.
    Boxes show Q1-Q3, whiskers extend to min/max (full variance), median line
    inside box, mean shown as a white diamond marker.
    """
    bplot = ax.boxplot(data_list, positions=positions, **_BOX_PROPS)
    # Fill boxes white for clean look
    for patch in bplot["boxes"]:
        patch.set_facecolor("white")


# ---------------------------------------------------------------------------
# Plot 1 — Amplitude sensitivity  (averaged across all attenuation levels)
# ---------------------------------------------------------------------------

def _plot_amp_errorbar(records: list[dict], output_path: Path) -> None:
    """Box-and-whisker: flip rate distribution vs frequency.

    All attenuation levels and all 16 codebooks are pooled per frequency.
    Box = Q1-Q3, whiskers = min/max, median line, mean diamond marker.
    """
    by_freq: dict[float, list[np.ndarray]] = {}

    for r in records:
        freq = r.get("freq_hz")
        flip_map = r.get("flip_rate_by_variant", {})
        if not isinstance(freq, (int, float)) or not isinstance(flip_map, dict):
            continue
        for tag, vals in flip_map.items():
            if not isinstance(vals, list) or not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            if arr.ndim == 1 and arr.shape[0] > 0:
                by_freq.setdefault(float(freq), []).append(arr)

    if not by_freq:
        print("[WARN] _plot_amp_errorbar: no data, skipping.")
        return

    sorted_freqs = sorted(by_freq.keys())
    # One 1-D array per frequency: all codebook values across all attenuation levels
    data_list = [np.concatenate(by_freq[f]) for f in sorted_freqs]
    freq_labels = [f"{f:.0f}" if f >= 10 else f"{f:.1f}" for f in sorted_freqs]
    xs = np.arange(len(sorted_freqs))

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    _boxplot(ax, data_list, xs)

    ax.set_xticks(xs)
    ax.set_xticklabels(freq_labels, rotation=55, ha="right", fontsize=8)
    ax.set_xlabel("Frequency (Hz)", fontsize=11)
    _style_ax(ax)
    fig.suptitle(
        "Amplitude & Frequency Code Book Flip Rate",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[OK] Wrote {output_path}")


# ---------------------------------------------------------------------------
# Plot 2 — Phase sensitivity  (two stacked subplots, one per chosen angle)
# ---------------------------------------------------------------------------

def _plot_phase_errorbar(
    records: list[dict],
    angles: list[int],
    output_path: Path,
) -> None:
    """Two stacked candlestick subplots: mean \u00b1 std flip rate vs frequency.

    For each chosen angle, codebooks AND amplitude groups (0 dBFS / -70 dBFS)
    are pooled per frequency to give a model-level mean \u00b1 std datapoint.

    Parameters
    ----------
    angles : list of 2 ints  — phase angles in degrees to plot (e.g. [15, 180])
    """
    by_angle_freq: dict[int, dict[float, list[np.ndarray]]] = {a: {} for a in angles}

    for r in records:
        freq = r.get("freq_hz")
        flip_map = r.get("flip_rate_by_variant", {})
        if not isinstance(freq, (int, float)) or not isinstance(flip_map, dict):
            continue
        for tag, vals in flip_map.items():
            try:
                angle = int(float(tag))
            except (ValueError, TypeError):
                continue
            if angle not in by_angle_freq:
                continue
            if not isinstance(vals, list) or not vals:
                continue
            arr = np.asarray(vals, dtype=float)
            if arr.ndim == 1 and arr.shape[0] > 0:
                by_angle_freq[angle].setdefault(float(freq), []).append(arr)

    fig, axes = plt.subplots(len(angles), 1, figsize=(12, 5 * len(angles)),
                             constrained_layout=True, sharex=False)
    if len(angles) == 1:
        axes = [axes]

    any_data = False
    for ax, angle in zip(axes, angles):
        freq_map = by_angle_freq[angle]
        if not freq_map:
            ax.set_title(f"Phase {angle}\u00b0  [no data]", fontsize=10)
            _style_ax(ax)
            continue

        sorted_freqs = sorted(freq_map.keys())
        # One 1-D array per frequency: all codebook values across both amp groups
        data_list = [np.concatenate(freq_map[f]) for f in sorted_freqs]
        freq_labels = [f"{f:.0f}" if f >= 10 else f"{f:.1f}" for f in sorted_freqs]
        xs = np.arange(len(sorted_freqs))

        _boxplot(ax, data_list, xs)
        ax.set_xticks(xs)
        ax.set_xticklabels(freq_labels, rotation=55, ha="right", fontsize=8)
        ax.set_xlabel(f"Phase {angle}\u00b0  \u2014  Frequency (Hz)", fontsize=10)
        _style_ax(ax)
        any_data = True

    if not any_data:
        print("[WARN] _plot_phase_errorbar: no data for any requested angle, skipping.")
        plt.close(fig)
        return

    fig.suptitle(
        "Phase & Frequency Code Book Flip Rate",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[OK] Wrote {output_path}")


# ---------------------------------------------------------------------------
# Temporal trajectory helper  (unchanged logic)
# ---------------------------------------------------------------------------

def _collect_temporal_trajectory(
    records: list[dict],
    experiment: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]] | None:
    """Build (n_codebooks, n_offsets) matrices averaged across source files.

    For the *room* experiment, only synthetic RT60 conditions (``rt60_0.1s \u2026``)
    are included; the dry ``rt60_0.0s`` and all mesh/SOFA entries are skipped.
    For the *time* experiment, ``baseline_0ms`` is skipped (zero offset).

    Returns (mean_mat, std_mat, var_mat, x_labels).
    """
    by_key: dict[str, list[np.ndarray]] = {}

    def _sort_key(label: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)", label)
        return float(m.group(1)) if m else float("inf")

    for r in records:
        offsets = r.get("offsets", [])
        if not isinstance(offsets, list):
            continue
        for offset in offsets:
            if not isinstance(offset, dict):
                continue
            key = str(offset.get("key") or offset.get("label") or "")
            if not key:
                continue
            if experiment == "room":
                if not key.startswith("rt60_") or key == "rt60_0.0s":
                    continue
            if key in ("baseline_0ms", "offset_000ms"):
                continue
            per_cb = offset.get("per_codebook_rates")
            if not isinstance(per_cb, list) or not per_cb:
                continue
            arr = np.asarray(per_cb, dtype=float)
            if arr.ndim == 1 and arr.shape[0] > 0:
                by_key.setdefault(key, []).append(arr)

    if not by_key:
        return None

    sorted_keys = sorted(by_key.keys(), key=_sort_key)

    def _display(key: str) -> str:
        if key.startswith("rt60_"):
            return key.replace("rt60_", "").replace("s", " s")
        m = re.search(r"(\d+)ms", key)
        return f"{m.group(1)} ms" if m else key

    x_labels = [_display(k) for k in sorted_keys]

    mean_cols, std_cols, var_cols = [], [], []
    for key in sorted_keys:
        stack = np.vstack(by_key[key])  # (n_sources, n_codebooks)
        mean_cols.append(stack.mean(axis=0))
        std_cols.append(stack.std(axis=0))
        var_cols.append(stack.var(axis=0))

    return (
        np.stack(mean_cols, axis=1),   # (n_codebooks, n_offsets)
        np.stack(std_cols, axis=1),
        np.stack(var_cols, axis=1),
        x_labels,
    )


# ---------------------------------------------------------------------------
# Plot 3 — Temporal / distance sensitivity
# ---------------------------------------------------------------------------

def _plot_temporal_errorbar(
    records: list[dict],
    experiment: str,
    output_path: Path,
) -> None:
    """Candlestick: mean \u00b1 std flip rate vs time offset or RT60.

    The 16 codebook means at each offset are reduced to a single model-level
    mean \u00b1 std (std = spread across codebooks at that offset).
    """
    traj = _collect_temporal_trajectory(records, experiment=experiment)
    if traj is None:
        print(f"[WARN] _plot_temporal_errorbar: no usable data for {experiment}, skipping.")
        return

    mean_mat, _std_mat, _var_mat, x_labels = traj
    # mean_mat shape: (n_codebooks, n_offsets)
    # Each column is the 16 per-codebook flip rates at that offset
    # Pass each column as a distribution to boxplot so variance is shown
    data_list = [mean_mat[:, i] for i in range(mean_mat.shape[1])]

    xs = np.arange(len(x_labels))
    x_axis_label = "RT60 (s)" if experiment == "room" else "Time offset (ms)"

    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    _boxplot(ax, data_list, xs)

    max_ticks = 20
    if len(x_labels) > max_ticks:
        idx = np.linspace(0, len(x_labels) - 1, max_ticks).astype(int)
        ax.set_xticks(idx)
        ax.set_xticklabels([x_labels[i] for i in idx], rotation=55, ha="right", fontsize=8)
    else:
        ax.set_xticks(xs)
        ax.set_xticklabels(x_labels, rotation=55, ha="right", fontsize=8)

    ax.set_xlabel(x_axis_label, fontsize=11)
    _style_ax(ax)
    fig.suptitle(
        "Temporal Shift & Frequency Code Book Flip Rate",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"[OK] Wrote {output_path}")



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Error-bar sensitivity plots for amplitude, phase, and temporal/distance."
    )
    parser.add_argument("--analysis-root", type=Path, default=DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to <analysis-root>/final_plots")
    parser.add_argument("--bandwidth", default="24.0",
                        help="Bandwidth tag used in stats filename, e.g. 24.0")
    parser.add_argument("--temporal-experiment", default="auto",
                        choices=["auto", "room", "time", "doa"],
                        help="Source experiment for temporal/distance stats")
    parser.add_argument("--phase-angles", type=int, nargs=2, default=[15, 180],
                        metavar=("ANGLE1", "ANGLE2"),
                        help="Two phase angles (degrees) to plot (default: 15 180)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root = args.analysis_root
    output_dir = args.output_dir or (analysis_root / "final_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Amplitude errorbar  (all levels averaged)
    amp_records = _collect_stats(analysis_root, "dsp_self_amp", args.bandwidth)
    if not amp_records:
        print("[WARN] No self-amplitude stats found. Skipping amplitude plot.")
    else:
        _plot_amp_errorbar(amp_records, output_dir / "amp_sensitivity_errorbar.png")

    # 2) Phase errorbar  (two stacked subplots)
    phase_records = _collect_stats(analysis_root, "dsp_self_phase", args.bandwidth)
    if not phase_records:
        print("[WARN] No self-phase stats found. Skipping phase plot.")
    else:
        _plot_phase_errorbar(
            phase_records,
            args.phase_angles,
            output_dir / "phase_sensitivity_errorbar.png",
        )

    # 3) Temporal / distance errorbar
    temporal_exp = _pick_temporal_experiment(analysis_root, args.bandwidth, args.temporal_experiment)
    if temporal_exp is None:
        print("[WARN] No temporal/distance stats found in time/room/doa. Skipping third plot.")
    else:
        temporal_records = _collect_stats(analysis_root, temporal_exp, args.bandwidth)
        _plot_temporal_errorbar(
            temporal_records,
            temporal_exp,
            output_dir / "temporal_sensitivity_errorbar.png",
        )

    print(f"Done. Output directory: {output_dir}")


if __name__ == "__main__":
    main()

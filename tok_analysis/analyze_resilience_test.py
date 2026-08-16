"""Analyze EnCodec token resilience to circular noise rotation and sine phase offset.

Mirrors analyze_dsp_test.py but with a per-signal "variant axis" instead of eps:
  - rotation signals (*_noise_rot): variant = rotation amount in samples
  - sine_phase: variant = phase offset in degrees

Inputs (per signal subdir under <project_root>/datasets/audio_tokens/dsp_resilience/):
    <name>_baseline_bw<bw>_tokens.npy
    <name>_repeat_1_bw<bw>_tokens.npy
    <name>_repeat_2_bw<bw>_tokens.npy
    <name>_var_<tag>_bw<bw>_tokens.npy   (one per variant value)

Outputs (under <project_root>/datasets/analysis/dsp_resilience/):
    <name>/stats_bw<bw>.json
    <name>/heatmap_bw<bw>.png
    <name>/mean_flip_vs_var_bw<bw>.png
    results_bw<bw>.pdf            ← single combined PDF for all signals

Usage:
    python analyze_resilience_test.py [tokens_root] [output_root] [bandwidth]
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
from matplotlib.colors import LogNorm

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent

# Per-signal axis metadata (axis_label, x_unit, log_x)
SIGNAL_AXES: dict[str, dict] = {
    "white_noise_rot": {"label": "rotation (samples)", "log_x": True},  # log-x because rotation spans 4 orders of magnitude
    "pink_noise_rot":  {"label": "rotation (samples)", "log_x": True},
    "brown_noise_rot": {"label": "rotation (samples)", "log_x": True},
    "sine_phase":      {"label": "phase offset (degrees)", "log_x": False},  # linear — degrees are uniformly spaced
}

SIGNALS = list(SIGNAL_AXES.keys())

_VAR_RE = re.compile(r"_var_([^_]+)_bw")


def _load(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(str(path)).astype(np.int64, copy=False)


def _flip_rate_per_codebook(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
    n_frames = min(ref.shape[1], other.shape[1])
    diff = ref[:, :n_frames] != other[:, :n_frames]
    return diff.mean(axis=1).astype(np.float64)


def _discover_variants(tokens_dir: Path, signal: str, bandwidth: str) -> list[tuple[float, str, Path]]:
    """Find all *_var_<tag>_bw<bw>_tokens.npy files for this signal.
    Returns list of (numeric_value, tag_str, path), sorted by numeric_value.
    """
    items: list[tuple[float, str, Path]] = []
    for p in tokens_dir.glob(f"{signal}_var_*_bw{bandwidth}_tokens.npy"):
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


def analyze_signal(signal: str, tokens_dir: Path, out_dir: Path,
                   bandwidth: str) -> dict | None:
    base = tokens_dir / f"{signal}_baseline_bw{bandwidth}_tokens.npy"
    baseline = _load(base)
    if baseline is None:
        print(f"  [SKIP] {signal}: missing baseline ({base.name})")
        return None

    n_codebooks = baseline.shape[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determinism repeats
    repeats: dict[str, dict] = {}
    for r in (1, 2):
        rp = tokens_dir / f"{signal}_repeat_{r}_bw{bandwidth}_tokens.npy"
        rt = _load(rp)
        if rt is None:
            repeats[f"repeat_{r}"] = {"present": False}
            continue
        n = min(baseline.shape[1], rt.shape[1])
        diff_mask = baseline[:, :n] != rt[:, :n]
        repeats[f"repeat_{r}"] = {
            "present": True,
            "exact": bool(diff_mask.sum() == 0),
            "mismatch_count": int(diff_mask.sum()),
            "per_codebook_flip_rate": diff_mask.mean(axis=1).astype(float).tolist(),
        }

    # Variant sweep
    variants = _discover_variants(tokens_dir, signal, bandwidth)
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

    axis = SIGNAL_AXES.get(signal, {"label": "variant", "log_x": False})
    stats = {
        "signal": signal,
        "bandwidth": bandwidth,
        "n_codebooks": int(n_codebooks),
        "n_frames": int(baseline.shape[1]),
        "axis_label": axis["label"],
        "axis_log_x": axis["log_x"],
        "variant_values": var_values,
        "variant_tags": var_tags,
        "determinism": repeats,
        "flip_rate_by_variant": flip_by_var,
        "mean_flip_rate_by_variant": mean_flip_by_var,
    }

    with (out_dir / f"stats_bw{bandwidth}.json").open("w") as f:
        json.dump(stats, f, indent=2)

    _save_heatmap_png(stats, out_dir / f"heatmap_bw{bandwidth}.png")
    _save_lineplot_png(stats, out_dir / f"mean_flip_vs_var_bw{bandwidth}.png")

    cb0_max = max(
        (flip_by_var[t][0] for t in var_tags), default=float("nan")
    )
    print(f"  [OK] {signal}: variants={len(var_tags)}  "
          f"repeat_1={repeats.get('repeat_1', {}).get('exact')}  "
          f"max_cb0_flip={cb0_max:.4f}")
    return stats


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

def _build_heatmap_data(stats: dict) -> tuple[np.ndarray, list[str]]:
    n_cb = stats["n_codebooks"]
    rows: list[np.ndarray] = []
    labels: list[str] = []
    for r in (1, 2):
        d = stats["determinism"].get(f"repeat_{r}", {})
        if d.get("present"):
            rows.append(np.asarray(d["per_codebook_flip_rate"], dtype=float))
            labels.append(f"repeat_{r}")
    for tag in stats["variant_tags"]:
        rows.append(np.asarray(stats["flip_rate_by_variant"][tag], dtype=float))
        labels.append(f"var={tag}")
    if not rows:
        return np.zeros((1, n_cb)), ["(no data)"]
    return np.vstack(rows), labels


def _heatmap_figure(stats: dict) -> plt.Figure:
    matrix, labels = _build_heatmap_data(stats)
    n_cb = matrix.shape[1]
    fig, ax = plt.subplots(figsize=(max(8.0, n_cb * 0.35), 0.45 * len(labels) + 2.0))
    display = np.where(matrix > 0, matrix, 1e-6)
    im = ax.imshow(display, aspect="auto", cmap="magma",
                   norm=LogNorm(vmin=1e-4, vmax=1.0))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(n_cb))
    ax.set_xticklabels([str(i) for i in range(n_cb)], fontsize=7)
    ax.set_xlabel("Codebook index")
    ax.set_title(f"{stats['signal']} — per-codebook flip rate vs baseline "
                 f"(rows = {stats['axis_label']}, bw={stats['bandwidth']})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("flip rate (frames differing)")
    fig.tight_layout()
    return fig


def _lineplot_figure(stats: dict) -> plt.Figure:
    n_cb = stats["n_codebooks"]
    var_vals = list(stats["variant_values"])
    tags = list(stats["variant_tags"])
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    if not var_vals:
        ax.text(0.5, 0.5, "no variant data", ha="center", va="center")
        return fig

    rate_mat = np.vstack([
        np.asarray(stats["flip_rate_by_variant"][t], dtype=float) for t in tags
    ])  # [n_var, n_cb]

    highlighted = sorted({0, n_cb // 4, n_cb // 2, 3 * n_cb // 4, n_cb - 1})
    for cb in range(n_cb):
        ax.plot(var_vals, rate_mat[:, cb], color="0.85", linewidth=0.7, zorder=1)
    cmap = plt.get_cmap("viridis")
    for i, cb in enumerate(highlighted):
        ax.plot(var_vals, rate_mat[:, cb],
                marker="o", linewidth=1.8,
                color=cmap(i / max(1, len(highlighted) - 1)),
                label=f"cb {cb}", zorder=3)
    ax.plot(var_vals, rate_mat.mean(axis=1), color="black", linewidth=2.0,
            linestyle="--", label="mean (all cb)", zorder=4)

    if stats.get("axis_log_x"):
        ax.set_xscale("log")
    ax.set_xlabel(stats["axis_label"])
    ax.set_ylabel("flip rate")
    ax.set_title(f"{stats['signal']} — flip rate vs {stats['axis_label']} "
                 f"(bw={stats['bandwidth']})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def _summary_figure(stats: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.axis("off")
    lines: list[str] = []
    lines.append(f"Signal: {stats['signal']}    bandwidth: {stats['bandwidth']}")
    lines.append(f"axis: {stats['axis_label']}")
    lines.append(f"n_codebooks={stats['n_codebooks']}   n_frames={stats['n_frames']}")
    lines.append("")
    lines.append("Determinism (identical-input repeats vs baseline):")
    for r in (1, 2):
        d = stats["determinism"].get(f"repeat_{r}", {})
        if not d.get("present"):
            lines.append(f"  repeat_{r}: MISSING")
            continue
        verdict = ("EXACT MATCH" if d["exact"]
                   else f"DIFFERS ({d['mismatch_count']} token mismatches)")
        lines.append(f"  repeat_{r}: {verdict}")
    lines.append("")
    lines.append("Mean flip rate (avg over codebooks & frames) and cb0 flip rate:")
    lines.append(f"  {'variant':>14}   {'mean':>10}   {'cb0':>10}")
    for tag in stats["variant_tags"]:
        m = stats["mean_flip_rate_by_variant"].get(tag, float("nan"))
        cb0 = stats["flip_rate_by_variant"][tag][0]
        lines.append(f"  {tag:>14}   {m:>10.6f}   {cb0:>10.6f}")
    text = "\n".join(lines)
    ax.text(0.02, 0.98, text, ha="left", va="top",
            family="monospace", fontsize=10, transform=ax.transAxes)
    return fig


def _save_heatmap_png(stats: dict, out_path: Path) -> None:
    fig = _heatmap_figure(stats)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_lineplot_png(stats: dict, out_path: Path) -> None:
    fig = _lineplot_figure(stats)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _write_combined_pdf(all_stats: list[dict], pdf_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(pdf_path)) as pdf:
        for stats in all_stats:
            for builder in (_heatmap_figure, _lineplot_figure, _summary_figure):
                fig = builder(stats)
                pdf.savefig(fig)
                plt.close(fig)
    print(f"  [PDF] wrote combined results → {pdf_path}")


def main() -> None:
    tokens_root = Path(sys.argv[1]) if len(sys.argv) > 1 \
        else _PROJ_ROOT / "datasets" / "audio_tokens" / "dsp_resilience"
    out_root = Path(sys.argv[2]) if len(sys.argv) > 2 \
        else _PROJ_ROOT / "datasets" / "analysis" / "dsp_resilience"
    bandwidth = sys.argv[3] if len(sys.argv) > 3 else "24.0"

    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Resilience analysis — tokens={tokens_root} out={out_root} bw={bandwidth}")

    all_stats: list[dict] = []
    for signal in SIGNALS:
        sub = tokens_root / signal
        if not sub.is_dir():
            print(f"  [SKIP] {signal}: directory missing ({sub})")
            continue
        stats = analyze_signal(signal, sub, out_root / signal, bandwidth)
        if stats is not None:
            all_stats.append(stats)

    if not all_stats:
        print("No signals analyzed; nothing to write.")
        sys.exit(1)

    _write_combined_pdf(all_stats, out_root / f"results_bw{bandwidth}.pdf")

    print("\n--- Summary ---")
    for s in all_stats:
        det = s["determinism"]
        r1 = det.get("repeat_1", {})
        r2 = det.get("repeat_2", {})
        verdict = "yes" if (r1.get("exact") and r2.get("exact")) else "no"
        if s["variant_tags"]:
            cb0_max = max(s["flip_rate_by_variant"][t][0] for t in s["variant_tags"])
            mean_max = max(s["mean_flip_rate_by_variant"].values())
        else:
            cb0_max = mean_max = float("nan")
        print(f"  {s['signal']:>18}  DETERMINISTIC={verdict}  "
              f"max_cb0_flip={cb0_max:.4f}  max_mean_flip={mean_max:.4f}")


if __name__ == "__main__":
    main()

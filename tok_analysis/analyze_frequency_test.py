"""Analyze EnCodec token sensitivity to frequency changes (sine sweep).

Three subdirs — one per amplitude level — each containing 20 frequency variants.
For each subdir, the baseline is a 1000 Hz sine at that amplitude.

Inputs (under <tokens_root>/sine_freq_<amp>dB/ for each amplitude):
    sine_freq_<amp>dB_baseline_bw<bw>_tokens.npy
    sine_freq_<amp>dB_repeat_{1,2}_bw<bw>_tokens.npy
    sine_freq_<amp>dB_var_<hz>_bw<bw>_tokens.npy   hz ∈ {10,15,22,...,20000}

Outputs (under <out_root>/):
    sine_freq_<amp>dB/heatmap_bw<bw>.png
    sine_freq_<amp>dB/mean_flip_vs_var_bw<bw>.png
    sine_freq_<amp>dB/stats_bw<bw>.json
    results_bw<bw>.pdf  ← combined PDF across all three amplitude levels

Usage:
    python analyze_frequency_test.py [tokens_root] [out_root] [bandwidth]
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

SIGNALS = ["sine_freq_0dB", "sine_freq_70dB", "sine_freq_140dB"]  # three amplitude levels from the generator
_SIGNAL_AMP_LABELS = {
    "sine_freq_0dB": "0 dBFS",
    "sine_freq_70dB": "−70 dBFS",
    "sine_freq_140dB": "−140 dBFS (≈ silence)",
}

_VAR_RE = re.compile(r"_var_([^_]+)_bw")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(str(path)).astype(np.int64, copy=False)


def _flip_rate_per_codebook(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
    n = min(ref.shape[1], other.shape[1])  # trim to shorter to handle segment boundary differences
    return (ref[:, :n] != other[:, :n]).mean(axis=1).astype(np.float64)


def _discover_variants(tokens_dir: Path, signal: str, bw: str) -> list[tuple[float, str, Path]]:
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

def analyze_signal(signal: str, tokens_dir: Path, out_dir: Path, bw: str) -> dict | None:
    base = tokens_dir / f"{signal}_baseline_bw{bw}_tokens.npy"
    baseline = _load(base)
    if baseline is None:
        print(f"  [SKIP] {signal}: missing baseline ({base.name})")
        return None

    n_cb = baseline.shape[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    repeats: dict[str, dict] = {}
    for r in (1, 2):
        rp = tokens_dir / f"{signal}_repeat_{r}_bw{bw}_tokens.npy"
        rt = _load(rp)
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

    amp_label = _SIGNAL_AMP_LABELS.get(signal, signal)
    stats = {
        "signal": signal,
        "amplitude_label": amp_label,
        "bandwidth": bw,
        "n_codebooks": int(n_cb),
        "n_frames": int(baseline.shape[1]),
        "axis_label": "frequency (Hz)",
        "axis_log_x": True,
        "variant_values": var_values,
        "variant_tags": var_tags,
        "determinism": repeats,
        "flip_rate_by_variant": flip_by_var,
        "mean_flip_rate_by_variant": mean_flip_by_var,
    }
    with (out_dir / f"stats_bw{bw}.json").open("w") as f:
        json.dump(stats, f, indent=2)

    _save_heatmap_png(stats, out_dir / f"heatmap_bw{bw}.png")
    _save_lineplot_png(stats, out_dir / f"mean_flip_vs_var_bw{bw}.png")

    cb0_max = max((flip_by_var[t][0] for t in var_tags), default=float("nan"))
    print(f"  [OK] {signal}: variants={len(var_tags)}  "
          f"repeat_1={repeats.get('repeat_1', {}).get('exact')}  "
          f"max_cb0_flip={cb0_max:.4f}")
    return stats


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _build_heatmap_data(stats: dict) -> tuple[np.ndarray, list[str]]:
    n_cb = stats["n_codebooks"]
    rows, labels = [], []
    for r in (1, 2):
        d = stats["determinism"].get(f"repeat_{r}", {})
        if d.get("present"):
            rows.append(np.asarray(d["per_codebook_flip_rate"], dtype=float))
            labels.append(f"repeat_{r}")
    for tag in stats["variant_tags"]:
        rows.append(np.asarray(stats["flip_rate_by_variant"][tag], dtype=float))
        hz = int(float(tag))
        labels.append(f"{hz} Hz")
    if not rows:
        return np.zeros((1, n_cb)), ["(no data)"]
    return np.vstack(rows), labels


def _heatmap_figure(stats: dict) -> plt.Figure:
    matrix, labels = _build_heatmap_data(stats)
    n_cb = matrix.shape[1]
    fig, ax = plt.subplots(figsize=(max(8.0, n_cb * 0.35), 0.38 * len(labels) + 2.0))
    display = np.where(matrix > 0, matrix, 1e-6)
    im = ax.imshow(display, aspect="auto", cmap="magma",
                   norm=LogNorm(vmin=1e-4, vmax=1.0))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xticks(np.arange(n_cb))
    ax.set_xticklabels([str(i) for i in range(n_cb)], fontsize=7)
    ax.set_xlabel("Codebook index")
    ax.set_title(f"{stats['signal']} [{stats['amplitude_label']}]\n"
                 f"per-codebook flip rate vs baseline  (bw={stats['bandwidth']})")
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

    rate_mat = np.vstack([
        np.asarray(stats["flip_rate_by_variant"][t], dtype=float) for t in tags
    ])  # [n_var, n_cb]

    highlighted = sorted({0, n_cb // 4, n_cb // 2, 3 * n_cb // 4, n_cb - 1})
    for cb in range(n_cb):
        ax.plot(var_vals, rate_mat[:, cb], color="0.85", linewidth=0.7, zorder=1)
    cmap = plt.get_cmap("viridis")
    for i, cb in enumerate(highlighted):
        ax.plot(var_vals, rate_mat[:, cb], marker="o", linewidth=1.8,
                color=cmap(i / max(1, len(highlighted) - 1)),
                label=f"cb {cb}", zorder=3)
    ax.plot(var_vals, rate_mat.mean(axis=1), color="black", linewidth=2.0,
            linestyle="--", label="mean (all cb)", zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("flip rate")
    ax.set_title(f"{stats['signal']} [{stats['amplitude_label']}]\n"
                 f"flip rate vs frequency  (bw={stats['bandwidth']})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def _summary_figure(stats: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.axis("off")
    lines = [
        f"Signal: {stats['signal']}    bandwidth: {stats['bandwidth']}",
        f"Amplitude: {stats['amplitude_label']}",
        f"axis: {stats['axis_label']}  (log scale)",
        f"n_codebooks={stats['n_codebooks']}   n_frames={stats['n_frames']}",
        "",
        "Determinism (identical-input repeats vs baseline):",
    ]
    for r in (1, 2):
        d = stats["determinism"].get(f"repeat_{r}", {})
        if not d.get("present"):
            lines.append(f"  repeat_{r}: MISSING")
            continue
        verdict = "EXACT MATCH" if d["exact"] else f"DIFFERS ({d['mismatch_count']} mismatches)"
        lines.append(f"  repeat_{r}: {verdict}")
    lines += ["", "Flip rate per frequency variant (mean over codebooks; cb0):"]
    lines.append(f"  {'freq (Hz)':>12}   {'mean':>10}   {'cb0':>10}")
    for tag in stats["variant_tags"]:
        m = stats["mean_flip_rate_by_variant"].get(tag, float("nan"))
        cb0 = stats["flip_rate_by_variant"][tag][0]
        lines.append(f"  {int(float(tag)):>12}   {m:>10.6f}   {cb0:>10.6f}")
    ax.text(0.02, 0.98, "\n".join(lines), ha="left", va="top",
            family="monospace", fontsize=9, transform=ax.transAxes)
    return fig


def _save_heatmap_png(stats: dict, path: Path) -> None:
    fig = _heatmap_figure(stats)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_lineplot_png(stats: dict, path: Path) -> None:
    fig = _lineplot_figure(stats)
    fig.savefig(path, dpi=150)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    tokens_root = (Path(sys.argv[1]) if len(sys.argv) > 1
                   else _PROJ_ROOT / "datasets" / "audio_tokens" / "dsp_frequency")
    out_root = (Path(sys.argv[2]) if len(sys.argv) > 2
                else _PROJ_ROOT / "datasets" / "analysis" / "dsp_frequency")
    bw = sys.argv[3] if len(sys.argv) > 3 else "24.0"

    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Frequency analysis — tokens={tokens_root}  out={out_root}  bw={bw}")

    all_stats: list[dict] = []
    for signal in SIGNALS:
        sub = tokens_root / signal
        if not sub.is_dir():
            print(f"  [SKIP] {signal}: directory missing ({sub})")
            continue
        stats = analyze_signal(signal, sub, out_root / signal, bw)
        if stats is not None:
            all_stats.append(stats)

    if not all_stats:
        print("No signals analyzed; nothing to write.")
        sys.exit(1)

    _write_combined_pdf(all_stats, out_root / f"results_bw{bw}.pdf")

    print("\n--- Summary ---")
    for s in all_stats:
        det = s["determinism"]
        det_ok = (det.get("repeat_1", {}).get("exact")
                  and det.get("repeat_2", {}).get("exact"))
        cb0_max = max((s["flip_rate_by_variant"][t][0] for t in s["variant_tags"]),
                      default=float("nan"))
        mean_max = max(s["mean_flip_rate_by_variant"].values(), default=float("nan"))
        print(f"  {s['signal']:>22}  DETERMINISTIC={'yes' if det_ok else 'no'}  "
              f"max_cb0_flip={cb0_max:.4f}  max_mean_flip={mean_max:.4f}")


if __name__ == "__main__":
    main()

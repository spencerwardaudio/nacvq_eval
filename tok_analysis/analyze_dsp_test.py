"""Analyze EnCodec token determinism and roundoff sensitivity for DSP test signals.

Inputs (per signal subdir under <project_root>/datasets/audio_tokens/dsp/):
    <signal>_baseline_bw<bw>_tokens.npy
    <signal>_repeat_1_bw<bw>_tokens.npy
    <signal>_repeat_2_bw<bw>_tokens.npy
    <signal>_eps_<mag>_bw<bw>_tokens.npy   (one per perturbation magnitude)

Outputs (under <project_root>/datasets/analysis/dsp/):
    <signal>/stats_bw<bw>.json
    <signal>/heatmap_bw<bw>.png
    <signal>/mean_flip_vs_eps_bw<bw>.png
    results_bw<bw>.pdf            ← single combined PDF for all signals (results only)

Usage:
    python analyze_dsp_test.py [tokens_root] [output_root] [bandwidth]

Defaults:
    tokens_root = <project_root>/datasets/audio_tokens/dsp
    output_root = <project_root>/datasets/analysis/dsp
    bandwidth   = 24.0
"""

from __future__ import annotations

import json
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

SIGNALS = ["sine_1k", "impulse", "white_noise", "chirp"]
EPS_LIST = [1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2]


def _eps_tag(eps: float) -> str:
    return f"{eps:g}"


def _load(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.load(str(path))
    return arr.astype(np.int64, copy=False)


def _flip_rate_per_codebook(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Return per-codebook fraction of frames where tokens differ. Shape: [n_codebooks]."""
    n_frames = min(ref.shape[1], other.shape[1])  # min to handle segment boundary length differences
    diff = ref[:, :n_frames] != other[:, :n_frames]
    return diff.mean(axis=1).astype(np.float64)


def analyze_signal(signal: str, tokens_dir: Path, out_dir: Path,
                   bandwidth: str) -> dict | None:
    """Compute determinism + per-codebook flip rates. Returns stats dict or None."""
    base = tokens_dir / f"{signal}_baseline_bw{bandwidth}_tokens.npy"
    baseline = _load(base)
    if baseline is None:
        print(f"  [SKIP] {signal}: missing baseline ({base.name})")
        return None

    n_codebooks = baseline.shape[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determinism: identical-input repeats
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

    # Roundoff sweep
    flip_by_eps: dict[str, list[float]] = {}
    mean_flip_by_eps: dict[str, float] = {}
    for eps in EPS_LIST:  # eps_list is ascending; expect near-zero flip at small eps, rising at large eps
        tag = _eps_tag(eps)
        ep = tokens_dir / f"{signal}_eps_{tag}_bw{bandwidth}_tokens.npy"
        et = _load(ep)
        if et is None:
            print(f"  [WARN] {signal}: missing {ep.name}")
            continue
        per_cb = _flip_rate_per_codebook(baseline, et)
        flip_by_eps[tag] = per_cb.tolist()
        mean_flip_by_eps[tag] = float(per_cb.mean())

    stats = {
        "signal": signal,
        "bandwidth": bandwidth,
        "n_codebooks": int(n_codebooks),
        "n_frames": int(baseline.shape[1]),
        "determinism": repeats,
        "flip_rate_by_eps": flip_by_eps,
        "mean_flip_rate_by_eps": mean_flip_by_eps,
    }

    with (out_dir / f"stats_bw{bandwidth}.json").open("w") as f:
        json.dump(stats, f, indent=2)

    # Per-signal PNGs
    _save_heatmap_png(stats, out_dir / f"heatmap_bw{bandwidth}.png")
    _save_lineplot_png(stats, out_dir / f"mean_flip_vs_eps_bw{bandwidth}.png")

    print(f"  [OK] {signal}: determinism repeat_1={repeats.get('repeat_1', {}).get('exact')} "
          f"repeat_2={repeats.get('repeat_2', {}).get('exact')} "
          f"mean_flip@1e-2={mean_flip_by_eps.get('0.01', float('nan')):.4f}")
    return stats


# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

def _build_heatmap_data(stats: dict) -> tuple[np.ndarray, list[str]]:
    """Return (matrix [n_rows, n_codebooks], row_labels)."""
    n_cb = stats["n_codebooks"]
    rows: list[np.ndarray] = []
    labels: list[str] = []
    # Repeats first (sanity)
    for r in (1, 2):
        d = stats["determinism"].get(f"repeat_{r}", {})
        if d.get("present"):
            rows.append(np.asarray(d["per_codebook_flip_rate"], dtype=float))
            labels.append(f"repeat_{r}")
    # Then eps in ascending magnitude
    for eps in EPS_LIST:
        tag = _eps_tag(eps)
        if tag in stats["flip_rate_by_eps"]:
            rows.append(np.asarray(stats["flip_rate_by_eps"][tag], dtype=float))
            labels.append(f"eps={tag}")
    if not rows:
        return np.zeros((1, n_cb)), ["(no data)"]
    return np.vstack(rows), labels


def _heatmap_figure(stats: dict) -> plt.Figure:
    matrix, labels = _build_heatmap_data(stats)
    n_cb = matrix.shape[1]
    fig, ax = plt.subplots(figsize=(max(8.0, n_cb * 0.35), 0.45 * len(labels) + 2.0))
    # LogNorm chokes on zeros; clamp display floor
    display = np.where(matrix > 0, matrix, 1e-6)
    im = ax.imshow(
        display, aspect="auto", cmap="magma",
        norm=LogNorm(vmin=1e-4, vmax=1.0),
    )
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(n_cb))
    ax.set_xticklabels([str(i) for i in range(n_cb)], fontsize=7)
    ax.set_xlabel("Codebook index")
    ax.set_title(f"{stats['signal']} — per-codebook flip rate vs baseline (bw={stats['bandwidth']})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("flip rate (frames differing)")
    fig.tight_layout()
    return fig


def _lineplot_figure(stats: dict) -> plt.Figure:
    n_cb = stats["n_codebooks"]
    eps_vals = []
    rates: list[np.ndarray] = []
    for eps in EPS_LIST:
        tag = _eps_tag(eps)
        if tag in stats["flip_rate_by_eps"]:
            eps_vals.append(eps)
            rates.append(np.asarray(stats["flip_rate_by_eps"][tag], dtype=float))
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    if not eps_vals:
        ax.text(0.5, 0.5, "no eps data", ha="center", va="center")
        return fig

    rate_mat = np.vstack(rates)  # [n_eps, n_cb]
    # Highlighted codebooks: 0, n/4, n/2, 3n/4, n-1 (deduped)
    highlighted = sorted({0, n_cb // 4, n_cb // 2, 3 * n_cb // 4, n_cb - 1})
    # All codebooks faint
    for cb in range(n_cb):
        ax.plot(eps_vals, rate_mat[:, cb], color="0.85", linewidth=0.7, zorder=1)
    # Highlighted on top
    cmap = plt.get_cmap("viridis")
    for i, cb in enumerate(highlighted):
        ax.plot(eps_vals, rate_mat[:, cb],
                marker="o", linewidth=1.8,
                color=cmap(i / max(1, len(highlighted) - 1)),
                label=f"cb {cb}", zorder=3)
    # Mean across codebooks
    ax.plot(eps_vals, rate_mat.mean(axis=1), color="black", linewidth=2.0,
            linestyle="--", label="mean (all cb)", zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("perturbation magnitude (eps)")
    ax.set_ylabel("flip rate")
    ax.set_title(f"{stats['signal']} — flip rate vs roundoff magnitude (bw={stats['bandwidth']})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def _summary_figure(stats: dict) -> plt.Figure:
    """Text-only matplotlib figure: determinism verdict + mean-flip mini-table."""
    fig, ax = plt.subplots(figsize=(8.5, 6.0))
    ax.axis("off")
    lines: list[str] = []
    lines.append(f"Signal: {stats['signal']}    bandwidth: {stats['bandwidth']}")
    lines.append(f"n_codebooks={stats['n_codebooks']}   n_frames={stats['n_frames']}")
    lines.append("")
    lines.append("Determinism (identical-input repeats vs baseline):")
    for r in (1, 2):
        d = stats["determinism"].get(f"repeat_{r}", {})
        if not d.get("present"):
            lines.append(f"  repeat_{r}: MISSING")
            continue
        verdict = "EXACT MATCH" if d["exact"] else f"DIFFERS ({d['mismatch_count']} token mismatches)"
        lines.append(f"  repeat_{r}: {verdict}")
    lines.append("")
    lines.append("Mean flip rate (avg over codebooks & frames) vs perturbation eps:")
    lines.append(f"  {'eps':>10}   {'mean flip rate':>16}")
    for eps in EPS_LIST:
        tag = _eps_tag(eps)
        if tag in stats["mean_flip_rate_by_eps"]:
            v = stats["mean_flip_rate_by_eps"][tag]
            lines.append(f"  {tag:>10}   {v:>16.6f}")
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


# ---------------------------------------------------------------------------
# Combined PDF (all signals, results only)
# ---------------------------------------------------------------------------

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
    tokens_root = Path(sys.argv[1]) if len(sys.argv) > 1 \
        else _PROJ_ROOT / "datasets" / "audio_tokens" / "dsp"
    out_root = Path(sys.argv[2]) if len(sys.argv) > 2 \
        else _PROJ_ROOT / "datasets" / "analysis" / "dsp"
    bandwidth = sys.argv[3] if len(sys.argv) > 3 else "24.0"

    out_root.mkdir(parents=True, exist_ok=True)
    print(f"DSP analysis — tokens={tokens_root} out={out_root} bw={bandwidth}")

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

    # Stdout summary
    print("\n--- Summary ---")
    for s in all_stats:
        det = s["determinism"]
        r1 = det.get("repeat_1", {})
        r2 = det.get("repeat_2", {})
        verdict = "yes" if (r1.get("exact") and r2.get("exact")) else "no"
        m1e2 = s["mean_flip_rate_by_eps"].get("0.01")
        m1e10 = s["mean_flip_rate_by_eps"].get("1e-10")
        print(f"  {s['signal']:>12}  DETERMINISTIC={verdict}  "
              f"mean_flip[1e-10]={m1e10}  mean_flip[1e-2]={m1e2}")


if __name__ == "__main__":
    main()

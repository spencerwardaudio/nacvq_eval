"""Aggregate per-file token flip stats into a professional consolidated PDF.

Reads all stats_bw*.json files from analysis/{experiment}/*/ directories,
computes cross-file means and standard deviations, and produces a single
publication-quality PDF per experiment.

Pages:
  1. Cover page — experiment metadata, file count, date
  2. Summary table — mean flip rate per perturbation level (Timbru-style)
  3. Per-codebook sensitivity — mean ± std bar chart across all files
  4. Aggregate water cube surface — mean flip matrix across all sources
  5. Per-genre heatmap — flip rate by genre × codebook (if GTZAN naming)
  6. Variance analysis — which codebooks are most/least stable

Usage:
    python aggregate_analysis.py --experiment time [--bandwidth 24.0]
    python aggregate_analysis.py --experiment room
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
_DATASETS_DIR = _PROJ_ROOT / "datasets"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_stats(experiment: str, bandwidth: str = "24.0") -> list[dict]:
    """Load all per-file stats JSON files for an experiment."""
    analysis_dir = _DATASETS_DIR / "analysis" / experiment
    if not analysis_dir.exists():
        print(f"Analysis directory not found: {analysis_dir}")
        return []

    stats = []
    for stats_file in sorted(analysis_dir.rglob(f"stats_bw{bandwidth}.json")):
        with open(stats_file) as f:
            data = json.load(f)
        data["_path"] = str(stats_file)
        stats.append(data)

    return stats


def _extract_genre(source_name: str) -> str:
    """Extract genre from GTZAN-style naming (e.g., 'blues_00' → 'blues')."""
    parts = source_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return "unknown"


# ---------------------------------------------------------------------------
# Aggregate computations
# ---------------------------------------------------------------------------

def compute_aggregates(stats: list[dict]) -> dict:
    """Compute cross-file aggregate statistics."""
    if not stats:
        return {}

    # Get perturbation keys from first file
    first = stats[0]
    n_codebooks = first["n_codebooks"]
    offset_keys = [o["key"] for o in first["offsets"]]
    offset_labels = [o["label"] for o in first["offsets"]]

    n_files = len(stats)
    n_offsets = len(offset_keys)

    # Collect matrices: (n_files, n_offsets) for total flip rate
    # and (n_files, n_offsets, n_codebooks) for per-codebook rates
    total_rates = np.zeros((n_files, n_offsets))  # shape: [files, conditions]
    cb_rates = np.zeros((n_files, n_offsets, n_codebooks))  # shape: [files, conditions, codebooks]
    max_frame_rates = np.zeros((n_files, n_offsets))

    genres = []
    source_names = []

    for fi, st in enumerate(stats):
        source_names.append(st["source_name"])
        genres.append(_extract_genre(st["source_name"]))
        for oi, offset in enumerate(st["offsets"]):
            total_rates[fi, oi] = offset["total_flip_rate"]  # fraction of all tokens that changed
            cb_rates[fi, oi] = offset["per_codebook_rates"]  # shape [n_codebooks]
            max_frame_rates[fi, oi] = offset["max_frame_rate"]

    return {
        "n_files": n_files,
        "n_codebooks": n_codebooks,
        "n_offsets": n_offsets,
        "offset_keys": offset_keys,
        "offset_labels": offset_labels,
        "source_names": source_names,
        "genres": genres,
        "unique_genres": sorted(set(genres)),
        # Per-offset aggregates
        "total_rates": total_rates,                    # (n_files, n_offsets)
        "total_rates_mean": total_rates.mean(axis=0),  # (n_offsets,)
        "total_rates_std": total_rates.std(axis=0),
        # Per-codebook aggregates
        "cb_rates": cb_rates,                                          # (n_files, n_offsets, n_cb)
        "cb_rates_mean": cb_rates.mean(axis=(0, 1)),                   # (n_cb,)
        "cb_rates_std": cb_rates.std(axis=(0, 1)),
        "cb_rates_per_offset_mean": cb_rates.mean(axis=0),             # (n_offsets, n_cb)
        # Max frame rates
        "max_frame_mean": max_frame_rates.mean(axis=0),
        "max_frame_std": max_frame_rates.std(axis=0),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2, linewidth=0.5)


def plot_summary_table(agg: dict, fig: plt.Figure) -> None:
    """Page 2: Summary table of mean flip rates per perturbation level."""
    ax = fig.add_subplot(111)
    ax.axis("off")

    labels = agg["offset_labels"]
    means = agg["total_rates_mean"]
    stds = agg["total_rates_std"]

    col_labels = ["Perturbation", "Mean Flip Rate", "Std Dev", "Max Frame Rate"]
    table_data = []
    for i, (lbl, m, s) in enumerate(zip(labels, means, stds)):
        mfr = agg["max_frame_mean"][i]
        table_data.append([lbl, f"{m:.4f}", f"{s:.4f}", f"{mfr:.4f}"])

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(len(table_data)):
        color = "#ecf0f1" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[i + 1, j].set_facecolor(color)

    fig.suptitle("Token Flip Rate Summary", fontsize=16, fontweight="bold", y=0.95)


def plot_codebook_sensitivity(agg: dict, fig: plt.Figure) -> None:
    """Page 3: Per-codebook mean ± std bar chart."""
    ax = fig.add_subplot(111)
    n_cb = agg["n_codebooks"]
    x = np.arange(n_cb)

    bars = ax.bar(x, agg["cb_rates_mean"], yerr=agg["cb_rates_std"],
                  capsize=3, color="#3498db", edgecolor="#2c3e50",
                  linewidth=0.5, alpha=0.85, error_kw=dict(lw=1, capthick=1))

    ax.set_xlabel("Codebook Index (RVQ Layer)", fontsize=12)
    ax.set_ylabel("Mean Flip Rate", fontsize=12)
    ax.set_title("Per-Codebook Token Sensitivity (Mean ± Std Across All Files)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"CB{i}" for i in x], fontsize=8)
    _style_axes(ax)

    # Annotate most/least sensitive
    most = int(np.argmax(agg["cb_rates_mean"]))
    least = int(np.argmin(agg["cb_rates_mean"]))
    ax.annotate(f"Most sensitive\nCB{most}: {agg['cb_rates_mean'][most]:.3f}",
                xy=(most, agg["cb_rates_mean"][most]),
                xytext=(most + 1.5, agg["cb_rates_mean"][most] + 0.02),
                fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1.2))


def plot_flip_rate_curve(agg: dict, fig: plt.Figure) -> None:
    """Page 4: Mean flip rate vs perturbation level with confidence band."""
    ax = fig.add_subplot(111)
    keys = np.arange(len(agg["offset_labels"]))
    mean = agg["total_rates_mean"]
    std = agg["total_rates_std"]

    ax.fill_between(keys, mean - std, mean + std, alpha=0.2, color="#3498db")
    ax.plot(keys, mean, "o-", color="#2c3e50", linewidth=2, markersize=5)

    ax.set_xlabel("Perturbation Level", fontsize=12)
    ax.set_ylabel("Mean Flip Rate", fontsize=12)
    ax.set_title("Token Flip Rate vs. Perturbation (Mean ± Std)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(keys)
    ax.set_xticklabels(agg["offset_labels"], fontsize=7, rotation=45, ha="right")
    _style_axes(ax)


def plot_genre_heatmap(agg: dict, fig: plt.Figure) -> None:
    """Page 5: Per-genre × per-codebook heatmap."""
    genres = agg["unique_genres"]
    if len(genres) <= 1:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Single genre — heatmap skipped",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        ax.axis("off")
        return

    n_cb = agg["n_codebooks"]
    genre_cb_rates = np.zeros((len(genres), n_cb))

    for fi, genre in enumerate(agg["genres"]):
        gi = genres.index(genre)
        # Mean across all offsets for this file
        genre_cb_rates[gi] += agg["cb_rates"][fi].mean(axis=0)

    # Average per genre
    genre_counts = np.array([agg["genres"].count(g) for g in genres], dtype=float)
    genre_cb_rates /= genre_counts[:, None]

    ax = fig.add_subplot(111)
    im = ax.imshow(genre_cb_rates, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xticks(range(n_cb))
    ax.set_xticklabels([f"CB{i}" for i in range(n_cb)], fontsize=8)
    ax.set_yticks(range(len(genres)))
    ax.set_yticklabels([g.capitalize() for g in genres], fontsize=9)
    ax.set_xlabel("Codebook Index", fontsize=12)
    ax.set_ylabel("Genre", fontsize=12)
    ax.set_title("Token Sensitivity by Genre × Codebook",
                 fontsize=14, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Mean Flip Rate", fontsize=10)

    # Annotate cells
    for i in range(len(genres)):
        for j in range(n_cb):
            val = genre_cb_rates[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if val > genre_cb_rates.mean() else "black")


def plot_aggregate_surface(agg: dict, fig: plt.Figure) -> None:
    """Page 6: Aggregate water cube — mean codebook × offset surface."""
    ax = fig.add_subplot(111, projection="3d")

    # cb_rates_per_offset_mean: (n_offsets, n_cb)
    data = agg["cb_rates_per_offset_mean"]
    n_offsets, n_cb = data.shape
    X, Y = np.meshgrid(np.arange(n_cb), np.arange(n_offsets))

    surf = ax.plot_surface(X, Y, data, cmap="viridis", alpha=0.9,
                           linewidth=0, antialiased=True)

    ax.set_xlabel("Codebook Index", fontsize=9)
    ax.set_ylabel("Perturbation Level", fontsize=9)
    ax.set_zlabel("Mean Flip Rate", fontsize=9)
    ax.set_title("Aggregate Token Sensitivity Surface",
                 fontsize=13, fontweight="bold")
    fig.colorbar(surf, ax=ax, shrink=0.6, label="Mean Flip Rate")


def plot_variance_summary(agg: dict, fig: plt.Figure) -> None:
    """Page 7: Variance analysis across files."""
    axes = fig.subplots(1, 2)

    n_cb = agg["n_codebooks"]
    x = np.arange(n_cb)

    # Left: CoV per codebook
    mean = agg["cb_rates_mean"]
    std = agg["cb_rates_std"]
    cov = np.where(mean > 0, std / mean, 0)

    axes[0].bar(x, cov, color="#e74c3c", edgecolor="#c0392b", linewidth=0.5, alpha=0.85)
    axes[0].set_xlabel("Codebook Index")
    axes[0].set_ylabel("Coefficient of Variation")
    axes[0].set_title("Cross-File Variability per Codebook")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"CB{i}" for i in x], fontsize=7)
    _style_axes(axes[0])

    # Right: File-level total flip rate distribution
    file_means = agg["total_rates"].mean(axis=1)  # (n_files,)
    axes[1].hist(file_means, bins=20, color="#2ecc71", edgecolor="#27ae60",
                 linewidth=0.5, alpha=0.85)
    axes[1].set_xlabel("Mean Flip Rate (per file)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of Per-File Flip Rates")
    _style_axes(axes[1])

    fig.suptitle("Variance Analysis", fontsize=14, fontweight="bold")


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def generate_report(
    experiment: str,
    bandwidth: str = "24.0",
    output_path: Path | None = None,
) -> Path | None:
    """Generate the consolidated aggregate PDF report."""
    from matplotlib.backends.backend_pdf import PdfPages

    stats = load_all_stats(experiment, bandwidth)
    if not stats:
        print(f"No stats found for experiment '{experiment}' at bandwidth {bandwidth}")
        return None

    agg = compute_aggregates(stats)
    print(f"\nAggregate report for '{experiment}':")
    print(f"  Files: {agg['n_files']}")
    print(f"  Codebooks: {agg['n_codebooks']}")
    print(f"  Perturbation levels: {agg['n_offsets']}")
    print(f"  Genres: {agg['unique_genres']}")

    if output_path is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = (_DATASETS_DIR / "analysis" / experiment /
                       f"aggregate_report_bw{bandwidth}_{ts}.pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(str(output_path)) as pdf:
        # Page 1: Cover
        cover = plt.figure(figsize=(11, 8.5))
        cover.suptitle("EnCodec Token Sensitivity Report",
                       fontsize=24, fontweight="bold", y=0.65)
        info = [
            f"Experiment : {experiment}",
            f"Files      : {agg['n_files']}",
            f"Codebooks  : {agg['n_codebooks']}",
            f"Bandwidth  : {bandwidth} kbps",
            f"Genres     : {', '.join(g.capitalize() for g in agg['unique_genres'])}",
            f"Generated  : {datetime.now().isoformat(timespec='seconds')}",
            "",
            f"Mean flip rate range: {agg['total_rates_mean'].min():.4f} — {agg['total_rates_mean'].max():.4f}",
        ]
        cover.text(0.5, 0.45, "\n".join(info), ha="center", va="top",
                   fontsize=12, family="monospace", transform=cover.transFigure)
        cover.patch.set_facecolor("#fafafa")
        plt.axis("off")
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)
        print("  Added: cover page")

        # Page 2: Summary table
        fig = plt.figure(figsize=(11, 8.5))
        plot_summary_table(agg, fig)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        print("  Added: summary table")

        # Page 3: Per-codebook sensitivity
        fig = plt.figure(figsize=(11, 6))
        plot_codebook_sensitivity(agg, fig)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        print("  Added: codebook sensitivity bars")

        # Page 4: Flip rate curve
        fig = plt.figure(figsize=(11, 6))
        plot_flip_rate_curve(agg, fig)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        print("  Added: flip rate curve")

        # Page 5: Genre heatmap
        fig = plt.figure(figsize=(11, 7))
        plot_genre_heatmap(agg, fig)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        print("  Added: genre heatmap")

        # Page 6: Aggregate 3D surface
        fig = plt.figure(figsize=(11, 8))
        plot_aggregate_surface(agg, fig)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        print("  Added: aggregate 3D surface")

        # Page 7: Variance analysis
        fig = plt.figure(figsize=(14, 6))
        plot_variance_summary(agg, fig)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        print("  Added: variance analysis")

        # Page 8: Water cube (Plotly frozen image)
        if HAS_PLOTLY and agg["n_offsets"] > 0:
            try:
                from scipy.ndimage import gaussian_filter
                import io

                data = agg["cb_rates_per_offset_mean"]  # (n_offsets, n_cb)
                smooth = gaussian_filter(data.T, sigma=1.0)  # (n_cb, n_offsets)

                n_cb, n_off = smooth.shape
                pfig = go.Figure(data=[go.Surface(
                    x=np.arange(n_off),
                    y=np.arange(n_cb),
                    z=smooth,
                    colorscale=[
                        [0.00, "rgb(5, 10, 40)"],
                        [0.20, "rgb(10, 40, 100)"],
                        [0.45, "rgb(20, 90, 160)"],
                        [0.65, "rgb(40, 140, 190)"],
                        [0.82, "rgb(100, 200, 220)"],
                        [1.00, "rgb(220, 240, 255)"],
                    ],
                    lighting=dict(ambient=0.55, diffuse=0.85, specular=0.5,
                                  roughness=0.3, fresnel=0.4),
                    colorbar=dict(title=dict(text="Avg Flip Rate",
                                             font=dict(color="white")),
                                  tickfont=dict(color="white")),
                    showscale=True,
                )])
                pfig.update_layout(
                    title=dict(text=f"Aggregate Token Sensitivity — {experiment}",
                               font=dict(size=18, color="white")),
                    scene=dict(
                        xaxis=dict(title="Perturbation Level", color="white",
                                   showbackground=False),
                        yaxis=dict(title="Codebook Index", color="white",
                                   showbackground=False),
                        zaxis=dict(title="Mean Flip Rate", color="white",
                                   showbackground=False),
                        bgcolor="rgba(5, 8, 25, 1.0)",
                        camera=dict(eye=dict(x=1.7, y=1.7, z=0.9)),
                    ),
                    paper_bgcolor="rgba(5, 8, 25, 1.0)",
                    font=dict(color="white"),
                    width=1200, height=800,
                )

                png_bytes = pio.to_image(pfig, format="png", engine="kaleido",
                                         width=1400, height=900, scale=2)
                from matplotlib.image import imread as mpl_imread
                buf = io.BytesIO(png_bytes)
                img = mpl_imread(buf, format="png")
                mfig, ax = plt.subplots(figsize=(14, 9), dpi=100)
                ax.imshow(img)
                ax.axis("off")
                mfig.tight_layout(pad=0)
                pdf.savefig(mfig, bbox_inches="tight")
                plt.close(mfig)
                print("  Added: water cube surface (Plotly)")
            except Exception as exc:
                print(f"  Warning: could not render water cube: {exc}")

    print(f"\nPDF saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate aggregate token sensitivity PDF report",
    )
    parser.add_argument("--experiment", "-e", required=True,
                        choices=["time", "room"],
                        help="Which experiment to aggregate")
    parser.add_argument("--bandwidth", type=str, default="24.0",
                        help="Bandwidth string (default: 24.0)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output PDF path (auto-generated if omitted)")
    args = parser.parse_args()

    generate_report(args.experiment, args.bandwidth, args.output)


if __name__ == "__main__":
    main()

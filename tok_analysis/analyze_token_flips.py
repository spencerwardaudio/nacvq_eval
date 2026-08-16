"""Analyze EnCodec token flip rates from time-offset audio experiments.

Produces:
  - 2x2 summary plot (flip rate, peak codebook, per-cb bar, temporal peak)
  - 4-panel variance/std report (per-cb std, top-5 trajectories, temporal var, CoV)
  - Per-offset 3D surface plots (matplotlib)
  - "Water cube" averaged flip surface (Plotly HTML, frozen image for PDF)
  - Multi-source progressive-difference comparison (when batch mode)
  - Multi-page PDF report

Usage:
    python analyze_token_flips.py [tokens_dir] [output_dir] [bandwidth] [--q2d2] [--stats-only]

Arguments:
    tokens_dir  Path to directory with .npy token files (default: ../datasets/audio_tokens/)
    output_dir  Output directory for analysis results (default: ../datasets/analysis/)
    bandwidth   Bandwidth string for filename matching (default: 24.0)
    --q2d2      Use Q²D² "Grid Pair" labels instead of "Codebook" in plots
    --stats-only  Skip visualizations, compute stats only

Defaults:
    tokens_dir  = ../datasets/audio_tokens/
    output_dir  = ../datasets/analysis/
    bandwidth   = 24.0
"""

from __future__ import annotations

import io
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

# Global labels - can be set to "Grid Pair" for Q2D2 mode
CB_LABEL = "Codebook"
CB_LABEL_LOWER = "codebook"


# ---------------------------------------------------------------------------
# Token loading
# ---------------------------------------------------------------------------

def load_tokens(offset_ms: int, token_dir: Path, bandwidth: str = "24.0") -> np.ndarray | None:
    if offset_ms == 0:
        fname = f"baseline_0ms_bw{bandwidth}_tokens.npy"
    else:
        fname = f"offset_{offset_ms:03d}ms_bw{bandwidth}_tokens.npy"
    token_path = token_dir / fname
    if token_path.exists():
        return np.load(str(token_path))
    return None


def load_baseline_tokens(token_dir: Path, bandwidth: str = "24.0") -> np.ndarray | None:
    """Load baseline tokens. First tries explicit baseline name, then uses first sorted token file."""
    baseline_path = token_dir / f"baseline_0ms_bw{bandwidth}_tokens.npy"
    if baseline_path.exists():
        return np.load(str(baseline_path))
    
    # Fallback: use first sorted token file as baseline (for room/doa experiments)
    token_files = sorted(token_dir.glob(f"*_bw{bandwidth}_tokens.npy"))
    if token_files:
        print(f"[INFO] No explicit baseline found; using first file as baseline: {token_files[0].name}")  # room/DOA experiments use alphabetical first as reference
        return np.load(str(token_files[0]))
    
    return None


def discover_offsets(token_dir: Path, bandwidth: str = "24.0") -> list[int]:
    offsets = []
    for p in token_dir.glob(f"offset_*ms_bw{bandwidth}_tokens.npy"):
        parts = p.stem.split("_")
        ms_str = parts[1].replace("ms", "")
        try:
            offsets.append(int(ms_str))
        except ValueError:
            pass
    return sorted(offsets)


def discover_all_token_files(token_dir: Path, bandwidth: str = "24.0") -> list[Path]:
    """Discover all token .npy files, excluding the baseline (for room/doa experiments)."""
    all_files = sorted(token_dir.glob(f"*_bw{bandwidth}_tokens.npy"))
    if not all_files:
        return []
    baseline_name = f"baseline_0ms_bw{bandwidth}_tokens.npy"
    baseline_path = token_dir / baseline_name
    if not baseline_path.exists() and all_files:
        baseline_path = all_files[0]
    return [f for f in all_files if f != baseline_path]


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def compute_token_flip_matrix(baseline: np.ndarray, perturbed: np.ndarray) -> np.ndarray:
    min_frames = min(baseline.shape[1], perturbed.shape[1])  # different offsets may produce slightly different lengths at segment edges
    return (baseline[:, :min_frames] != perturbed[:, :min_frames]).astype(float)


# ---------------------------------------------------------------------------
# Variance / standard deviation across offsets
# ---------------------------------------------------------------------------

def compute_variance_stats(flip_matrices: dict[int, np.ndarray]) -> dict:
    """Per-codebook and per-frame variance/std of flip rate across offset levels."""
    stacked = np.stack(list(flip_matrices.values()), axis=0)  # (n_offsets, n_cb, n_frames)

    per_cb_rates = stacked.mean(axis=2)   # (n_offsets, n_cb) — average over frames to get per-condition per-cb rate
    cb_var = per_cb_rates.var(axis=0)  # variance across offset levels shows which codebooks are most sensitive
    cb_std = per_cb_rates.std(axis=0)

    per_frame_rates = stacked.mean(axis=1)   # (n_offsets, n_frames)
    frame_var = per_frame_rates.var(axis=0)
    frame_std = per_frame_rates.std(axis=0)

    n_top = min(5, stacked.shape[1])
    top_idx = np.argsort(cb_var)[::-1][:n_top]

    return {
        "per_codebook_variance": cb_var,
        "per_codebook_std": cb_std,
        "per_frame_variance": frame_var,
        "per_frame_std": frame_std,
        "top_codebooks": top_idx,
        "top_codebook_variances": cb_var[top_idx],
        "n_offsets": stacked.shape[0],
        "n_codebooks": stacked.shape[1],
        "n_frames": stacked.shape[2],
    }


# ---------------------------------------------------------------------------
# Visualization — summary plots
# ---------------------------------------------------------------------------

def plot_flip_rate_summary(offset_stats: list[dict], output_path: Path) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    offsets = [s["offset_ms"] for s in offset_stats]
    flip_rates = [s["total_flip_rate"] for s in offset_stats]

    axes[0, 0].plot(offsets, flip_rates, "o-", color="steelblue", linewidth=2, markersize=8)
    axes[0, 0].set_xlabel("Time Offset (ms)")
    axes[0, 0].set_ylabel("Overall Token Flip Rate")
    axes[0, 0].set_title("Flip Rate vs. Time Offset")
    axes[0, 0].grid(True, alpha=0.3)

    most_sensitive_rates = [s["most_sensitive_rate"] for s in offset_stats]
    axes[0, 1].plot(offsets, most_sensitive_rates, "s-", color="darkorange", linewidth=2, markersize=8)
    axes[0, 1].set_xlabel("Time Offset (ms)")
    axes[0, 1].set_ylabel("Most Sensitive CB Flip Rate")
    axes[0, 1].set_title(f"Peak {CB_LABEL} Sensitivity vs. Offset")
    axes[0, 1].grid(True, alpha=0.3)

    if offset_stats:
        last = offset_stats[-1]
        cb_rates = last["per_codebook_rates"]
        axes[1, 0].bar(range(len(cb_rates)), cb_rates, color="green", alpha=0.7)
        axes[1, 0].set_xlabel(f"{CB_LABEL} Index")
        axes[1, 0].set_ylabel("Flip Rate")
        axes[1, 0].set_title(f"Per-{CB_LABEL} at {last['offset_ms']}ms")
        axes[1, 0].grid(True, alpha=0.3, axis="y")

    max_frame_rates = [s["max_frame_rate"] for s in offset_stats]
    axes[1, 1].plot(offsets, max_frame_rates, "^-", color="purple", linewidth=2, markersize=8)
    axes[1, 1].set_xlabel("Time Offset (ms)")
    axes[1, 1].set_ylabel("Max Flip Rate in Any Frame")
    axes[1, 1].set_title("Peak Temporal Sensitivity vs. Offset")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Token Flip Analysis — Time Offset Experiment", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved summary plot -> {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Visualization — variance / std report
# ---------------------------------------------------------------------------

def plot_variance_stats(var_stats: dict, offset_stats: list[dict], output_path: Path) -> plt.Figure:
    """4-panel: per-cb std, top-5 trajectories, per-frame var, CoV."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    cb_std = var_stats["per_codebook_std"]
    frame_var = var_stats["per_frame_variance"]
    top_idx = var_stats["top_codebooks"]
    n_cb = var_stats["n_codebooks"]
    offsets = [s["offset_ms"] for s in offset_stats]

    # [0,0] Per-codebook std bar
    bar_colors = ["tomato" if i in top_idx else "steelblue" for i in range(n_cb)]
    axes[0, 0].bar(range(n_cb), cb_std, color=bar_colors, alpha=0.85)
    for cb_i in top_idx:
        axes[0, 0].annotate(f"CB{cb_i}", (cb_i, cb_std[cb_i]),
                             textcoords="offset points", xytext=(0, 4),
                             ha="center", fontsize=7, color="tomato", fontweight="bold")
    axes[0, 0].set_xlabel(f"{CB_LABEL} Index")
    axes[0, 0].set_ylabel("Std. Dev. of Flip Rate")
    axes[0, 0].set_title(f"Per-{CB_LABEL} Variance  (red = top-5)")
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # [0,1] Top-5 codebook trajectories
    traj_colors = plt.cm.tab10(np.linspace(0, 0.6, len(top_idx)))
    for cb_i, color in zip(top_idx, traj_colors):
        rates = [s["per_codebook_rates"][cb_i] for s in offset_stats]
        axes[0, 1].plot(offsets, rates, "o-", color=color, linewidth=2,
                        markersize=6, label=f"CB{cb_i}")
    axes[0, 1].set_xlabel("Time Offset (ms)")
    axes[0, 1].set_ylabel("Flip Rate")
    axes[0, 1].set_title(f"Top-5 Variable {CB_LABEL}s vs. Offset")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # [1,0] Per-frame temporal variance
    frames = np.arange(len(frame_var))
    axes[1, 0].fill_between(frames, 0, frame_var, alpha=0.5, color="darkorange")
    axes[1, 0].plot(frames, frame_var, color="darkorange", linewidth=0.8)
    axes[1, 0].set_xlabel("Frame (Latent Time Step)")
    axes[1, 0].set_ylabel("Variance of Flip Rate")
    axes[1, 0].set_title("Per-Frame Temporal Variance")
    axes[1, 0].grid(True, alpha=0.3)

    # [1,1] Coefficient of variation per codebook
    per_cb_mean = np.array([s["per_codebook_rates"] for s in offset_stats]).mean(axis=0)
    cov = np.where(per_cb_mean > 1e-6, cb_std / per_cb_mean, 0.0)
    axes[1, 1].bar(range(n_cb), cov, color="mediumpurple", alpha=0.8)
    axes[1, 1].set_xlabel("Codebook Index")
    axes[1, 1].set_ylabel("Std / Mean  (CoV)")
    axes[1, 1].set_title("Coefficient of Variation per Codebook")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Token Flip Variance Analysis", fontsize=16, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved variance plot -> {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Visualization — per-offset 3D surface (matplotlib for PDF)
# ---------------------------------------------------------------------------

def plot_3d_surface_matplotlib(flip_matrix: np.ndarray, offset_ms: int,
                               output_path: Path) -> plt.Figure:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    n_codebooks, n_frames = flip_matrix.shape
    X, Y = np.meshgrid(np.arange(n_frames), np.arange(n_codebooks))
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, flip_matrix, cmap="viridis", alpha=0.9,
                           linewidth=0, antialiased=True)
    ax.set_xlabel("Frame (Latent Step)")
    ax.set_ylabel(f"{CB_LABEL} Index")
    ax.set_zlabel("Token Flip")
    ax.set_title(f"Token Flip Pattern at {offset_ms}ms Offset",
                 fontsize=14, fontweight="bold")
    fig.colorbar(surf, ax=ax, label="Flipped", shrink=0.5)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3D plot -> {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Visualization — per-offset 3D surface (Plotly interactive)
# ---------------------------------------------------------------------------

def create_3d_surface_plotly(flip_matrix: np.ndarray, offset_ms: int):
    n_codebooks, n_frames = flip_matrix.shape
    fig = go.Figure(data=[go.Surface(
        x=np.arange(n_frames),
        y=np.arange(n_codebooks),
        z=flip_matrix,
        colorscale="Viridis",
        colorbar=dict(title="Flipped"),
        showscale=True,
    )])
    fig.update_layout(
        title=f"Token Flip Pattern at {offset_ms}ms Offset",
        scene=dict(
            xaxis=dict(title="Frame (Latent Step)"),
            yaxis=dict(title=f"{CB_LABEL} Index"),
            zaxis=dict(title="Token Flip (1=yes)"),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        ),
        width=1000, height=700,
    )
    return fig


# ---------------------------------------------------------------------------
# "Water Cube" — averaged flip surface, frozen interactive Plotly HTML
# ---------------------------------------------------------------------------

def create_water_cube_plotly(
    flip_matrices: dict[int, np.ndarray],
    source_name: str = "",
) -> "go.Figure":
    """
    Z = mean flip rate per (codebook x frame) averaged across all offsets,
    then Gaussian-smoothed so the surface looks fluid.
    Axes: X=frame (time), Y=codebook (token index), Z=mean flip rate.
    """
    from scipy.ndimage import gaussian_filter

    stacked = np.stack(list(flip_matrices.values()), axis=0)
    avg = stacked.mean(axis=0)
    smooth = gaussian_filter(avg, sigma=1.5)

    n_codebooks, n_frames = smooth.shape
    title = (f"Average Token Flip Sensitivity — {source_name}"
             if source_name else "Average Token Flip Sensitivity")

    fig = go.Figure(data=[go.Surface(
        x=np.arange(n_frames),
        y=np.arange(n_codebooks),
        z=smooth,
        colorscale=[
            [0.00, "rgb(5,  10, 40)"],
            [0.20, "rgb(10, 40, 100)"],
            [0.45, "rgb(20, 90, 160)"],
            [0.65, "rgb(40, 140, 190)"],
            [0.82, "rgb(100, 200, 220)"],
            [1.00, "rgb(220, 240, 255)"],
        ],
        lighting=dict(ambient=0.55, diffuse=0.85, specular=0.5,
                      roughness=0.3, fresnel=0.4),
        lightposition=dict(x=200, y=100, z=300),
        colorbar=dict(title=dict(text="Avg Flip Rate", font=dict(color="white")),
                      tickfont=dict(color="white")),
        contours=dict(z=dict(
            show=True, usecolormap=True,
            highlightcolor="rgba(255,255,255,0.5)",
            project_z=True,
        )),
        showscale=True,
    )])

    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="white")),
        scene=dict(
            xaxis=dict(title="Frame (Time)", color="white",
                       gridcolor="rgba(255,255,255,0.15)", showbackground=False),
            yaxis=dict(title="Codebook (Token)", color="white",
                       gridcolor="rgba(255,255,255,0.15)", showbackground=False),
            zaxis=dict(title="Mean Flip Rate", color="white",
                       gridcolor="rgba(255,255,255,0.15)", showbackground=False),
            bgcolor="rgba(5, 8, 25, 1.0)",
            camera=dict(up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=-0.05),
                        eye=dict(x=1.7, y=1.7, z=0.9)),
        ),
        paper_bgcolor="rgba(5, 8, 25, 1.0)",
        font=dict(color="white"),
        width=1200, height=800,
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig


# ---------------------------------------------------------------------------
# Multi-source comparison (overlay all source files)
# ---------------------------------------------------------------------------

def plot_multi_source_comparison(
    all_stats: dict[str, list[dict]],
    output_path: Path,
) -> plt.Figure:
    """
    Left panel : one flip-rate-vs-offset line per source.
    Right panel: progressive absolute differences between adjacent sources.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    n_sources = len(all_stats)
    palette = plt.cm.tab10(np.linspace(0, 0.9, max(n_sources, 1)))

    source_curves: dict[str, tuple[list, list]] = {}

    for (source_name, stats), color in zip(all_stats.items(), palette):
        if not stats:
            continue
        offsets = [s["offset_ms"] for s in stats]
        rates = [s["total_flip_rate"] for s in stats]
        source_curves[source_name] = (offsets, rates)
        axes[0].plot(offsets, rates, "o-", color=color, linewidth=2,
                     markersize=6, label=source_name, zorder=3)

    axes[0].set_xlabel("Time Offset (ms)")
    axes[0].set_ylabel("Mean Token Flip Rate")
    axes[0].set_title("Flip Rate vs. Offset — All Sources")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(True, alpha=0.3)

    sources = list(source_curves.keys())
    if len(sources) > 1:
        diff_pal = plt.cm.plasma(np.linspace(0.1, 0.85, len(sources) - 1))
        for i, (s1, s2) in enumerate(zip(sources, sources[1:])):
            off1, r1 = source_curves[s1]
            off2, r2 = source_curves[s2]
            common = sorted(set(off1) & set(off2))
            d1 = dict(zip(off1, r1))
            d2 = dict(zip(off2, r2))
            diffs = [abs(d2[o] - d1[o]) for o in common]
            axes[1].plot(common, diffs, "--o", color=diff_pal[i], linewidth=1.5,
                         markersize=5, label=f"|{s2} - {s1}|")
        axes[1].legend(fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "Only one source -- no comparison available",
                     ha="center", va="center", transform=axes[1].transAxes)

    axes[1].set_xlabel("Time Offset (ms)")
    axes[1].set_ylabel("Absolute Flip Rate Difference")
    axes[1].set_title("Progressive Difference Between Sources")
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Multi-Source Token Sensitivity Comparison",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved multi-source comparison -> {output_path}")
    return fig


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------

def export_pdf(
    summary_fig: plt.Figure,
    variance_fig: plt.Figure | None,
    flip_matrices: dict[int, np.ndarray],
    token_dir: Path,
    output_dir: Path,
    bandwidth: str,
    source_name: str = "",
) -> Path:
    from matplotlib.backends.backend_pdf import PdfPages

    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = output_dir / f"token_flip_analysis_bw{bandwidth}_{ts}.pdf"

    with PdfPages(pdf_path) as pdf:
        # Cover
        cover = plt.figure(figsize=(11, 8.5))
        cover.suptitle("EnCodec Token Sensitivity Report\n(Time-Offset Experiment)",
                        fontsize=20, fontweight="bold", y=0.62)
        info_lines = [
            f"Source    : {source_name or token_dir.name}",
            f"Generated : {datetime.now().isoformat(timespec='seconds')}",
            f"Token dir : {token_dir}",
            f"Bandwidth : {bandwidth} kbps",
            f"Offsets   : {sorted(flip_matrices.keys())} ms",
        ]
        cover.text(0.5, 0.45, "\n".join(info_lines), ha="center", va="top",
                   fontsize=11, family="monospace", transform=cover.transFigure)
        cover.patch.set_facecolor("#f7f7f7")
        plt.axis("off")
        pdf.savefig(cover, bbox_inches="tight")
        plt.close(cover)

        # Summary 2x2
        pdf.savefig(summary_fig, bbox_inches="tight")
        print("  Added page: summary chart (2x2)")

        # Variance 4-panel
        if variance_fig is not None:
            pdf.savefig(variance_fig, bbox_inches="tight")
            print("  Added page: variance/std report")

        # Water-cube frozen image (if Plotly + kaleido available)
        if HAS_PLOTLY and flip_matrices:
            try:
                water_fig = create_water_cube_plotly(flip_matrices, source_name)
                from matplotlib.image import imread as mpl_imread
                png_bytes = pio.to_image(water_fig, format="png", engine="kaleido",
                                         width=1400, height=900, scale=2)
                buf = io.BytesIO(png_bytes)
                img = mpl_imread(buf, format="png")
                mfig, ax = plt.subplots(figsize=(14, 9), dpi=100)
                ax.imshow(img)
                ax.axis("off")
                mfig.tight_layout(pad=0)
                pdf.savefig(mfig, bbox_inches="tight")
                plt.close(mfig)
                print("  Added page: water-cube averaged surface")
            except Exception as exc:
                print(f"  Warning: could not render water-cube to PDF: {exc}")

        # 3D surfaces per offset
        for offset_ms in sorted(flip_matrices.keys()):
            flip_matrix = flip_matrices[offset_ms]
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            n_codebooks, n_frames = flip_matrix.shape
            X, Y = np.meshgrid(np.arange(n_frames), np.arange(n_codebooks))
            mfig = plt.figure(figsize=(12, 8))
            ax = mfig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(X, Y, flip_matrix, cmap="viridis", alpha=0.9,
                                   linewidth=0, antialiased=True)
            ax.set_xlabel("Frame (Latent Step)")
            ax.set_ylabel("Codebook Index")
            ax.set_zlabel("Token Flip")
            ax.set_title(f"Token Flip at {offset_ms}ms", fontsize=14, fontweight="bold")
            mfig.colorbar(surf, ax=ax, label="Flipped", shrink=0.5)
            pdf.savefig(mfig, bbox_inches="tight")
            plt.close(mfig)
            print(f"  Added page: {offset_ms}ms 3D surface")

        d = pdf.infodict()
        d["Title"] = "EnCodec Token Sensitivity Analysis (Time Offset)"
        d["Subject"] = f"Bandwidth {bandwidth} kbps | {len(flip_matrices)} offsets"

    print(f"\nPDF saved -> {pdf_path}")
    return pdf_path


# ---------------------------------------------------------------------------
# Main analysis runner — returns stats for multi-source aggregation
# ---------------------------------------------------------------------------

def run_analysis(
    token_dir: Path,
    output_dir: Path,
    bandwidth: str = "24.0",
    source_name: str = "",
    stats_only: bool = False,
) -> tuple[list[dict], dict[int, np.ndarray]]:
    """Run full analysis for one source. Returns (offset_stats, flip_matrices)."""
    token_dir = token_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_name = source_name or token_dir.name

    print(f"Token directory : {token_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Bandwidth       : {bandwidth} kbps\n")

    baseline_tokens = load_baseline_tokens(token_dir, bandwidth)
    if baseline_tokens is None:
        print(f"ERROR: baseline token file not found in {token_dir}")
        print(f"  Expected: baseline_0ms_bw{bandwidth}_tokens.npy")
        return [], {}

    print(f"Baseline tokens: {baseline_tokens.shape}  "
          f"({baseline_tokens.shape[0]} codebooks x {baseline_tokens.shape[1]} frames)")

    offsets_ms = discover_offsets(token_dir, bandwidth)
    non_baseline_files: list[Path] = []

    if not offsets_ms:
        non_baseline_files = discover_all_token_files(token_dir, bandwidth)
        if not non_baseline_files:
            print("No offset or comparison token files found.")
            return [], {}
        print(f"Found {len(non_baseline_files)} comparison file(s) (generic mode)\n")
    else:
        print(f"Found {len(offsets_ms)} offset(s): {offsets_ms}\n")

    # ---- Collect stats -----------------------------------------------------
    offset_stats: list[dict] = []
    flip_matrices: dict[int, np.ndarray] = {}

    def _add_flip(key: int, tokens: np.ndarray, label: str = "") -> None:
        flip_matrix = compute_token_flip_matrix(baseline_tokens, tokens)
        flip_matrices[key] = flip_matrix

        per_cb = flip_matrix.mean(axis=1)
        most_sensitive_cb = int(np.argmax(per_cb))
        offset_stats.append({
            "offset_ms": key,
            "total_flip_rate": float(flip_matrix.mean()),
            "per_codebook_rates": per_cb.tolist(),
            "most_sensitive_cb": most_sensitive_cb,
            "most_sensitive_rate": float(per_cb[most_sensitive_cb]),
            "max_frame_rate": float(flip_matrix.mean(axis=0).max()),
            "label": label or f"{key}ms",
        })
        display = label or f"{key:4d}ms"
        print(f"  {display} | flip_rate={flip_matrix.mean():.4f} | "
              f"peak_cb=CB{most_sensitive_cb} ({per_cb[most_sensitive_cb]:.4f})")

    if offsets_ms:
        for offset_ms in offsets_ms:
            tokens = load_tokens(offset_ms, token_dir, bandwidth)
            if tokens is None:
                print(f"[SKIP] {offset_ms}ms -- token file missing")
                continue
            _add_flip(offset_ms, tokens)
    else:
        for idx, token_file in enumerate(non_baseline_files, start=1):
            tokens = np.load(str(token_file))
            label = token_file.stem.replace(f"_bw{bandwidth}_tokens", "")
            _add_flip(idx, tokens, label=label)

    if not offset_stats:
        print("No valid offset tokens found.")
        return [], {}

    # ---- Always save compact JSON stats for aggregation ---------------------
    import json
    stats_path = output_dir / f"stats_bw{bandwidth}.json"
    json_payload = {
        "source_name": source_name or token_dir.name,
        "bandwidth": bandwidth,
        "n_codebooks": int(flip_matrices[list(flip_matrices.keys())[0]].shape[0]),
        "n_frames": int(flip_matrices[list(flip_matrices.keys())[0]].shape[1]),
        "offsets": [{
            "key": s["offset_ms"],
            "label": s["label"],
            "total_flip_rate": s["total_flip_rate"],
            "per_codebook_rates": s["per_codebook_rates"] if isinstance(s["per_codebook_rates"], list) else s["per_codebook_rates"].tolist(),
            "most_sensitive_cb": s["most_sensitive_cb"],
            "most_sensitive_rate": s["most_sensitive_rate"],
            "max_frame_rate": s["max_frame_rate"],
        } for s in offset_stats],
    }
    with open(stats_path, "w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"Saved stats JSON -> {stats_path}")

    if stats_only:
        plt.close("all")
        print(f"\nStats-only mode. JSON saved to: {output_dir}")
        return offset_stats, flip_matrices

    # ---- Summary plot -------------------------------------------------------
    summary_fig = plot_flip_rate_summary(
        offset_stats, output_dir / f"summary_bw{bandwidth}.png")

    # ---- Variance / std report ----------------------------------------------
    variance_fig = None
    if len(flip_matrices) >= 2:
        var_stats = compute_variance_stats(flip_matrices)
        variance_fig = plot_variance_stats(
            var_stats, offset_stats,
            output_dir / f"variance_bw{bandwidth}.png")

        # Print top-5 most variable codebooks
        print(f"\n  Top-5 most variable codebooks (std across offsets):")
        for rank, cb_i in enumerate(var_stats["top_codebooks"]):
            print(f"    #{rank+1}: CB{cb_i}  std={var_stats['per_codebook_std'][cb_i]:.4f}  "
                  f"var={var_stats['per_codebook_variance'][cb_i]:.6f}")

    # ---- 3D surface plots per offset ----------------------------------------
    for offset_ms, flip_matrix in flip_matrices.items():
        plot_3d_surface_matplotlib(
            flip_matrix, offset_ms,
            output_dir / f"surface_{offset_ms:03d}ms_bw{bandwidth}.png")

    # ---- Water cube (Plotly HTML) -------------------------------------------
    if HAS_PLOTLY and flip_matrices:
        water_fig = create_water_cube_plotly(flip_matrices, source_name)
        html_path = output_dir / f"water_cube_bw{bandwidth}.html"
        water_fig.write_html(str(html_path), include_plotlyjs="cdn")
        print(f"Saved water-cube HTML -> {html_path}")

    # ---- PDF report ---------------------------------------------------------
    export_pdf(summary_fig, variance_fig, flip_matrices,
               token_dir, output_dir, bandwidth, source_name)

    plt.close("all")
    print(f"\nAnalysis complete. All outputs in: {output_dir}")
    return offset_stats, flip_matrices


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Check for Q2D2 mode flag
    if "--q2d2" in sys.argv:
        CB_LABEL = "Grid Pair"
        CB_LABEL_LOWER = "grid pair"
    
    proj_root = _PROJ_ROOT
    default_tokens_base = proj_root / "datasets" / "audio_tokens"
    default_analysis_base = proj_root / "datasets" / "analysis"
    default_bandwidth = "24.0"

    if len(sys.argv) >= 2 and sys.argv[1] not in ["--q2d2", "--stats-only"]:
        tokens_dir = Path(sys.argv[1])
        output_dir = (Path(sys.argv[2]) if len(sys.argv) > 2
                      else default_analysis_base / tokens_dir.name)
        bandwidth = sys.argv[3] if len(sys.argv) > 3 else default_bandwidth
        stats_only = "--stats-only" in sys.argv
        run_analysis(tokens_dir, output_dir, bandwidth, stats_only=stats_only)
    else:
        # Batch mode: analyze every sub-directory in audio_tokens/
        subdirs = (sorted(p for p in default_tokens_base.iterdir() if p.is_dir())
                   if default_tokens_base.exists() else [])
        if not subdirs:
            print(f"No token sub-directories found in {default_tokens_base}")
            print("Usage: python analyze_token_flips.py [tokens_dir] [output_dir] [bandwidth]")
            sys.exit(1)

        print(f"Batch mode: analyzing {len(subdirs)} source(s)\n")

        # Collect per-source stats for multi-source comparison
        all_stats: dict[str, list[dict]] = {}

        for subdir in subdirs:
            out_dir = default_analysis_base / subdir.name
            print(f"\n{'='*70}")
            print(f"Source: {subdir.name}")
            print(f"{'='*70}")
            stats, _ = run_analysis(subdir, out_dir, default_bandwidth,
                                     source_name=subdir.name)
            if stats:
                all_stats[subdir.name] = stats

        # Multi-source comparison plot
        if len(all_stats) > 1:
            comparison_dir = default_analysis_base / "_comparison"
            comparison_dir.mkdir(parents=True, exist_ok=True)
            plot_multi_source_comparison(
                all_stats,
                comparison_dir / f"multi_source_comparison_bw{default_bandwidth}.png")
            print(f"\nMulti-source comparison saved to {comparison_dir}")

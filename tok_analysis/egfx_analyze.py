"""Generate comparative analysis and visualizations for EGFx metrics.

Usage:
    python tok_analysis/egfx_analyze.py \\
        --metrics datasets/analysis/egfx_metrics.json \\
        --output datasets/analysis/egfx_report.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


CODEC_CONFIGS = {
    "encodec": {"label": "EnCodec", "color": "#2196F3", "n_layers": 32},
    "q2d2": {"label": "Q2D2", "color": "#4CAF50", "n_layers": 16},
    "hificodec": {"label": "HiFiCodec", "color": "#FF9800", "n_layers": 4},
    "dac_fsq": {"label": "DAC-FSQ", "color": "#9C27B0", "n_layers": 1},
    "speechtokenizer": {"label": "SpeechTokenizer", "color": "#F44336", "n_layers": 8},
}


def plot_metric_by_layer(metrics_data: dict, metric_key: str, ylabel: str) -> plt.Figure:
    """Plot metric values across layers for all codecs and categories."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Get all categories
    categories = set()
    for codec_metrics in metrics_data.values():
        categories.update(codec_metrics.keys())
    categories = sorted(categories)
    
    for idx, category in enumerate(categories):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        for codec_name, codec_metrics in metrics_data.items():
            if category not in codec_metrics:
                continue
            
            cfg = CODEC_CONFIGS.get(codec_name, {})
            color = cfg.get("color", "gray")
            label = cfg.get("label", codec_name)
            
            # Aggregate metric across pairs
            metric_values = []
            for pair in codec_metrics[category]:
                metric_values.append(pair[metric_key])
            
            if not metric_values:
                continue
            # Mean and std across pairs
            mean_values = np.mean(metric_values, axis=0)
            std_values = np.std(metric_values, axis=0)
            if np.ndim(mean_values) == 0:
                continue
            layers = np.arange(1, len(mean_values) + 1)
            ax.plot(layers, mean_values, marker='o', label=label, color=color, linewidth=2)
            ax.fill_between(layers, mean_values - std_values, mean_values + std_values,
                           alpha=0.2, color=color)
        
        ax.set_title(f"{category.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Quantization Layer")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # Hide unused subplots
    for idx in range(len(categories), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle(f"{ylabel} by Effect Category", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_cross_codec_summary(metrics_data: dict) -> plt.Figure:
    """Create summary comparison across codecs."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics_to_plot = [
        ("tfr", "Token Flip Rate"),
        ("cosine_similarity", "Cosine Similarity"),
        ("l2_distance", "L2 Distance"),
    ]
    
    for ax, (metric_key, ylabel) in zip(axes, metrics_to_plot):
        for codec_name, codec_metrics in metrics_data.items():
            cfg = CODEC_CONFIGS.get(codec_name, {})
            color = cfg.get("color", "gray")
            label = cfg.get("label", codec_name)
            
            # Aggregate across all categories and pairs
            all_values = []
            for category_metrics in codec_metrics.values():
                for pair in category_metrics:
                    all_values.append(pair[metric_key])
            
            if not all_values:
                continue
            
            mean_values = np.mean(all_values, axis=0)
            
            # Normalize layer position (0-1) for fair comparison
            n_layers = len(mean_values)
            normalized_layers = np.linspace(0, 1, n_layers)
            
            ax.plot(normalized_layers, mean_values, marker='o', label=label,
                   color=color, linewidth=2, markersize=6)
        
        ax.set_xlabel("Normalized Layer Depth (0=first, 1=last)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    fig.suptitle("Cross-Codec Comparison (All Effects)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def generate_summary_table(metrics_data: dict, sampled_pairs: dict | None = None) -> str:
    """Generate markdown summary table plus sampling composition details."""
    lines = ["# EGFx Multi-Codec Geometric Analysis", "", "## Summary Statistics", ""]

    # Optional sampled-pairs metadata for reproducibility on the summary page.
    if sampled_pairs is not None:
        lines.append("## Sample Composition")
        lines.append("")
        lines.append("| Category | Pair Count |")
        lines.append("|----------|------------|")
        counts = {k: len(v) for k, v in sampled_pairs.items()}
        for category in sorted(counts.keys()):
            lines.append(f"| {category} | {counts[category]} |")
        nonzero = [v for v in counts.values() if v > 0]
        balanced = bool(nonzero) and all(v == nonzero[0] for v in nonzero)
        if nonzero:
            lines.append("")
            lines.append(
                f"Balanced across categories: {'Yes' if balanced else 'No'} "
                f"(target={min(nonzero)}, total={sum(nonzero)})"
            )
        lines.append("")
    
    # Header
    lines.append("| Codec | Layers | Avg TFR | Avg Cosine Sim | Avg L2 Dist |")
    lines.append("|-------|--------|---------|----------------|-------------|")
    
    for codec_name, codec_metrics in metrics_data.items():
        cfg = CODEC_CONFIGS.get(codec_name, {})
        label = cfg.get("label", codec_name)
        n_layers = cfg.get("n_layers", "?")
        
        # Aggregate metrics
        all_tfr, all_cos, all_l2 = [], [], []
        for category_metrics in codec_metrics.values():
            for pair in category_metrics:
                all_tfr.extend(pair["tfr"])
                all_cos.extend(pair["cosine_similarity"])
                all_l2.extend(pair["l2_distance"])
        
        if all_tfr:
            avg_tfr = np.mean(all_tfr)
            avg_cos = np.mean(all_cos)
            avg_l2 = np.mean(all_l2)
            
            lines.append(f"| {label} | {n_layers} | {avg_tfr:.3f} | {avg_cos:.3f} | {avg_l2:.1f} |")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze EGFx metrics")
    parser.add_argument("--metrics", type=Path, required=True, help="Metrics JSON file")
    parser.add_argument("--sampled-pairs", type=Path, default=None,
                        help="Optional sampled-pairs JSON for summary composition metadata")
    parser.add_argument("--output", type=Path, default=Path("datasets/analysis/egfx_report.pdf"),
                        help="Output PDF report")
    args = parser.parse_args()
    
    # Load metrics
    with open(args.metrics) as f:
        metrics_data = json.load(f)

    sampled_pairs = None
    if args.sampled_pairs is not None and args.sampled_pairs.exists():
        with open(args.sampled_pairs) as f:
            sampled_pairs = json.load(f)
    
    print("Generating analysis report...")
    
    # Generate plots
    with PdfPages(args.output) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.5, "EGFx Multi-Codec\nGeometric Analysis",
                ha='center', va='center', fontsize=24, fontweight='bold')
        pdf.savefig(fig)
        plt.close()
        
        # Summary table
        summary_text = generate_summary_table(metrics_data, sampled_pairs=sampled_pairs)
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.1, 0.9, summary_text, ha='left', va='top', fontsize=10,
                family='monospace', wrap=True)
        pdf.savefig(fig)
        plt.close()
        
        # TFR by layer
        fig = plot_metric_by_layer(metrics_data, "tfr", "Token Flip Rate")
        pdf.savefig(fig)
        plt.close()
        
        # Cosine similarity by layer
        fig = plot_metric_by_layer(metrics_data, "cosine_similarity", "Cosine Similarity")
        pdf.savefig(fig)
        plt.close()
        
        # L2 distance by layer
        fig = plot_metric_by_layer(metrics_data, "l2_distance", "L2 Distance")
        pdf.savefig(fig)
        plt.close()
        
        # Cross-codec summary
        fig = plot_cross_codec_summary(metrics_data)
        pdf.savefig(fig)
        plt.close()
    
    print(f"\n✓ Analysis report saved to {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot semantic evaluation results for codec comparison.

This script generates plots comparing classification accuracy across different codecs
and bitrates on HEAR benchmark tasks, matching the style of the SemantiCodec paper.

Usage:
    # Generate all plots
    python plot_semantic_results.py
    
    # Generate plots for specific task
    python plot_semantic_results.py --task esc50
    
    # Custom input/output paths
    python plot_semantic_results.py --input results.csv --output-dir plots/
"""

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Set publication-quality defaults
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']


# ============================================================================
# Task Information
# ============================================================================

TASK_DISPLAY_NAMES = {
    'esc50': 'ESC-50',
    'speech_commands': 'Speech Commands',
    'nsynth_pitch': 'NSynth Pitch',
    'libricount': 'LibriCount',
    'crema_d': 'CREMA-D',
    'vocal_imitation': 'Vocal Imitations',
}

CODEC_DISPLAY_NAMES = {
    'encodec': 'Encodec',
    'semanticodec': 'SemantiCodec',
}

CODEC_COLORS = {
    'encodec': '#1f77b4',  # Blue
    'semanticodec': '#ff7f0e',  # Orange
}

CODEC_MARKERS = {
    'encodec': 'o',
    'semanticodec': 's',
}


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_accuracy_vs_bitrate(
    df: pd.DataFrame,
    task: str,
    output_path: Path,
    metric: str = 'test_acc',
    show_train: bool = False,
):
    """Plot classification accuracy vs bitrate for a single task.
    
    Args:
        df: DataFrame with evaluation results
        task: HEAR task name
        output_path: Path to save figure
        metric: Metric to plot ('test_acc', 'val_acc', 'train_acc')
        show_train: Whether to show training accuracy
    """
    # Filter data for this task
    task_df = df[df['task'] == task].copy()
    
    if len(task_df) == 0:
        print(f"No data for task: {task}")
        return
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot each codec
    for codec in task_df['codec'].unique():
        codec_df = task_df[task_df['codec'] == codec].sort_values('bitrate')
        
        # Get display properties
        label = CODEC_DISPLAY_NAMES.get(codec, codec)
        color = CODEC_COLORS.get(codec, None)
        marker = CODEC_MARKERS.get(codec, 'o')
        
        # Plot test accuracy
        ax.plot(
            codec_df['bitrate'],
            codec_df[metric],
            marker=marker,
            markersize=8,
            linewidth=2,
            label=label,
            color=color,
        )
        
        # Optionally plot training accuracy (lighter)
        if show_train and 'train_acc' in codec_df.columns:
            ax.plot(
                codec_df['bitrate'],
                codec_df['train_acc'],
                marker=marker,
                markersize=6,
                linewidth=1.5,
                linestyle='--',
                alpha=0.5,
                color=color,
                label=f'{label} (train)',
            )
    
    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Bitrate (kbps)', fontweight='bold')
    ax.set_ylabel('Classification Accuracy (%)', fontweight='bold')
    
    task_name = TASK_DISPLAY_NAMES.get(task, task)
    ax.set_title(f'{task_name} Semantic Evaluation', fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(framealpha=0.9, loc='best')
    
    # Set reasonable axis limits
    ax.set_ylim(bottom=0, top=min(100, ax.get_ylim()[1] * 1.1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_path}")
    plt.close()


def plot_all_tasks_grid(
    df: pd.DataFrame,
    output_path: Path,
    metric: str = 'test_acc',
):
    """Plot accuracy vs bitrate for all tasks in a grid layout.
    
    Args:
        df: DataFrame with evaluation results
        output_path: Path to save figure
        metric: Metric to plot
    """
    tasks = df['task'].unique()
    n_tasks = len(tasks)
    
    if n_tasks == 0:
        print("No tasks to plot")
        return
    
    # Create grid layout
    n_cols = 3
    n_rows = int(np.ceil(n_tasks / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    # Plot each task
    for idx, task in enumerate(sorted(tasks)):
        ax = axes[idx]
        task_df = df[df['task'] == task].copy()
        
        # Plot each codec
        for codec in task_df['codec'].unique():
            codec_df = task_df[task_df['codec'] == codec].sort_values('bitrate')
            
            label = CODEC_DISPLAY_NAMES.get(codec, codec)
            color = CODEC_COLORS.get(codec, None)
            marker = CODEC_MARKERS.get(codec, 'o')
            
            ax.plot(
                codec_df['bitrate'],
                codec_df[metric],
                marker=marker,
                markersize=6,
                linewidth=1.5,
                label=label,
                color=color,
            )
        
        # Formatting
        ax.set_xscale('log')
        ax.set_xlabel('Bitrate (kbps)')
        ax.set_ylabel('Accuracy (%)')
        
        task_name = TASK_DISPLAY_NAMES.get(task, task)
        ax.set_title(task_name, fontweight='bold')
        
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(framealpha=0.9, fontsize=8)
        ax.set_ylim(bottom=0, top=min(100, ax.get_ylim()[1] * 1.05))
    
    # Hide unused subplots
    for idx in range(n_tasks, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved grid plot: {output_path}")
    plt.close()


def plot_aggregate_comparison(
    df: pd.DataFrame,
    output_path: Path,
    metric: str = 'test_acc',
):
    """Plot average accuracy across all tasks vs bitrate.
    
    Args:
        df: DataFrame with evaluation results
        output_path: Path to save figure
        metric: Metric to plot
    """
    # Group by codec and bitrate, average across tasks
    agg_df = df.groupby(['codec', 'bitrate'])[metric].agg(['mean', 'std']).reset_index()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Plot each codec
    for codec in agg_df['codec'].unique():
        codec_df = agg_df[agg_df['codec'] == codec].sort_values('bitrate')
        
        label = CODEC_DISPLAY_NAMES.get(codec, codec)
        color = CODEC_COLORS.get(codec, None)
        marker = CODEC_MARKERS.get(codec, 'o')
        
        # Plot mean with error bars
        ax.errorbar(
            codec_df['bitrate'],
            codec_df['mean'],
            yerr=codec_df['std'],
            marker=marker,
            markersize=8,
            linewidth=2,
            capsize=5,
            label=label,
            color=color,
        )
    
    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Bitrate (kbps)', fontweight='bold')
    ax.set_ylabel('Average Classification Accuracy (%)', fontweight='bold')
    ax.set_title('Average Performance Across All HEAR Tasks', fontweight='bold')
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(framealpha=0.9, loc='best')
    ax.set_ylim(bottom=0, top=min(100, ax.get_ylim()[1] * 1.1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved aggregate plot: {output_path}")
    plt.close()


def plot_bitrate_efficiency(
    df: pd.DataFrame,
    output_path: Path,
    metric: str = 'test_acc',
    target_accuracy: float = 80.0,
):
    """Plot bitrate required to achieve target accuracy for each task.
    
    Args:
        df: DataFrame with evaluation results
        output_path: Path to save figure
        metric: Metric to use
        target_accuracy: Target accuracy threshold
    """
    tasks = sorted(df['task'].unique())
    codecs = sorted(df['codec'].unique())
    
    # For each task and codec, find minimum bitrate to achieve target
    results = []
    
    for task in tasks:
        for codec in codecs:
            subset = df[(df['task'] == task) & (df['codec'] == codec)].copy()
            subset = subset.sort_values('bitrate')
            
            # Find first bitrate where accuracy >= target
            above_target = subset[subset[metric] >= target_accuracy]
            
            if len(above_target) > 0:
                min_bitrate = above_target['bitrate'].min()
                results.append({
                    'task': task,
                    'codec': codec,
                    'bitrate': min_bitrate,
                })
            else:
                # Not achievable
                results.append({
                    'task': task,
                    'codec': codec,
                    'bitrate': np.nan,
                })
    
    results_df = pd.DataFrame(results)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(tasks))
    width = 0.35
    
    for idx, codec in enumerate(codecs):
        codec_data = results_df[results_df['codec'] == codec]
        bitrates = [
            codec_data[codec_data['task'] == task]['bitrate'].values[0]
            for task in tasks
        ]
        
        label = CODEC_DISPLAY_NAMES.get(codec, codec)
        color = CODEC_COLORS.get(codec, None)
        
        ax.bar(
            x + idx * width,
            bitrates,
            width,
            label=label,
            color=color,
            alpha=0.8,
        )
    
    # Formatting
    ax.set_xlabel('Task', fontweight='bold')
    ax.set_ylabel('Minimum Bitrate (kbps)', fontweight='bold')
    ax.set_title(f'Bitrate Efficiency (≥{target_accuracy}% Accuracy)', fontweight='bold')
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([TASK_DISPLAY_NAMES.get(t, t) for t in tasks], rotation=45, ha='right')
    ax.legend(framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved efficiency plot: {output_path}")
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Plot semantic evaluation results'
    )
    
    parser.add_argument('--input', type=Path,
                        default=Path('datasets/analysis/semantic_metrics.csv'),
                        help='Input CSV with evaluation results')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('datasets/analysis/final_plots'),
                        help='Output directory for plots')
    parser.add_argument('--task', type=str,
                        help='Generate plot for specific task only')
    parser.add_argument('--metric', type=str, default='test_acc',
                        choices=['test_acc', 'val_acc', 'train_acc'],
                        help='Metric to plot')
    parser.add_argument('--show-train', action='store_true',
                        help='Show training accuracy on task plots')
    
    args = parser.parse_args()
    
    # Load results
    if not args.input.exists():
        print(f"Error: Results file not found: {args.input}")
        print("Run evaluate_codec_semantics.py first to generate results.")
        return
    
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} results from {args.input}")
    print(f"Codecs: {df['codec'].unique()}")
    print(f"Tasks: {df['task'].unique()}")
    print(f"Bitrates: {sorted(df['bitrate'].unique())}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    if args.task:
        # Single task plot
        output_path = args.output_dir / f'semantic_accuracy_{args.task}.png'
        plot_accuracy_vs_bitrate(df, args.task, output_path, args.metric, args.show_train)
    else:
        # Generate all plots
        print("\nGenerating plots...")
        
        # Individual task plots
        for task in df['task'].unique():
            output_path = args.output_dir / f'semantic_accuracy_{task}.png'
            plot_accuracy_vs_bitrate(df, task, output_path, args.metric, args.show_train)
        
        # Grid plot with all tasks
        grid_path = args.output_dir / 'semantic_accuracy_all_tasks.png'
        plot_all_tasks_grid(df, grid_path, args.metric)
        
        # Aggregate comparison
        aggregate_path = args.output_dir / 'semantic_accuracy_aggregate.png'
        plot_aggregate_comparison(df, aggregate_path, args.metric)
        
        # Bitrate efficiency
        efficiency_path = args.output_dir / 'bitrate_efficiency.png'
        plot_bitrate_efficiency(df, efficiency_path, args.metric, target_accuracy=80.0)
        
        print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

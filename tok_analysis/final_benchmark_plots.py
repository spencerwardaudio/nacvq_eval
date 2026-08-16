from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
_DEFAULT_ANALYSIS_ROOT = _PROJ_ROOT / "datasets" / "analysis"
_DEFAULT_OUTPUT_DIR = _DEFAULT_ANALYSIS_ROOT / "final_plots"

_STATS_RE = re.compile(r"stats_bw(?P<bw>.+)\.json$")


def _safe_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan"), "var": float("nan")}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "var": float(arr.var()),
    }


def _parse_bandwidth(path: Path) -> float | None:
    match = _STATS_RE.search(path.name)
    if not match:
        return None
    return _safe_float(match.group("bw"))


def _load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def _collect_experiment_stats(analysis_root: Path, experiment: str) -> list[dict]:
    exp_root = analysis_root / experiment
    if not exp_root.exists():
        return []

    records: list[dict] = []
    for stats_path in sorted(exp_root.rglob("stats_bw*.json")):
        bandwidth = _parse_bandwidth(stats_path)
        if bandwidth is None:
            continue
        payload = _load_json(stats_path)
        payload["_bandwidth"] = bandwidth
        payload["_stats_path"] = str(stats_path)
        records.append(payload)
    return records


def _load_metrics_csv(metrics_csv: Path | None) -> list[dict]:
    if metrics_csv is None or not metrics_csv.exists():
        return []

    rows: list[dict] = []
    with metrics_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metric = (row.get("metric") or "").strip().lower()
            value = _safe_float(row.get("value"))
            bitrate_realized = _safe_float(row.get("bitrate_realized"))
            bitrate_target = _safe_float(row.get("bitrate_target"))
            bitrate = bitrate_realized if bitrate_realized is not None else bitrate_target
            if not metric or value is None or bitrate is None:
                continue
            row["metric"] = metric
            row["value"] = value
            row["bitrate"] = bitrate
            row["codec"] = (row.get("codec") or "unknown").strip()
            row["dataset"] = (row.get("dataset") or "unknown").strip()
            row["waveform_family"] = (row.get("waveform_family") or "unknown").strip()
            row["perturbation"] = (row.get("perturbation") or "unknown").strip()
            rows.append(row)
    return rows


def _group_metric_rows(rows: list[dict], metric_names: set[str]) -> dict[str, dict[float, dict[str, float]]]:
    grouped: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["metric"] not in metric_names:
            continue
        grouped[row["codec"]][row["bitrate"]].append(row["value"])

    summarized: dict[str, dict[float, dict[str, float]]] = {}
    for codec, by_bitrate in grouped.items():
        summarized[codec] = {bitrate: _stats(values) for bitrate, values in by_bitrate.items()}
    return summarized


def _render_missing(ax: plt.Axes, title: str, detail: str) -> None:
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, detail, ha="center", va="center", fontsize=11, wrap=True)


def _annotate_stats(ax: plt.Axes, entries: list[tuple[str, float, float, float]]) -> None:
    if not entries:
        return
    lines = ["codec  mean  std  var"]
    for codec, mean, std, var in entries:
        lines.append(f"{codec[:12]:<12} {mean:>5.3f} {std:>5.3f} {var:>5.3f}")
    ax.text(
        0.99,
        0.02,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        family="monospace",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )


def _plot_metric_panel(
    ax: plt.Axes,
    grouped: dict[str, dict[float, dict[str, float]]],
    title: str,
    ylabel: str,
    metric_hint: str,
) -> None:
    if not grouped:
        _render_missing(ax, title, f"No {metric_hint} rows found in benchmark metrics CSV.")
        return

    summary_rows: list[tuple[str, float, float, float]] = []
    for codec in sorted(grouped):
        by_bitrate = grouped[codec]
        xs = sorted(by_bitrate)
        means = [by_bitrate[x]["mean"] for x in xs]
        stds = [by_bitrate[x]["std"] for x in xs]
        ax.errorbar(xs, means, yerr=stds, marker="o", linewidth=1.7, capsize=3, label=codec)
        flat_means = [by_bitrate[x]["mean"] for x in xs]
        flat_stds = [by_bitrate[x]["std"] for x in xs]
        flat_vars = [by_bitrate[x]["var"] for x in xs]
        summary_rows.append((codec, float(np.mean(flat_means)), float(np.mean(flat_stds)), float(np.mean(flat_vars))))

    ax.set_title(title)
    ax.set_xlabel("Bitrate (kbps)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _annotate_stats(ax, summary_rows)


def _aggregate_phase_records(records: list[dict]) -> dict[float, dict[str, list[float]]]:
    phase_by_bw: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        bandwidth = record["_bandwidth"]
        for tag in record.get("variant_tags", []):
            label = f"{int(float(tag))} deg"
            mean_value = record.get("mean_flip_rate_by_variant", {}).get(tag)
            if mean_value is not None:
                phase_by_bw[bandwidth][label].append(float(mean_value))
    return phase_by_bw


def _aggregate_self_amp_records(records: list[dict]) -> tuple[dict[float, dict[str, list[float]]], dict[float, list[float]], dict[float, list[float]]]:
    bins_by_bw: dict[float, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    low_by_bw: dict[float, list[float]] = defaultdict(list)
    overall_by_bw: dict[float, list[float]] = defaultdict(list)
    for record in records:
        bandwidth = record["_bandwidth"]
        freq_hz = int(record.get("freq_hz", -1))
        freq_label = f"{freq_hz} Hz"
        means = record.get("mean_flip_rate_by_variant", {})
        if means:
            values = [float(v) for v in means.values()]
            avg = float(np.mean(values))
            bins_by_bw[bandwidth][freq_label].append(avg)
            overall_by_bw[bandwidth].append(avg)
            if 0 < freq_hz < 200:
                low_by_bw[bandwidth].append(avg)
    return bins_by_bw, low_by_bw, overall_by_bw


def _aggregate_frequency_records(records: list[dict]) -> tuple[dict[float, list[float]], dict[float, list[float]]]:
    low_by_bw: dict[float, list[float]] = defaultdict(list)
    overall_by_bw: dict[float, list[float]] = defaultdict(list)
    for record in records:
        bandwidth = record["_bandwidth"]
        tags = record.get("variant_tags", [])
        means = record.get("mean_flip_rate_by_variant", {})
        for tag in tags:
            value = means.get(tag)
            if value is None:
                continue
            freq_hz = int(float(tag))
            value = float(value)
            overall_by_bw[bandwidth].append(value)
            if freq_hz < 200:
                low_by_bw[bandwidth].append(value)
    return low_by_bw, overall_by_bw


def _heatmap_from_mapping(
    ax: plt.Axes,
    mapping: dict[float, dict[str, list[float]]],
    title: str,
    ylabel: str,
) -> None:
    if not mapping:
        _render_missing(ax, title, "No analyzer JSON found for this plot.")
        return

    x_values = sorted(mapping)
    y_labels = sorted({label for rows in mapping.values() for label in rows})
    matrix = np.full((len(y_labels), len(x_values)), np.nan)
    annot: list[tuple[str, float, float, float]] = []

    for xi, bitrate in enumerate(x_values):
        collected_for_bw: list[float] = []
        for yi, label in enumerate(y_labels):
            values = mapping[bitrate].get(label, [])
            if values:
                matrix[yi, xi] = float(np.mean(values))
                collected_for_bw.extend(values)
        if collected_for_bw:
            stats = _stats(collected_for_bw)
            annot.append((f"{bitrate:g}kbps", stats["mean"], stats["std"], stats["var"]))

    shown = np.where(np.isnan(matrix), 0.0, matrix)
    im = ax.imshow(shown, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(x_values)))
    ax.set_xticklabels([f"{x:g}" for x in x_values])
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel("Bitrate (kbps)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="Mean flip rate")
    _annotate_stats(ax, annot)


def create_reconstruction_plot(rows: list[dict], output_path: Path) -> None:
    grouped = {
        "SI-SNR": _group_metric_rows(rows, {"si-snr", "si_snr"}),
        "STOI": _group_metric_rows(rows, {"stoi"}),
        "ViSQOL": _group_metric_rows(rows, {"visqol", "visqol_moslqo"}),
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    _plot_metric_panel(axes[0], grouped["SI-SNR"], "Reconstruction Quality vs Bitrate: SI-SNR", "SI-SNR (dB)", "SI-SNR")
    _plot_metric_panel(axes[1], grouped["STOI"], "Reconstruction Quality vs Bitrate: STOI", "STOI", "STOI")
    _plot_metric_panel(axes[2], grouped["ViSQOL"], "Reconstruction Quality vs Bitrate: ViSQOL", "ViSQOL", "ViSQOL")
    fig.suptitle("Final Plot 1: Reconstruction Quality vs Bitrate", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_semantic_plot(rows: list[dict], output_path: Path) -> None:
    grouped = {
        "WER": _group_metric_rows(rows, {"wer"}),
        "Classification Accuracy": _group_metric_rows(rows, {"classification_accuracy", "accuracy"}),
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _plot_metric_panel(axes[0], grouped["WER"], "Semantic Robustness vs Bitrate: Speech WER", "WER", "WER")
    _plot_metric_panel(axes[1], grouped["Classification Accuracy"], "Semantic Robustness vs Bitrate: General-Sound Classification Accuracy", "Accuracy", "classification accuracy")
    fig.suptitle("Final Plot 2: Semantic Robustness vs Bitrate", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_stability_plot(analysis_root: Path, output_path: Path) -> None:
    phase_records = _collect_experiment_stats(analysis_root, "dsp_self_phase")
    self_amp_records = _collect_experiment_stats(analysis_root, "dsp_self_amp")
    phase_map = _aggregate_phase_records(phase_records)
    self_amp_map, _, _ = _aggregate_self_amp_records(self_amp_records)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    _heatmap_from_mapping(
        axes[0],
        phase_map,
        "Phase Sweep Stability Heatmap",
        "Phase Offset Bin",
    )
    _heatmap_from_mapping(
        axes[1],
        self_amp_map,
        "Self-Amplitude vs Frequency Stability Heatmap",
        "Frequency Bin",
    )
    fig.suptitle(
        "Final Plot 3: Stability Heatmap for Phase and Self-Amplitude-vs-Frequency Sweeps",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def create_low_frequency_plot(analysis_root: Path, output_path: Path) -> None:
    self_amp_records = _collect_experiment_stats(analysis_root, "dsp_self_amp")
    frequency_records = _collect_experiment_stats(analysis_root, "dsp_frequency")
    _, self_amp_low, self_amp_overall = _aggregate_self_amp_records(self_amp_records)
    freq_low, freq_overall = _aggregate_frequency_records(frequency_records)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for ax, low_map, overall_map, title in [
        (axes[0], self_amp_low, self_amp_overall, "Low-Frequency Diagnostic from Self-Amplitude Sweep (<200 Hz)"),
        (axes[1], freq_low, freq_overall, "Low-Frequency Diagnostic from Frequency Sweep (<200 Hz)"),
    ]:
        if not low_map and not overall_map:
            _render_missing(ax, title, "No low-frequency analyzer JSON found for this panel.")
            continue

        x_values = sorted(set(low_map) | set(overall_map))
        low_mean = [float(np.mean(low_map[x])) if low_map.get(x) else float("nan") for x in x_values]
        low_std = [float(np.std(low_map[x])) if low_map.get(x) else float("nan") for x in x_values]
        overall_mean = [float(np.mean(overall_map[x])) if overall_map.get(x) else float("nan") for x in x_values]
        ax.errorbar(x_values, low_mean, yerr=low_std, marker="o", linewidth=1.7, capsize=3, label="<200 Hz")
        ax.plot(x_values, overall_mean, marker="s", linewidth=1.5, linestyle="--", label="overall")
        ax.set_title(title)
        ax.set_xlabel("Bitrate (kbps)")
        ax.set_ylabel("Mean flip rate")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        combined = []
        for x in x_values:
            combined.extend(low_map.get(x, []))
        if combined:
            stats = _stats(combined)
            _annotate_stats(ax, [("<200Hz", stats["mean"], stats["std"], stats["var"])])

    fig.suptitle("Final Plot 4: Low-Frequency Diagnostic (<200 Hz)", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate exactly four final benchmark PNG plots.")
    parser.add_argument("--analysis-root", type=Path, default=_DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--metrics-csv", type=Path, default=None,
                        help="Optional normalized benchmark metrics CSV for reconstruction/semantic plots.")
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_metrics_csv(args.metrics_csv)
    create_reconstruction_plot(rows, args.output_dir / "reconstruction_quality_vs_bitrate.png")
    create_semantic_plot(rows, args.output_dir / "semantic_robustness_vs_bitrate.png")
    create_stability_plot(args.analysis_root, args.output_dir / "stability_heatmap_phase_selfampfreq.png")
    create_low_frequency_plot(args.analysis_root, args.output_dir / "lowfreq_diagnostic_lt200hz.png")

    print("Wrote final benchmark plots to", args.output_dir)


if __name__ == "__main__":
    main()
import matplotlib
matplotlib.use("Agg")
import argparse
import json
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_TABLES_DIR = (Path(__file__).parent / "../results/codec_results").resolve()
PROJECT_ROOT = Path(__file__).resolve().parent.parent

COLORS = {
    "encodec": "#1f77b4",
    "q2d2": "#ff7f0e",
    "hificodec": "#2ca02c",
    "speechtokenizer": "#d62728",
    "dac_fsq": "#9467bd",
}

def _load_checkpoint_manifest(tables_dir: Path) -> dict[str, dict[str, str | None]]:
    candidates = [
        tables_dir / "checkpoints_used.json",
        PROJECT_ROOT / "datasets" / "analysis" / "tables" / "checkpoints_used.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return json.loads(candidate.read_text())
    return {}

def _format_checkpoint_footer(manifest: dict[str, dict[str, str | None]]) -> str:
    if not manifest:
        return "Checkpoint titles used: unavailable"

    ordered = ["encodec", "q2d2", "hificodec", "speechtokenizer", "dac_fsq"]
    labels = []
    for codec in ordered:
        item = manifest.get(codec)
        if not item or not item.get("title"):
            continue
        labels.append(f"{codec}: {item['title']}")
    text = "Checkpoint titles used: " + " | ".join(labels)
    return textwrap.fill(text, width=150)

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tables-dir", default=str(DEFAULT_TABLES_DIR))
    ap.add_argument("--output", default="")
    ap.add_argument("--title", default="Codec Results Summary")
    args = ap.parse_args()

    tables_dir = Path(args.tables_dir).resolve()
    out = Path(args.output).resolve() if args.output else tables_dir / "all_tables_summary.png"
    manifest = _load_checkpoint_manifest(tables_dir)

    fig, axes = plt.subplots(4, 2, figsize=(16, 21.5))
    fig.suptitle(args.title, fontsize=14, y=0.995)

    ax = axes[0, 0]
    df = pd.read_csv(tables_dir / "table1_amplitude_response.csv")
    for model, grp in df.groupby("model"):
        first_unit = grp["unit"].unique()[0]
        sub = grp[grp["unit"] == first_unit].dropna(subset=["tfr_mean"])
        ax.plot(sub["dbfs_attenuation"], sub["tfr_mean"], "o-", label=model, color=COLORS.get(model, "gray"))
    ax.set_xlabel("dBFS Attenuation")
    ax.set_ylabel("TFR (mean)")
    ax.set_title("Table 1: Amplitude Response (Unit 1)")
    ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)

    ax = axes[0, 1]
    df = pd.read_csv(tables_dir / "table2_phase_sensitivity.csv")
    for model, grp in df.groupby("model"):
        first_unit = grp["unit"].unique()[0]
        sub = grp[(grp["unit"] == first_unit) & (grp["phase_deg"] < 360)]
        ax.plot(sub["phase_deg"], sub["tfr_mean"], "o-", label=model, color=COLORS.get(model, "gray"))
    ax.set_xlabel("Phase Shift (degrees)")
    ax.set_ylabel("TFR (mean)")
    ax.set_title("Table 2: Phase Sensitivity (Unit 1)")
    ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    df = pd.read_csv(tables_dir / "table3_temporal_offset.csv")
    for model, grp in df.groupby("model"):
        first_unit = grp["unit"].unique()[0]
        sub = grp[grp["unit"] == first_unit]
        ax.plot(sub["offset_ms"], sub["tfr_mean"], "o-", label=model, color=COLORS.get(model, "gray"))
    ax.set_xlabel("Time Offset (ms)")
    ax.set_ylabel("TFR (mean)")
    ax.set_title("Table 3: Temporal Offset (Unit 1)")
    ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.05); ax.set_xscale("log"); ax.grid(alpha=0.3)

    ax = axes[1, 1]
    df = pd.read_csv(tables_dir / "table4_centroid_magnitude.csv")
    for model, grp in df.groupby("model"):
        ax.plot(range(len(grp)), grp["mean_mag"].values, "-", label=model, color=COLORS.get(model, "gray"))
    ax.set_xlabel("Codebook Unit Index")
    ax.set_ylabel("Mean Centroid Magnitude")
    ax.set_title("Table 4: Centroid Magnitude by Depth")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[2, 0]
    df = pd.read_csv(tables_dir / "table5_svd_evr2.csv")
    for model, grp in df.groupby("model"):
        valid = grp.dropna(subset=["evr2_phase_pct"])
        ax.plot(range(len(valid)), valid["evr2_phase_pct"].values, "o-", label=model, color=COLORS.get(model, "gray"))
    ax.set_xlabel("Codebook Unit Index")
    ax.set_ylabel("EVR₂ Phase (%)")
    ax.set_title("Table 5: SVD EVR₂ (Phase) by Depth")
    ax.legend(fontsize=8); ax.set_ylim(0, 105); ax.grid(alpha=0.3)

    ax = axes[2, 1]
    df = pd.read_csv(tables_dir / "table6_egfx_response.csv")
    categories = df["effect_category"].unique()
    x = np.arange(len(categories))
    for model, grp in df.groupby("model"):
        vals = [grp[grp["effect_category"] == c]["tfr_mean"].values[0] for c in categories]
        ax.plot(x, vals, "o-", label=model, color=COLORS.get(model, "gray"))
    ax.set_xticks(x); ax.set_xticklabels(categories)
    ax.set_ylabel("TFR (mean)")
    ax.set_title("Table 6: EGFX Response by Category")
    ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)

    ax = axes[3, 0]
    df = pd.read_csv(tables_dir / "table7_cross_codec_summary.csv")
    metrics = ["amp_tfr_cb1_at_40dBFS", "phase_tfr_cb1_at_90deg", "temporal_tfr_cb1_at_10ms", "egfx_tfr_mean"]
    labels = ["Amp@40dB", "Phase@90°", "Temp@10ms", "EGFX"]
    x = np.arange(len(metrics))
    width = 0.15
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [float(row[m]) if pd.notna(row[m]) else 0 for m in metrics]
        ax.bar(x + i * width, vals, width, label=row["model"], color=COLORS.get(row["model"], "gray"))
    ax.set_xticks(x + width * 2); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("TFR")
    ax.set_title("Table 7: Cross-Codec Summary")
    ax.legend(fontsize=7); ax.set_ylim(0, 1.1); ax.grid(alpha=0.3, axis="y")

    ax = axes[3, 1]
    df = pd.read_csv(tables_dir / "table8_perplexity_by_unit.csv")
    baseline = df[(df["test"] == "amplitude") & (df["condition"] == 0)]
    for model, grp in baseline.groupby("model"):
        grp_dedup = grp.drop_duplicates(subset=["unit"])
        ax.plot(range(len(grp_dedup)), grp_dedup["perplexity_mean"].values, "o-", ms=3, label=model, color=COLORS.get(model, "gray"))
    ax.set_xlabel("Codebook Unit Index")
    ax.set_ylabel("Perplexity")
    ax.set_title("Table 8: Baseline Perplexity by Depth")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    fig.text(0.01, 0.01, _format_checkpoint_footer(manifest), ha="left", va="bottom", fontsize=8)
    plt.tight_layout(rect=(0, 0.05, 1, 0.985))
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
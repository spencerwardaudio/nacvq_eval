"""Generate analysis CSV tables from completed analysis outputs.

Reads pre-computed token .npy files (token flip rates), SVD EVR JSON files,
and EGFx metrics JSON. No model loading required.

Tables written to <output-dir>/table{N}_{name}.csv:
  1. table1_amplitude_response.csv   — TFR by dBFS level per model/unit
  2. table2_phase_sensitivity.csv    — TFR by phase angle per model/unit
  3. table3_temporal_offset.csv      — TFR by offset ms per model/unit
  4. table4_centroid_magnitude.csv   — (skipped if no centroid JSON found)
  5. table5_egfx_response.csv        — TFR/L2/cosim by effect category
  6. table6_cross_codec_summary.csv  — aggregate 5-row synthesis
  7. table7_perplexity_by_unit.csv   — perplexity by model/unit/condition

Representative units per model
  encodec        (32 CB)  : 0,7,15,23,31  → CB-1/8/16/24/32
  speechtokenizer (8 CB)  : 0,1,3,7       → CB-1/2/4/8
  hificodec       (4 CB)  : 0,1,2,3       → CB-1/2/3/4
  dac_fsq         (1 CB)  : 0             → FSQ-flat
  q2d2            (16 GP) : 0,3,7,11,15   → GP-1/4/8/12/16

Phase conditions sampled:  15°, 30°, 90°, 180°, 270°, 360°  (0° excluded)
Amplitude conditions:      0, 40, 80, 140 dBFS attenuation
Temporal conditions:       1, 5, 10, 20, 100 ms

Usage:
    python tok_analysis/generate_dissertation_tables.py \\
        --tokens-root   datasets/audio_tokens \\
        --egfx-metrics  datasets/analysis/egfx_metrics.json \\
        --output-dir    datasets/analysis/tables
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
_DS = _PROJ_ROOT / "datasets"

# ---------------------------------------------------------------------------
# Codec registry
# ---------------------------------------------------------------------------

CODEC_META: dict[str, dict] = {
    "encodec": {
        "bw_tag":   "24.0",
        "n_units":  32,
        "unit_label": "Codebook",
        "rep_idx":  [0, 7, 15, 23, 31],
        "amp_root": "dsp_self_amp",
        "phase_root": "dsp_self_phase",
        "temp_root": "time_sine",
        "subdir":   "",          # flat: audio_tokens/dsp_self_amp/
        "vocab_size": 1024,
    },
    "q2d2": {
        "bw_tag":   "9.8",
        "n_units":  16,
        "unit_label": "Grid Pair",
        "rep_idx":  [0, 3, 7, 11, 15],
        "amp_root": "dsp_self_amp",
        "phase_root": "dsp_self_phase",
        "temp_root": "time_sine",
        "subdir":   "",
        "vocab_size": [81]*5 + [49]*11,  # 9.8 kbps: streams 1-5 = 9×9, 6-16 = 7×7
    },
    "hificodec": {
        "bw_tag":   "HFC",
        "n_units":  4,
        "unit_label": "Codebook",
        "rep_idx":  [0, 1, 2, 3],
        "amp_root": "dsp_self_amp",
        "phase_root": "dsp_self_phase",
        "temp_root": "time_sine",
        "subdir":   "hificodec",
        "vocab_size": 1024,
    },
    "speechtokenizer": {
        "bw_tag":   "ST",
        "n_units":  8,
        "unit_label": "RVQ Codebook",
        "rep_idx":  [0, 1, 3, 7],
        "amp_root": "dsp_self_amp",
        "phase_root": "dsp_self_phase",
        "temp_root": "time_sine",
        "subdir":   "speechtokenizer",
        "vocab_size": 1024,
    },
    "dac_fsq": {
        "bw_tag":   "FSQ",
        "n_units":  1,
        "unit_label": "FSQ Stream",
        "rep_idx":  [0],
        "amp_root": "dsp_self_amp",
        "phase_root": "dsp_self_phase",
        "temp_root": "time_sine",
        "subdir":   "dac_fsq",
        "vocab_size": 2_560_000,  # [8,8,8,8,5,5,5,5] mixed-radix flat index
    },
}

# Conditions to include in each table
AMP_TAGS = ["0", "40", "80", "140"]        # dBFS attenuation levels
PHASE_TAGS = ["15", "30", "90", "180", "270", "360"]
TEMPORAL_TAGS_MS = ["1", "5", "10", "20", "100"]

_VAR_RE = re.compile(r"_var_([^_]+)_bw")


def _codebook_perplexity(
    tokens: np.ndarray,
    vocab_size: "int | list[int]" = 1024,
) -> np.ndarray:
    """Per-unit perplexity exp(H) from integer token indices [n_units, T]."""
    n_units = tokens.shape[0]
    perps = np.zeros(n_units, dtype=np.float64)
    vs_list = vocab_size if isinstance(vocab_size, list) else [vocab_size] * n_units
    for unit in range(n_units):
        vs = vs_list[min(unit, len(vs_list) - 1)]
        flat = tokens[unit].reshape(-1).astype(np.int64, copy=False)
        flat = np.clip(flat, 0, vs - 1)
        counts = np.bincount(flat, minlength=vs).astype(np.float64)
        total = counts.sum()
        if total <= 0:
            perps[unit] = 1.0
            continue
        probs = counts / total
        nz = probs[probs > 0]
        entropy = -np.sum(nz * np.log(nz))
        perps[unit] = float(np.exp(entropy))
    return perps


# ---------------------------------------------------------------------------
# Token loading helpers
# ---------------------------------------------------------------------------

def _tokens_root(tokens_root: Path, meta: dict) -> Path:
    subdir = meta["subdir"]
    if subdir:
        return tokens_root / subdir
    return tokens_root


def _load_npy(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(str(path)).astype(np.int64, copy=False)


def _tfr_per_unit(base: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Token Flip Rate per codebook/unit. Returns [n_units] float array."""
    n = min(base.shape[1], var.shape[1])
    return (base[:, :n] != var[:, :n]).mean(axis=1).astype(np.float64)


def _collect_tfr(
    tokens_root: Path,
    meta: dict,
    test_dir_name: str,
    condition_tags: list[str],
) -> dict[str, list[float]]:
    """Average TFR per unit across all frequencies, for each condition tag.

    Returns {tag: [mean_tfr_per_unit]} where each value is length n_units.
    Returns empty dict if no token dirs found.
    """
    root = _tokens_root(tokens_root, meta) / test_dir_name
    if not root.exists():
        return {}

    bw = meta["bw_tag"]
    n_units = meta["n_units"]
    tag_accum: dict[str, list[np.ndarray]] = {t: [] for t in condition_tags}

    for sig_dir in sorted(root.iterdir()):
        if not sig_dir.is_dir():
            continue
        signal = sig_dir.name

        # time_sine tokens use baseline_0ms/offset_NNNms naming; all others use signal-prefixed naming
        if test_dir_name == "time_sine":
            base_path = sig_dir / f"baseline_0ms_bw{bw}_tokens.npy"
        else:
            base_path = sig_dir / f"{signal}_baseline_bw{bw}_tokens.npy"
        base = _load_npy(base_path)
        if base is None:
            continue
        if base.shape[0] != n_units:
            continue  # shape mismatch — skip

        for tag in condition_tags:
            if test_dir_name == "time_sine":
                try:
                    var_path = sig_dir / f"offset_{int(tag):03d}ms_bw{bw}_tokens.npy"
                except ValueError:
                    continue
            else:
                var_path = sig_dir / f"{signal}_var_{tag}_bw{bw}_tokens.npy"
            var = _load_npy(var_path)
            if var is None:
                continue
            tfr = _tfr_per_unit(base, var)
            tag_accum[tag].append(tfr)

    result: dict[str, list[float]] = {}
    for tag, arrays in tag_accum.items():
        if arrays:
            mean_tfr = np.mean(np.stack(arrays, axis=0), axis=0)  # [n_units]
            result[tag] = mean_tfr.tolist()
    return result


def _collect_perplexity_amplitude(
    tokens_root: Path,
    meta: dict,
    condition_tags: list[str],
    vocab_size: int = 1024,
) -> dict[str, list[float]]:
    """Average per-unit perplexity for dsp_self_amp by dBFS condition.

    Returns keys for baseline "0" and requested variant tags.
    """
    root = _tokens_root(tokens_root, meta) / "dsp_self_amp"
    if not root.exists():
        return {}

    bw = meta["bw_tag"]
    n_units = meta["n_units"]
    tags = ["0"] + [str(t) for t in condition_tags]
    accum: dict[str, list[np.ndarray]] = {t: [] for t in tags}

    for sig_dir in sorted(root.iterdir()):
        if not sig_dir.is_dir():
            continue
        signal = sig_dir.name
        base_path = sig_dir / f"{signal}_baseline_bw{bw}_tokens.npy"
        base = _load_npy(base_path)
        if base is None or base.shape[0] != n_units:
            continue
        accum["0"].append(_codebook_perplexity(base, vocab_size=vocab_size))

        for tag in condition_tags:
            var_path = sig_dir / f"{signal}_var_{tag}_bw{bw}_tokens.npy"
            var = _load_npy(var_path)
            if var is None or var.shape[0] != n_units:
                continue
            accum[str(tag)].append(_codebook_perplexity(var, vocab_size=vocab_size))

    out: dict[str, list[float]] = {}
    for tag, arrays in accum.items():
        if arrays:
            out[tag] = np.mean(np.stack(arrays, axis=0), axis=0).tolist()
    return out


def _collect_perplexity_phase(
    tokens_root: Path,
    meta: dict,
    condition_tags: list[str],
    vocab_size: int = 1024,
) -> dict[str, list[float]]:
    """Average per-unit perplexity for dsp_self_phase by phase condition.

    Returns keys for baseline "0" and requested phase degree tags.
    """
    root = _tokens_root(tokens_root, meta) / "dsp_self_phase"
    if not root.exists():
        return {}

    bw = meta["bw_tag"]
    n_units = meta["n_units"]
    tags = ["0"] + [str(t) for t in condition_tags]
    accum: dict[str, list[np.ndarray]] = {t: [] for t in tags}

    for sig_dir in sorted(root.iterdir()):
        if not sig_dir.is_dir():
            continue
        signal = sig_dir.name
        base_path = sig_dir / f"{signal}_baseline_bw{bw}_tokens.npy"
        base = _load_npy(base_path)
        if base is None or base.shape[0] != n_units:
            continue
        accum["0"].append(_codebook_perplexity(base, vocab_size=vocab_size))

        for tag in condition_tags:
            var_path = sig_dir / f"{signal}_var_{tag}_bw{bw}_tokens.npy"
            var = _load_npy(var_path)
            if var is None or var.shape[0] != n_units:
                continue
            accum[str(tag)].append(_codebook_perplexity(var, vocab_size=vocab_size))

    out: dict[str, list[float]] = {}
    for tag, arrays in accum.items():
        if arrays:
            out[tag] = np.mean(np.stack(arrays, axis=0), axis=0).tolist()
    return out


def _collect_perplexity_temporal(
    tokens_root: Path,
    meta: dict,
    condition_tags_ms: list[str],
    vocab_size: int = 1024,
) -> dict[str, list[float]]:
    """Average per-unit perplexity for time_sine by offset condition.

    Returns keys for baseline "0" and requested offset-ms tags.
    """
    root = _tokens_root(tokens_root, meta) / "time_sine"
    if not root.exists():
        return {}

    bw = meta["bw_tag"]
    n_units = meta["n_units"]
    tags = ["0"] + [str(t) for t in condition_tags_ms]
    accum: dict[str, list[np.ndarray]] = {t: [] for t in tags}

    for freq_dir in sorted(root.iterdir()):
        if not freq_dir.is_dir():
            continue
        base_path = freq_dir / f"baseline_0ms_bw{bw}_tokens.npy"
        base = _load_npy(base_path)
        if base is None or base.shape[0] != n_units:
            continue
        accum["0"].append(_codebook_perplexity(base, vocab_size=vocab_size))

        for tag in condition_tags_ms:
            try:
                offset_ms = int(tag)
            except ValueError:
                continue
            var_path = freq_dir / f"offset_{offset_ms:03d}ms_bw{bw}_tokens.npy"
            var = _load_npy(var_path)
            if var is None or var.shape[0] != n_units:
                continue
            accum[str(tag)].append(_codebook_perplexity(var, vocab_size=vocab_size))

    out: dict[str, list[float]] = {}
    for tag, arrays in accum.items():
        if arrays:
            out[tag] = np.mean(np.stack(arrays, axis=0), axis=0).tolist()
    return out


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_sinusoid_table(
    tokens_root: Path,
    test_dir: str,
    condition_tags: list[str],
    condition_col: str,
) -> pd.DataFrame:
    """Build TFR table for a sinusoid test (amp / phase / temporal).

    Rows: one per (model, unit_label, condition).
    Columns: model, unit, {condition_col}, tfr_mean.
    """
    rows = []
    for codec, meta in CODEC_META.items():
        tfr_by_tag = _collect_tfr(tokens_root, meta, test_dir, condition_tags)
        if not tfr_by_tag:
            print(f"  [WARN] {codec}: no token data found in {test_dir} — rows will be NaN")

        rep_idx = meta["rep_idx"]
        label = meta["unit_label"]
        n_units = meta["n_units"]

        for unit_pos in rep_idx:
            unit_name = f"{label} {unit_pos + 1}"
            for tag in condition_tags:
                tfr_arr = tfr_by_tag.get(tag)
                if tfr_arr is not None and unit_pos < len(tfr_arr):
                    tfr_val = tfr_arr[unit_pos]
                else:
                    tfr_val = float("nan")
                rows.append({
                    "model": codec,
                    "unit": unit_name,
                    condition_col: tag,
                    "tfr_mean": round(tfr_val, 6),
                })

    df = pd.DataFrame(rows)
    assert not df.empty, f"build_sinusoid_table: zero rows for test={test_dir}"
    return df


def build_svd_table(svd_root: Path) -> pd.DataFrame:
    """Build Table 5 from SVD(ΔZ) JSON files.

    Expects: svd_root/{codec}/svd_stats_bw{bw}.json
    Columns: model, unit, evr2_phase_pct, eff_rank_phase,
             evr1_amp_pct, eff_rank_amp, evr1_temporal_pct, eff_rank_temporal
    """
    rows = []
    for codec, meta in CODEC_META.items():
        stats_path = svd_root / codec / f"svd_stats_bw{meta['bw_tag']}.json"
        if not stats_path.exists():
            print(f"  [WARN] SVD stats not found: {stats_path}")
            continue
        try:
            with stats_path.open() as f:
                data = json.load(f)
        except Exception as exc:
            print(f"  [WARN] Could not read {stats_path}: {exc}")
            continue

        if data.get("error"):
            print(f"  [WARN] {codec}: {data['error']}")
            continue

        label   = meta["unit_label"]
        rep_idx = meta["rep_idx"]

        phase_evr2  = data.get("phase",     {}).get("evr2",     [])
        phase_er    = data.get("phase",     {}).get("eff_rank", [])
        amp_evr1    = data.get("amplitude", {}).get("evr1",     [])
        amp_er      = data.get("amplitude", {}).get("eff_rank", [])
        temp_evr1   = data.get("temporal",  {}).get("evr1",     [])
        temp_er     = data.get("temporal",  {}).get("eff_rank", [])

        def _get(lst: list, idx: int) -> float:
            return float(lst[idx]) if idx < len(lst) else float("nan")

        for unit_pos in rep_idx:
            rows.append({
                "model":              codec,
                "unit":               f"{label} {unit_pos + 1}",
                "evr2_phase_pct":     round(_get(phase_evr2, unit_pos) * 100, 2),
                "eff_rank_phase":     round(_get(phase_er,   unit_pos), 3),
                "evr1_amp_pct":       round(_get(amp_evr1,   unit_pos) * 100, 2),
                "eff_rank_amp":       round(_get(amp_er,     unit_pos), 3),
                "evr1_temporal_pct":  round(_get(temp_evr1,  unit_pos) * 100, 2),
                "eff_rank_temporal":  round(_get(temp_er,    unit_pos), 3),
            })

    return pd.DataFrame(rows)


def build_egfx_table(egfx_metrics_path: Path) -> pd.DataFrame:
    """Build Table 6 from egfx_metrics.json."""
    if not egfx_metrics_path.exists():
        print(f"  [WARN] EGFx metrics not found: {egfx_metrics_path}")
        return pd.DataFrame()

    with egfx_metrics_path.open() as f:
        data = json.load(f)

    rows = []
    for codec, categories in data.items():
        if codec not in CODEC_META:
            continue
        for category, metrics in categories.items():
            assert isinstance(metrics, (dict, list)), (
                f"Unexpected egfx_metrics structure for {codec}/{category}: "
                f"got {type(metrics).__name__}. Re-check egfx_metrics.py output format."
            )
            # metrics may be keyed by layer or be flat dicts
            if isinstance(metrics, dict) and "tfr" in metrics:
                # Flat dict: {tfr: ..., l2: ..., cosim: ...}
                rows.append({
                    "model": codec,
                    "effect_category": category,
                    "tfr_mean": round(float(metrics.get("tfr", float("nan"))), 6),
                    "l2_mean": round(float(metrics.get("l2", float("nan"))), 6),
                    "cosim_mean": round(float(metrics.get("cosim", float("nan"))), 6),
                })
            elif isinstance(metrics, dict):
                # Keyed by layer — average across layers
                tfr_vals, l2_vals, cosim_vals = [], [], []
                for layer_data in metrics.values():
                    if isinstance(layer_data, dict):
                        if "tfr" in layer_data:
                            tfr_vals.append(float(layer_data["tfr"]))
                        if "l2" in layer_data:
                            l2_vals.append(float(layer_data["l2"]))
                        if "cosim" in layer_data:
                            cosim_vals.append(float(layer_data["cosim"]))
                rows.append({
                    "model": codec,
                    "effect_category": category,
                    "tfr_mean": round(np.mean(tfr_vals) if tfr_vals else float("nan"), 6),
                    "l2_mean": round(np.mean(l2_vals) if l2_vals else float("nan"), 6),
                    "cosim_mean": round(np.mean(cosim_vals) if cosim_vals else float("nan"), 6),
                })
            elif isinstance(metrics, list) and metrics:
                # actual egfx_metrics.py output: list of per-pair dicts, each value a per-layer list
                tfr_vals, l2_vals, cosim_vals = [], [], []
                for pair in metrics:
                    if not isinstance(pair, dict):
                        continue
                    for key, acc in [("tfr", tfr_vals), ("l2_distance", l2_vals), ("cosine_similarity", cosim_vals)]:
                        v = pair.get(key)
                        if v is not None:
                            acc.append(float(np.mean(v)) if isinstance(v, list) else float(v))
                rows.append({
                    "model": codec,
                    "effect_category": category,
                    "tfr_mean":   round(np.mean(tfr_vals)   if tfr_vals   else float("nan"), 6),
                    "l2_mean":    round(np.mean(l2_vals)     if l2_vals    else float("nan"), 6),
                    "cosim_mean": round(np.mean(cosim_vals)  if cosim_vals else float("nan"), 6),
                })

    return pd.DataFrame(rows)


def _load_svd_stats_for_codec(codec: str, t5: pd.DataFrame) -> dict:
    """Extract per-codec SVD scalars from the pre-built SVD table and JSON."""
    _ds = Path(__file__).resolve().parent.parent / "datasets"
    meta  = CODEC_META[codec]
    bw    = meta["bw_tag"]
    # Try to read raw JSON for depth slopes (not stored in per-unit table rows)
    json_path = _ds / "analysis" / "svd" / codec / f"svd_stats_bw{bw}.json"
    slope_phase = float("nan")
    slope_amp   = float("nan")
    if json_path.exists():
        try:
            with json_path.open() as f:
                raw = json.load(f)
            slope_phase = float(raw.get("phase",     {}).get("depth_evr2_slope", float("nan")))
            slope_amp   = float(raw.get("amplitude", {}).get("depth_evr2_slope", float("nan")))
        except Exception:
            pass
    # Mean EVR values across all units from the per-unit table
    def _mean_col(col: str) -> float:
        if t5.empty or col not in t5.columns:
            return float("nan")
        sub = t5[t5["model"] == codec][col]
        return float(sub.mean()) if not sub.empty else float("nan")

    return {
        "mean_evr2_phase":        _mean_col("evr2_phase_pct"),
        "mean_evr1_amp":          _mean_col("evr1_amp_pct"),
        "mean_evr1_temporal":     _mean_col("evr1_temporal_pct"),
        "depth_evr2_slope_phase": slope_phase,
        "depth_evr2_slope_amp":   slope_amp,
    }


def build_cross_codec_summary(
    t1: pd.DataFrame,
    t2: pd.DataFrame,
    t3: pd.DataFrame,
    t5: pd.DataFrame,
    t6: pd.DataFrame,
) -> pd.DataFrame:
    """Build Table 7: 5-row cross-codec synthesis."""
    rows = []
    for codec in CODEC_META:
        meta = CODEC_META[codec]
        label = meta["unit_label"]
        first_unit = f"{label} 1"

        def _get_tfr(df: pd.DataFrame, condition_col: str, condition_val: str) -> float:
            if df.empty:
                return float("nan")
            mask = (df["model"] == codec) & (df["unit"] == first_unit) & (df[condition_col] == condition_val)
            sub = df[mask]
            if sub.empty:
                return float("nan")
            return float(sub["tfr_mean"].iloc[0])

        # Amplitude at 40 dBFS for first unit
        amp_tfr = _get_tfr(t1, "dbfs_attenuation", "40")
        # Phase at 90° for first unit
        phase_tfr = _get_tfr(t2, "phase_deg", "90")
        # Temporal at 10ms for first unit
        temporal_tfr = _get_tfr(t3, "offset_ms", "10")

        # SVD EVR metrics — mean across all units and depth slope
        svd_data = _load_svd_stats_for_codec(codec, t5)
        mean_evr2_phase  = svd_data["mean_evr2_phase"]
        mean_evr1_amp    = svd_data["mean_evr1_amp"]
        mean_evr1_temp   = svd_data["mean_evr1_temporal"]
        slope_phase      = svd_data["depth_evr2_slope_phase"]
        slope_amp        = svd_data["depth_evr2_slope_amp"]

        # EGFx mean TFR across all effect categories
        egfx_tfr = float("nan")
        if not t6.empty:
            sub = t6[t6["model"] == codec]
            if not sub.empty:
                egfx_tfr = float(sub["tfr_mean"].mean())

        rows.append({
            "model":                  codec,
            "amp_tfr_cb1_at_40dBFS":  round(amp_tfr, 6),
            "phase_tfr_cb1_at_90deg": round(phase_tfr, 6),
            "temporal_tfr_cb1_at_10ms": round(temporal_tfr, 6),
            "mean_evr2_phase_pct":    round(mean_evr2_phase, 2),
            "mean_evr1_amp_pct":      round(mean_evr1_amp, 2),
            "mean_evr1_temporal_pct": round(mean_evr1_temp, 2),
            "depth_evr2_slope_phase": round(slope_phase, 6),
            "depth_evr2_slope_amp":   round(slope_amp, 6),
            "egfx_tfr_mean":          round(egfx_tfr, 6),
        })

    return pd.DataFrame(rows)


def build_perplexity_table(tokens_root: Path) -> pd.DataFrame:
    """Build Table 8: perplexity by model, unit, test, and condition.

    Rows cover all units for each model across amplitude, phase, and temporal
    test conditions. Perplexity is averaged over all available signals/frequencies.
    """
    rows = []

    for codec, meta in CODEC_META.items():
        label = meta["unit_label"]
        n_units = meta["n_units"]
        vocab_size = meta.get("vocab_size", 1024)

        amp = _collect_perplexity_amplitude(tokens_root, meta, AMP_TAGS, vocab_size=vocab_size)
        for cond in ["0"] + AMP_TAGS:
            vals = amp.get(cond)
            for unit_idx in range(n_units):
                rows.append({
                    "model": codec,
                    "unit": f"{label} {unit_idx + 1}",
                    "test": "amplitude",
                    "condition": cond,
                    "condition_unit": "dbfs_attenuation",
                    "perplexity_mean": round(float(vals[unit_idx]), 6) if vals is not None and unit_idx < len(vals) else float("nan"),
                })

        phase = _collect_perplexity_phase(tokens_root, meta, PHASE_TAGS, vocab_size=vocab_size)
        for cond in ["0"] + PHASE_TAGS:
            vals = phase.get(cond)
            for unit_idx in range(n_units):
                rows.append({
                    "model": codec,
                    "unit": f"{label} {unit_idx + 1}",
                    "test": "phase",
                    "condition": cond,
                    "condition_unit": "phase_deg",
                    "perplexity_mean": round(float(vals[unit_idx]), 6) if vals is not None and unit_idx < len(vals) else float("nan"),
                })

        temporal = _collect_perplexity_temporal(tokens_root, meta, TEMPORAL_TAGS_MS, vocab_size=vocab_size)
        for cond in ["0"] + TEMPORAL_TAGS_MS:
            vals = temporal.get(cond)
            for unit_idx in range(n_units):
                rows.append({
                    "model": codec,
                    "unit": f"{label} {unit_idx + 1}",
                    "test": "temporal",
                    "condition": cond,
                    "condition_unit": "offset_ms",
                    "perplexity_mean": round(float(vals[unit_idx]), 6) if vals is not None and unit_idx < len(vals) else float("nan"),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens-root",
                    default=str(_DS / "audio_tokens"),
                    help="Root containing codec token subdirs")
    ap.add_argument("--svd-root",
                    default=str(_DS / "analysis" / "svd"),
                    help="Root containing per-codec SVD stats JSON dirs")
    ap.add_argument("--egfx-metrics",
                    default=str(_DS / "analysis" / "egfx_metrics.json"),
                    help="EGFx metrics JSON file")
    ap.add_argument("--output-dir",
                    default=str(_DS / "analysis" / "tables"),
                    help="Output directory for CSV tables")
    args = ap.parse_args()

    tokens_root  = Path(args.tokens_root)
    svd_root     = Path(args.svd_root)
    egfx_path    = Path(args.egfx_metrics)
    out_dir      = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables_written = 0

    # ── Table 1: Amplitude Response ──────────────────────────────────────────
    print("\n[Table 1] Amplitude response (TFR by dBFS attenuation) …")
    t1 = build_sinusoid_table(tokens_root, "dsp_self_amp", AMP_TAGS, "dbfs_attenuation")
    p1 = out_dir / "table1_amplitude_response.csv"
    t1.to_csv(p1, index=False)
    assert p1.stat().st_size > 100, f"Table 1 is suspiciously small: {p1}"
    print(f"  ✓ {p1}  ({len(t1)} rows)")
    tables_written += 1

    # ── Table 2: Phase Sensitivity ───────────────────────────────────────────
    print("\n[Table 2] Phase sensitivity (TFR by phase angle, 0° excluded) …")
    t2 = build_sinusoid_table(tokens_root, "dsp_self_phase", PHASE_TAGS, "phase_deg")
    p2 = out_dir / "table2_phase_sensitivity.csv"
    t2.to_csv(p2, index=False)
    assert p2.stat().st_size > 100
    print(f"  ✓ {p2}  ({len(t2)} rows)")
    tables_written += 1

    # ── Table 3: Temporal Offset ─────────────────────────────────────────────
    print("\n[Table 3] Temporal offset (TFR by offset ms) …")
    t3 = build_sinusoid_table(tokens_root, "time_sine", TEMPORAL_TAGS_MS, "offset_ms")
    p3 = out_dir / "table3_temporal_offset.csv"
    t3.to_csv(p3, index=False)
    assert p3.stat().st_size > 100
    print(f"  ✓ {p3}  ({len(t3)} rows)")
    tables_written += 1

    # ── Table 4: Centroid Magnitude ──────────────────────────────────────────
    # Requires pre-exported centroid stats JSON; skip gracefully if absent.
    print("\n[Table 4] Centroid magnitude …")
    centroid_rows = []
    centroid_root = svd_root.parent / "centroids"
    if centroid_root.exists():
        for codec in CODEC_META:
            cpath = centroid_root / f"{codec}_centroid_stats.json"
            if not cpath.exists():
                continue
            with cpath.open() as f:
                data = json.load(f)
            label = CODEC_META[codec]["unit_label"]
            # export_centroid_stats.py writes {"codec": ..., "codebooks": [...]}; support legacy flat list too
            entries = data.get("codebooks", []) if isinstance(data, dict) else data
            for i, entry in enumerate(entries):
                centroid_rows.append({
                    "model": codec,
                    "unit": f"{label} {i + 1}",
                    "mean_mag": round(float(entry.get("mean_mag", float("nan"))), 6),
                    "std_mag": round(float(entry.get("std_mag", float("nan"))), 6),
                    "skewness": round(float(entry.get("skewness", float("nan"))), 6),
                })
    if centroid_rows:
        t4 = pd.DataFrame(centroid_rows)
        p4 = out_dir / "table4_centroid_magnitude.csv"
        t4.to_csv(p4, index=False)
        print(f"  ✓ {p4}  ({len(t4)} rows)")
        tables_written += 1
    else:
        print("  [SKIP] No centroid stats found — run export_centroid_stats.py first")
        t4 = pd.DataFrame()

    # ── Table 5: SVD(ΔZ) EVR metrics ─────────────────────────────────────────
    print("\n[Table 5] SVD(ΔZ) EVR metrics (phase EVR₂, amp EVR₁, temporal EVR₁) …")
    t5 = build_svd_table(svd_root)
    if not t5.empty:
        p5 = out_dir / "table5_svd_evr2.csv"
        t5.to_csv(p5, index=False)
        assert p5.stat().st_size > 100
        print(f"  ✓ {p5}  ({len(t5)} rows)")
        tables_written += 1
    else:
        print("  [SKIP] No SVD stats JSON found — run stage_svd first")

    # ── Table 6: EGFx Response ───────────────────────────────────────────────
    print("\n[Table 6] EGFx non-linear effect response …")
    t6 = build_egfx_table(egfx_path)
    if not t6.empty:
        p6 = out_dir / "table6_egfx_response.csv"
        t6.to_csv(p6, index=False)
        assert p6.stat().st_size > 100
        print(f"  ✓ {p6}  ({len(t6)} rows)")
        tables_written += 1
    else:
        print("  [SKIP] EGFx metrics not found or empty")

    # ── Table 7: Cross-Codec Summary ─────────────────────────────────────────
    print("\n[Table 7] Cross-codec summary …")
    t7 = build_cross_codec_summary(t1, t2, t3, t5, t6)
    assert len(t7) == len(CODEC_META), f"Expected {len(CODEC_META)} rows in Table 7, got {len(t7)}"
    p7 = out_dir / "table7_cross_codec_summary.csv"
    t7.to_csv(p7, index=False)
    assert p7.stat().st_size > 100
    print(f"  ✓ {p7}  ({len(t7)} rows)")
    tables_written += 1

    # ── Table 8: Perplexity by Unit ─────────────────────────────────────────
    print("\n[Table 8] Perplexity by model/unit/condition …")
    t8 = build_perplexity_table(tokens_root)
    p8 = out_dir / "table8_perplexity_by_unit.csv"
    t8.to_csv(p8, index=False)
    assert p8.stat().st_size > 100
    print(f"  ✓ {p8}  ({len(t8)} rows)")
    tables_written += 1

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Tables written: {tables_written}")
    print(f"  Output dir    : {out_dir}")
    print(f"{'='*60}")
    csvs = sorted(out_dir.glob("table*.csv"))
    for p in csvs:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()

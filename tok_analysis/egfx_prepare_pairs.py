"""Parse EGFx metadata and create clean/processed effect pairs.

Expected EGFxSet layout after `egfx_download.py`:
  datasets/egfx/raw/
    Clean/
    BluesDriver/
    Chorus/
    ...
    egfxset_metadata.csv

The metadata CSV in this dataset does not include per-file `clean_file` /
`processed_file` columns. Pairing is therefore done by matching relative WAV
paths between each effect folder and the Clean folder.

Usage:
    python tok_analysis/egfx_prepare_pairs.py --egfx-dir datasets/egfx [--output effect_pairs.json]
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List


# Effect type categorization fallback
EFFECT_CATEGORIES = {
    "distortion": ["overdrive", "distortion", "fuzz", "saturation", "rat", "screamer", "driver"],
    "modulation": ["chorus", "flanger", "phaser", "tremolo", "vibrato"],
    "time_based": ["reverb", "delay", "echo"],
    "dynamics": ["compressor", "limiter", "gate"],
    "filter": ["equalizer", "wah", "filter", "lowpass", "highpass"],
}


# Known effect-folder aliases in the Zenodo archive
FOLDER_TO_EFFECT = {
    "bluesdriver": "blues driver",
    "tubescreamer": "tube screamer",
    "rat": "distortion",
    "chorus": "chorus",
    "flanger": "flanger",
    "phaser": "phaser",
    "tapeecho": "tape echo",
    "digitaldelay": "digital delay",
    "sweepecho": "sweep echo",
    "platereverb": "plate reverb",
    "hallreverb": "hall reverb",
    "springreverb": "spring reverb",
}


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def categorize_effect(effect_name: str, effect_type: str = "") -> str:
    """Categorize effect into predefined groups."""
    source = f"{effect_name} {effect_type}".lower()
    for category, keywords in EFFECT_CATEGORIES.items():
        if any(kw in source for kw in keywords):
            return category
    return "other"


def _find_metadata_file(raw_dir: Path) -> Path:
    direct = raw_dir / "egfxset_metadata.csv"
    if direct.exists():
        return direct
    candidates = list(raw_dir.rglob("*metadata*.csv"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No metadata CSV found in {raw_dir}")


def _load_metadata(metadata_file: Path) -> Dict[str, Dict[str, str]]:
    """Load metadata rows keyed by normalized effect name.

    Returns: {normalized_effect_name: {effect, model, effect_type, knob_names, knob_type, setting}}
    """
    rows_by_effect: Dict[str, Dict[str, str]] = {}
    with metadata_file.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Zenodo CSV has spaces in headers (e.g., " Effect ")
            clean_row = {k.strip().lower(): (v.strip() if isinstance(v, str) else v)
                         for k, v in row.items()}

            effect = clean_row.get("effect", "")
            if not effect:
                continue
            key = _normalize(effect)
            rows_by_effect[key] = {
                "effect": effect,
                "model": clean_row.get("model", ""),
                "effect_type": clean_row.get("effect type", ""),
                "knob_names": clean_row.get("knob names", ""),
                "knob_type": clean_row.get("knob type", ""),
                "setting": clean_row.get("setting", ""),
            }
    return rows_by_effect


def _find_clean_root(raw_dir: Path) -> Path:
    candidates = [d for d in raw_dir.iterdir() if d.is_dir() and d.name.lower() == "clean"]
    if not candidates:
        raise FileNotFoundError(
            f"Could not find Clean folder under {raw_dir}. "
            "Run egfx_download.py first."
        )
    return candidates[0]


def _effect_dirs(raw_dir: Path, clean_root: Path) -> list[Path]:
    dirs = []
    for d in raw_dir.iterdir():
        if not d.is_dir():
            continue
        if d == clean_root:
            continue
        if d.name.startswith("__"):
            continue
        # Keep only directories that contain wav files
        if any(d.rglob("*.wav")):
            dirs.append(d)
    return sorted(dirs)


def parse_egfx_metadata(egfx_dir: Path) -> Dict[str, List[Dict]]:
    """Build clean/processed pairs by relative-path matching.

    Returns:
        Dict mapping effect_category -> list of pair dicts.
    """
    raw_dir = egfx_dir / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Missing {raw_dir}. Run: python tok_analysis/egfx_download.py --output-dir {egfx_dir}"
        )

    metadata_file = _find_metadata_file(raw_dir)
    print(f"Using metadata file: {metadata_file}")
    metadata_by_effect = _load_metadata(metadata_file)

    clean_root = _find_clean_root(raw_dir)
    eff_dirs = _effect_dirs(raw_dir, clean_root)

    if not eff_dirs:
        raise FileNotFoundError(f"No effect directories with WAV files found in {raw_dir}")

    pairs_by_category: Dict[str, List[Dict]] = {}
    total_pairs = 0
    missing_clean_matches = 0

    for effect_dir in eff_dirs:
        folder_key = _normalize(effect_dir.name)
        effect_name = FOLDER_TO_EFFECT.get(folder_key, effect_dir.name)
        meta = metadata_by_effect.get(_normalize(effect_name), {})
        effect_type = meta.get("effect_type", "")
        category = categorize_effect(effect_name, effect_type)

        wavs = sorted(effect_dir.rglob("*.wav"))
        if not wavs:
            continue

        if category not in pairs_by_category:
            pairs_by_category[category] = []

        for proc_path in wavs:
            rel = proc_path.relative_to(effect_dir)
            clean_path = clean_root / rel
            if not clean_path.exists():
                missing_clean_matches += 1
                continue

            pairs_by_category[category].append({
                "clean": str(clean_path),
                "processed": str(proc_path),
                "effect": meta.get("effect", effect_name),
                "effect_dir": effect_dir.name,
                "effect_type": effect_type,
                "model": meta.get("model", ""),
                "params": {
                    "knob_names": meta.get("knob_names", ""),
                    "knob_type": meta.get("knob_type", ""),
                    "setting": meta.get("setting", ""),
                },
            })
            total_pairs += 1

    print(f"\nBuilt {total_pairs} clean/processed pairs")
    if missing_clean_matches > 0:
        print(f"[WARN] Skipped {missing_clean_matches} processed files without clean-match")

    return pairs_by_category


def main():
    parser = argparse.ArgumentParser(description="Prepare EGFx clean/processed pairs")
    parser.add_argument("--egfx-dir", type=Path, required=True, help="EGFx dataset directory")
    parser.add_argument("--output", type=Path, default=Path("datasets/egfx/effect_pairs.json"),
                        help="Output JSON file for effect pairs")
    args = parser.parse_args()
    
    pairs = parse_egfx_metadata(args.egfx_dir)
    
    # Summary
    print("\nEffect pairs by category:")
    for category, items in pairs.items():
        print(f"  {category}: {len(items)} pairs")
    
    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(pairs, f, indent=2)
    
    print(f"\n✓ Saved effect pairs to {args.output}")
    print(f"\nNext step: python tok_analysis/egfx_encode.py --pairs {args.output}")


if __name__ == "__main__":
    main()

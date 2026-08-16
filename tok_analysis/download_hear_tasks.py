#!/usr/bin/env python3
"""Download preprocessed HEAR 2021 benchmark tasks from Zenodo.

This is the RECOMMENDED approach for standardized, reproducible dataset acquisition.
It downloads the official HEAR preprocessing outputs which ensure consistent:
- Sample rates (48kHz default, other rates available)
- File formats and directory structures
- Train/validation/test splits
- Metadata and labels

For more information:
- HEAR Benchmark: https://hearbenchmark.com/
- Preprocessed tasks: https://doi.org/10.5281/zenodo.5885750
- Source preprocessing: https://github.com/hearbenchmark/hear-preprocess

Usage:
    # Download all tasks at 48kHz (default)
    python download_hear_tasks.py --output-dir datasets/hear_tasks

    # Download specific tasks only
    python download_hear_tasks.py --tasks speech_commands nsynth --output-dir datasets/hear_tasks

    # Download at different sample rate from Google Storage
    python download_hear_tasks.py --sample-rate 16000 --output-dir datasets/hear_tasks_16k
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.request
from pathlib import Path


# Official HEAR 2021 preprocessed tasks on Zenodo (48kHz)
# Use 2021.3 because it includes vocal_imitation and updated TFDS datasets.
ZENODO_RECORD = "6332517"  # pinned record: always the same archive regardless of future dataset updates
ZENODO_BASE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"

# Available tasks (subset of 19 total HEAR tasks)
AVAILABLE_TASKS = [
    "speech_commands",
    "nsynth_pitch",
    "esc50",
    "libricount",
    "vocal_imitation",
    "crema_d",
    "beijing_opera",
    "mridangam_tonic",
    "mridangam_stroke",
    "dcase2016_task2",
]

# For other sample rates, use Google Storage (requester pays)
GCS_BASE = "gs://hear2021-archive/tasks"
SUPPORTED_SAMPLE_RATES = [16000, 22050, 32000, 44100, 48000]

# Exact archive names in Zenodo record 6332517 (HEAR 2021.3)
TASK_ARCHIVES_48000 = {
    "speech_commands": "hear2021-speech_commands-v0.0.2-full-48000.tar.gz",
    "nsynth_pitch": "hear2021-nsynth_pitch-v2.2.3-50h-48000.tar.gz",
    "esc50": "hear2021-esc50-v2.0.0-full-48000.tar.gz",
    "libricount": "hear2021-libricount-v1.0.0-hear2021-full-48000.tar.gz",
    "vocal_imitation": "hear2021-vocal_imitation-v1.1.3-full-48000.tar.gz",
    "crema_d": "hear2021-tfds_crema_d-1.0.0-full-48000.tar.gz",
    "beijing_opera": "hear2021-beijing_opera-v1.0-hear2021-full-48000.tar.gz",
    "mridangam_tonic": "hear2021-mridangam_tonic-v1.5-full-48000.tar.gz",
    "mridangam_stroke": "hear2021-mridangam_stroke-v1.5-full-48000.tar.gz",
    "dcase2016_task2": "hear2021-dcase2016_task2-hear2021-full-48000.tar.gz",
}


def download_task_from_zenodo(task_name: str, output_dir: Path) -> bool:
    """Download a single task from Zenodo (48kHz preprocessed)."""
    task_dir = output_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if list(task_dir.rglob("*.wav")) or list(task_dir.rglob("*.npy")):
        print(f"  [SKIP] {task_name} already exists in {task_dir}")  # idempotent: skips re-download
        return True

    task_archive = TASK_ARCHIVES_48000.get(task_name)
    if not task_archive:
        print(f"  [FAILED] {task_name}: no archive mapping in TASK_ARCHIVES_48000")
        return False

    url = f"{ZENODO_BASE_URL}/{task_archive}/content"

    print(f"  [DOWNLOAD] {task_name} from Zenodo...")
    try:
        tmp_file = output_dir / f"_{task_name}_tmp.tar.gz"
        urllib.request.urlretrieve(url, tmp_file)

        # Extract to temporary directory
        import tarfile
        tmp_extract = output_dir / f"_extract_{task_name}"
        tmp_extract.mkdir(exist_ok=True)
        
        with tarfile.open(tmp_file, "r:gz") as tf:
            tf.extractall(tmp_extract)

        # Find the actual task folder (may be nested in tasks/ with version suffix)
        # Pattern: tasks/esc50-v2.0.0-full/48000/ -> datasets/hear_tasks/esc50/48000/
        task_candidates = list(tmp_extract.rglob(f"*{task_name}*"))
        
        # Prefer direct match, fall back to versioned folder
        extracted_task = None
        for candidate in task_candidates:
            if candidate.is_dir() and task_name in candidate.name.lower():
                extracted_task = candidate
                break
        
        if not extracted_task:
            # Try finding in tasks/ subdirectory
            tasks_dir = tmp_extract / "tasks"
            if tasks_dir.exists():
                task_candidates = [d for d in tasks_dir.iterdir() if d.is_dir() and task_name in d.name.lower()]
                if task_candidates:
                    extracted_task = task_candidates[0]
        
        if extracted_task:
            # Move to final location
            if task_dir.exists():
                shutil.rmtree(task_dir)
            shutil.move(str(extracted_task), str(task_dir))
        
        # Cleanup
        shutil.rmtree(tmp_extract)
        tmp_file.unlink()
        
        print(f"  [OK] {task_name} → {task_dir}")
        return True

    except Exception as exc:
        print(f"  [FAILED] {task_name}: {exc}")
        return False


def download_task_from_gcs(task_name: str, sample_rate: int, output_dir: Path) -> bool:
    """Download a single task from Google Cloud Storage at specific sample rate.

    Requires gsutil and requester pays configuration.
    """
    task_dir = output_dir / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    if list(task_dir.rglob("*.wav")) or list(task_dir.rglob("*.npy")):
        print(f"  [SKIP] {task_name} already exists in {task_dir}")
        return True

    gcs_path = f"{GCS_BASE}/{sample_rate}/{task_name}/"
    print(f"  [DOWNLOAD] {task_name} at {sample_rate}Hz from GCS...")

    try:
        # Use gsutil with requester pays
        cmd = [
            "gsutil",
            "-u", "YOUR_PROJECT_ID",  # User must set their GCP project
            "cp", "-r",
            gcs_path,
            str(task_dir)
        ]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"  [OK] {task_name} → {task_dir}")
        return True

    except FileNotFoundError:
        print(f"  [FAILED] {task_name}: gsutil not found. Install Google Cloud SDK.")
        print("         See: https://cloud.google.com/storage/docs/gsutil_install")
        return False
    except subprocess.CalledProcessError as exc:
        print(f"  [FAILED] {task_name}: {exc.stderr}")
        print("         Note: GCS downloads require requester pays setup.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download HEAR 2021 preprocessed benchmark tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all tasks at 48kHz from Zenodo
  python download_hear_tasks.py --output-dir datasets/hear_tasks

  # Download specific tasks only
  python download_hear_tasks.py --tasks speech_commands esc50 --output-dir datasets/hear_tasks

  # Download at 16kHz from Google Storage (requires gsutil + GCP project)
  python download_hear_tasks.py --sample-rate 16000 --output-dir datasets/hear_16k

For full HEAR evaluation pipeline, see:
  pip install heareval
  https://github.com/hearbenchmark/hear-eval-kit
        """
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/hear_tasks"),
        help="Root directory for downloaded tasks (default: datasets/hear_tasks)",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=AVAILABLE_TASKS + ["all"],
        default=["all"],
        help="Tasks to download (default: all)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        choices=SUPPORTED_SAMPLE_RATES,
        help="Sample rate (default: 48000 from Zenodo; others require GCS)",
    )
    parser.add_argument(
        "--gcp-project",
        type=str,
        help="GCP project ID for requester pays (required for non-48kHz rates)",
    )

    args = parser.parse_args()

    # Determine task list
    if "all" in args.tasks:
        tasks = AVAILABLE_TASKS
    else:
        tasks = args.tasks

    print(f"Downloading {len(tasks)} HEAR tasks to {args.output_dir}")
    print(f"Sample rate: {args.sample_rate}Hz")
    print()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    success_count = 0

    # Choose download method
    if args.sample_rate == 48000:
        # Use Zenodo (free, no auth required)
        for task in tasks:
            if download_task_from_zenodo(task, args.output_dir):
                success_count += 1
    else:
        # Use Google Cloud Storage (requires gsutil + project)
        if not args.gcp_project:
            print("[ERROR] --gcp-project required for non-48kHz sample rates")
            print("        GCS downloads require requester pays billing.")
            return 1

        for task in tasks:
            if download_task_from_gcs(task, args.sample_rate, args.output_dir):
                success_count += 1

    print()
    print(f"Download complete: {success_count}/{len(tasks)} tasks successful")
    print(f"Output directory: {args.output_dir}")

    if success_count < len(tasks):
        print()
        print("Some downloads failed. Common issues:")
        print("  - Network connectivity")
        print("  - Zenodo record format changes (update URLs in script)")
        print("  - GCS: missing gsutil or project billing setup")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())

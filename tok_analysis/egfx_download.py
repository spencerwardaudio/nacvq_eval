"""Download and extract EGFx dataset from Zenodo.

Zenodo no longer serves a single "EGFx.zip" file for record 7044411.
Instead, the record provides multiple zip files (Clean.zip + effect zips)
and a metadata CSV. This script downloads the record archive and extracts all
nested zip files into datasets/egfx/raw.

Usage:
    python tok_analysis/egfx_download.py [--output-dir datasets/egfx]
"""

import argparse
import json
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


RECORD_ID = "7044411"
ARCHIVE_URL = f"https://zenodo.org/api/records/{RECORD_ID}/files-archive"
RECORD_URL = f"https://zenodo.org/api/records/{RECORD_ID}"


def _is_valid_zip(path: Path) -> bool:
    """Return True if path is a readable zip archive."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad = zf.testzip()
            return bad is None
    except zipfile.BadZipFile:
        return False


def _download_archive(archive_path: Path) -> None:
    """Download Zenodo record files archive."""
    print(f"Downloading EGFx record archive from {ARCHIVE_URL}...")
    urllib.request.urlretrieve(ARCHIVE_URL, archive_path)
    if not _is_valid_zip(archive_path):
        archive_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Downloaded archive is not a valid zip. "
            "Please check network/proxy settings and retry."
        )
    print(f"✓ Downloaded archive: {archive_path}")


def _fetch_record_files() -> list[dict]:
    """Fetch Zenodo record metadata and return file entries."""
    print(f"Fetching record metadata from {RECORD_URL}...")
    with urllib.request.urlopen(RECORD_URL) as response:
        payload = json.loads(response.read().decode("utf-8"))

    files = payload.get("files", [])
    if not files:
        raise RuntimeError("No files found in Zenodo record metadata")

    print(f"✓ Found {len(files)} files in record")
    return files


def _download_record_files(extract_dir: Path) -> None:
    """Download all files listed in the Zenodo record into extract_dir."""
    files = _fetch_record_files()
    print("Downloading individual files from record API...")
    for entry in files:
        key = entry.get("key")
        links = entry.get("links", {})
        url = links.get("self")
        if not key or not url:
            print(f"[WARN] Skipping malformed file entry: {entry}")
            continue

        target = extract_dir / key
        if key.lower().endswith(".zip"):
            if _is_valid_zip(target):
                print(f"  ✓ Reusing existing zip: {key}")
                continue
            target.unlink(missing_ok=True)
        elif target.exists() and target.stat().st_size > 0:
            print(f"  ✓ Reusing existing file: {key}")
            continue

        print(f"  Downloading {key}...")
        urllib.request.urlretrieve(url, target)
    print("✓ Individual file downloads complete")


def _extract_record_archive(archive_path: Path, extract_dir: Path) -> None:
    """Extract top-level record archive into extract_dir."""
    print(f"Extracting record archive to {extract_dir}...")
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_dir)
    print("✓ Top-level files extracted")


def _extract_nested_archives(extract_dir: Path) -> None:
    """Extract each effect zip (and Clean.zip) into extract_dir."""
    nested_zips = sorted(extract_dir.glob("*.zip"))
    if not nested_zips:
        print("[WARN] No nested zip files found after top-level extraction")
        return

    print(f"Extracting {len(nested_zips)} nested zip files...")
    for zpath in nested_zips:
        if not _is_valid_zip(zpath):
            raise RuntimeError(f"Invalid nested zip: {zpath}")
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(extract_dir)
        print(f"  ✓ {zpath.name}")


def download_egfx(output_dir: Path) -> Path:
    """Download and extract EGFx dataset from Zenodo record archive."""
    output_dir.mkdir(parents=True, exist_ok=True)
    extract_dir = output_dir / "raw"
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Cleanup stale legacy file from old script behavior (often HTML 404 body)
    legacy_zip = output_dir / "EGFx.zip"
    if legacy_zip.exists() and not _is_valid_zip(legacy_zip):
        legacy_zip.unlink()
        print(f"Removed stale invalid legacy file: {legacy_zip}")

    # Fast-path: already prepared
    metadata_csv = extract_dir / "egfxset_metadata.csv"
    has_audio_dirs = any(p.is_dir() for p in extract_dir.iterdir())
    if metadata_csv.exists() and has_audio_dirs:
        print(f"✓ EGFx already prepared at {extract_dir}")
        return extract_dir

    archive_path = output_dir / f"record_{RECORD_ID}_files.zip"
    used_archive = False
    if _is_valid_zip(archive_path):
        print(f"✓ Archive already exists: {archive_path}")
        used_archive = True
    else:
        archive_path.unlink(missing_ok=True)
        try:
            _download_archive(archive_path)
            used_archive = True
        except urllib.error.HTTPError as exc:
            # Zenodo occasionally returns 400 for files-archive; fall back to per-file API.
            print(
                f"[WARN] files-archive download failed ({exc.code} {exc.reason}). "
                "Falling back to individual file downloads."
            )
        except RuntimeError as exc:
            print(f"[WARN] {exc}")
            print("[WARN] Falling back to individual file downloads.")

    if used_archive:
        _extract_record_archive(archive_path, extract_dir)
    else:
        _download_record_files(extract_dir)

    _extract_nested_archives(extract_dir)

    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Expected metadata CSV missing after extraction: {metadata_csv}"
        )

    print(f"✓ EGFx dataset ready at {extract_dir}")
    return extract_dir


def main():
    parser = argparse.ArgumentParser(description="Download EGFx dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/egfx"),
                        help="Output directory for EGFx dataset")
    args = parser.parse_args()

    extract_dir = download_egfx(args.output_dir)
    print(f"\n✓ EGFx dataset ready at {extract_dir}")
    print(f"\nNext step: python tok_analysis/egfx_prepare_pairs.py --egfx-dir {args.output_dir}")


if __name__ == "__main__":
    main()

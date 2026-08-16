#!/usr/bin/env python3
"""Validate FSD50K dataset paths for all 5 audio codecs.

This script validates that all required dataset filelists exist and point to
valid audio files for all 5 codecs: DAC-FSQ, Q2D2, Encodec, SpeechTokenizer, HiFiCodec.

Usage:
    # Validate current state
    python validate_dataset_paths.py
    
    # Auto-create missing codec-specific filelists from master CSVs
    python validate_dataset_paths.py --fix
    
    # Validate and exit with proper code for CI/CD
    python validate_dataset_paths.py && echo "Ready for training!"

Exit codes:
    0: All validations passed
    1: Validation failed (missing files or invalid paths)
"""

import argparse
import shutil
import sys
from pathlib import Path

# Add project root to path
_PROJ_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJ_ROOT))

from dataloader_aug.dataset_paths import get_dataset_config


def create_codec_filelists(config, verbose: bool = True):
    """Create codec-specific filelists from master CSVs.
    
    Args:
        config: DatasetConfig instance
        verbose: If True, print progress messages
    """
    if verbose:
        print("\n" + "="*80)
        print("🔧 CREATING CODEC-SPECIFIC FILELISTS")
        print("="*80)
    
    created_count = 0
    
    # Q2D2 (CODEC 2)
    if verbose:
        print("\n📋 CODEC 2: Q2D2")
    
    q2d2_data = config.proj_root / "Q2D2" / "data"
    q2d2_data.mkdir(parents=True, exist_ok=True)
    
    if config.train_csv.exists():
        shutil.copy(config.train_csv, config.q2d2_train)
        created_count += 1
        if verbose:
            print(f"   ✅ Created {config.q2d2_train.relative_to(config.proj_root)}")
    else:
        if verbose:
            print(f"   ⚠️  Cannot create - master CSV missing: {config.train_csv}")
    
    if config.val_csv.exists():
        shutil.copy(config.val_csv, config.q2d2_val)
        created_count += 1
        if verbose:
            print(f"   ✅ Created {config.q2d2_val.relative_to(config.proj_root)}")
    else:
        if verbose:
            print(f"   ⚠️  Cannot create - master CSV missing: {config.val_csv}")
    
    # SpeechTokenizer (CODEC 4)
    if verbose:
        print("\n📋 CODEC 4: SpeechTokenizer")
    
    data_dir = config.proj_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if config.train_csv.exists():
        shutil.copy(config.train_csv, config.speechtok_train)
        created_count += 1
        if verbose:
            print(f"   ✅ Created {config.speechtok_train.relative_to(config.proj_root)}")
    else:
        if verbose:
            print(f"   ⚠️  Cannot create - master CSV missing: {config.train_csv}")
    
    if config.val_csv.exists():
        shutil.copy(config.val_csv, config.speechtok_val)
        created_count += 1
        if verbose:
            print(f"   ✅ Created {config.speechtok_val.relative_to(config.proj_root)}")
    else:
        if verbose:
            print(f"   ⚠️  Cannot create - master CSV missing: {config.val_csv}")
    
    # HiFiCodec (CODEC 5)
    if verbose:
        print("\n📋 CODEC 5: HiFiCodec")
    
    hifi_data = config.proj_root / "hificodec" / "egs" / "data"
    hifi_data.mkdir(parents=True, exist_ok=True)
    
    if config.train_csv.exists():
        shutil.copy(config.train_csv, config.hificodec_train)
        created_count += 1
        if verbose:
            print(f"   ✅ Created {config.hificodec_train.relative_to(config.proj_root)}")
    else:
        if verbose:
            print(f"   ⚠️  Cannot create - master CSV missing: {config.train_csv}")
    
    if config.val_csv.exists():
        shutil.copy(config.val_csv, config.hificodec_val)
        created_count += 1
        if verbose:
            print(f"   ✅ Created {config.hificodec_val.relative_to(config.proj_root)}")
    else:
        if verbose:
            print(f"   ⚠️  Cannot create - master CSV missing: {config.val_csv}")
    
    if verbose:
        print("\n" + "="*80)
        print(f"✅ Created {created_count} codec-specific filelists")
        print("="*80 + "\n")
    
    return created_count


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, 
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--fix", 
        action="store_true",
        help="Auto-create missing codec-specific filelists from master CSVs"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output (only show errors)"
    )
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Get configuration and run initial validation
    config = get_dataset_config()  # resolves paths relative to the project root detected at import time
    is_valid, errors = config.validate(verbose=verbose)
    
    # If --fix flag provided and validation failed, try to fix
    if args.fix and not is_valid:
        created = create_codec_filelists(config, verbose=verbose)
        
        if created > 0:
            if verbose:
                print("🔄 Re-validating after auto-fix...\n")  # confirm auto-created files actually pass validation
            is_valid, errors = config.validate(verbose=verbose)
    
    # Final summary
    if is_valid:
        if verbose:
            print("\n" + "🎉"*40)
            print("✅ DATASET VALIDATION PASSED - READY FOR TRAINING!")
            print("🎉"*40 + "\n")
            print("All 5 codecs have valid dataset paths:")
            print("  1. DAC-FSQ      ✅")
            print("  2. Q2D2         ✅")
            print("  3. Encodec      ✅")
            print("  4. SpeechTokenizer ✅")
            print("  5. HiFiCodec    ✅")
            print()
        return 0
    else:
        if verbose:
            print("\n" + "❌"*40)
            print(f"❌ DATASET VALIDATION FAILED - {len(errors)} ERRORS")
            print("❌"*40 + "\n")
            print("💡 Troubleshooting:")
            print("  1. Check that master CSVs exist:")
            print(f"     {config.train_csv}")
            print(f"     {config.val_csv}")
            print("  2. Verify audio files exist at paths in CSVs")
            print("  3. Try running with --fix to auto-create missing filelists:")
            print(f"     python {Path(__file__).name} --fix")
            print()
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Unified dataset path validation and configuration for all 5 audio codecs.

This module provides a single source of truth for FSD50K dataset paths
with early validation to catch path issues before training starts.

Codecs covered:
1. DAC-FSQ (descript-audio-codec)
2. Q2D2
3. Encodec
4. SpeechTokenizer
5. HiFiCodec
"""

import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# Project root detection
_PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ_ROOT))


class DatasetConfig:
    """Centralized FSD50K dataset configuration with validation for all 5 codecs."""
    
    # Expected minimum file counts (for validation)
    MIN_TRAIN_FILES = 30000  # FSD50K dev set has ~36k files; this catches partially-downloaded datasets
    MIN_VAL_FILES = 3000     # FSD50K evaluation set has ~10k; 3k is a conservative lower bound
    
    def __init__(self, project_root: Optional[Path] = None):
        self.proj_root = project_root or _PROJ_ROOT
        
        # Assertions on initialization
        assert self.proj_root.exists(), f"Project root does not exist: {self.proj_root}"
        assert self.proj_root.is_dir(), f"Project root is not a directory: {self.proj_root}"
        
        # Expected dataset structure
        self.datasets_dir = self.proj_root / "datasets"
        self.fsd50k_dev_audio = self.datasets_dir / "FSD50K.dev_audio"
        
        # CSV filelists (single source of truth) - used by DAC-FSQ and Encodec
        self.train_csv = self.datasets_dir / "fsd50k_train.csv"
        self.val_csv = self.datasets_dir / "fsd50k_val.csv"
        self.test_csv = self.datasets_dir / "fsd50k_test.csv"
        
        # Codec-specific copied filelists (for compatibility)
        # Q2D2
        self.q2d2_train = self.proj_root / "Q2D2" / "data" / "fsd50k_train_files.txt"
        self.q2d2_val = self.proj_root / "Q2D2" / "data" / "fsd50k_val_files.txt"
        
        # SpeechTokenizer
        self.speechtok_train = self.proj_root / "data" / "fsd50k_train_files.txt"
        self.speechtok_val = self.proj_root / "data" / "fsd50k_val_files.txt"
        
        # HiFiCodec
        self.hificodec_train = self.proj_root / "hificodec" / "egs" / "data" / "fsd50k_train.lst"
        self.hificodec_val = self.proj_root / "hificodec" / "egs" / "data" / "fsd50k_val.lst"
        
    def _validate_filelist(self, path: Path, name: str, min_files: int = 100) -> Tuple[bool, Optional[str], int]:
        """Validate a single filelist file.
        
        Returns:
            (is_valid, error_message, file_count)
        """
        if not path.exists():
            return False, f"File not found: {path}", 0
        
        try:
            with open(path) as f:
                lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
            
            if len(lines) == 0:
                return False, f"Empty filelist: {path}", 0
            
            if len(lines) < min_files:
                return False, f"Too few entries ({len(lines)} < {min_files}): {path}", len(lines)
            
            # Check first path
            first_path = Path(lines[0])  # verify the filelist isn’t pointing to a stale or moved location
            if not first_path.exists():
                return False, f"First path does not exist: {first_path} (from {path})", len(lines)
            
            # Assertion: first file should be a WAV file
            assert first_path.suffix.lower() in ['.wav', '.flac'], \
                f"Expected audio file, got {first_path.suffix}: {first_path}"  # catches CSVs that contain metadata rows rather than audio paths
            
            return True, None, len(lines)
            
        except Exception as e:
            return False, f"Error reading {path}: {e}", 0
    
    def validate(self, verbose: bool = True) -> Tuple[bool, List[str]]:
        """Validate dataset paths for all 5 codecs and return (is_valid, errors).
        
        Performs comprehensive validation with assertions to ensure data integrity.
        """
        errors = []
        warnings = []
        
        if verbose:
            print("\n" + "="*80)
            print("🔍 DATASET PATH VALIDATION FOR ALL 5 CODECS")
            print("="*80)
            print(f"Project root: {self.proj_root}")
        
        # 1. Check project structure
        if verbose:
            print(f"\n📁 Datasets directory: {self.datasets_dir}")
        
        if not self.datasets_dir.exists():
            errors.append(f"Datasets directory not found: {self.datasets_dir}")
            if verbose:
                print(f"   ❌ NOT FOUND")
        else:
            if verbose:
                print(f"   ✅ EXISTS")
        
        # 2. Check audio directories
        # dev_audio is REQUIRED (contains train/val/test split)
        if verbose:
            print(f"\n📂 Audio: FSD50K.dev_audio (REQUIRED)")
            print(f"   Path: {self.fsd50k_dev_audio}")
        
        if not self.fsd50k_dev_audio.exists():
            errors.append(f"Audio directory not found: {self.fsd50k_dev_audio}")
            if verbose:
                print(f"   ❌ NOT FOUND")
        else:
            # Count audio files
            wav_files = list(self.fsd50k_dev_audio.glob("*.wav"))
            flac_files = list(self.fsd50k_dev_audio.glob("*.flac"))
            total_files = len(wav_files) + len(flac_files)  # FSD50K ships wav; some releases also contain flac
            
            if verbose:
                print(f"   ✅ EXISTS")
                print(f"   📊 Files: {len(wav_files)} WAV, {len(flac_files)} FLAC (total: {total_files})")
            
            # Assertion: should have some audio files
            if total_files == 0:
                errors.append(f"No audio files found in {self.fsd50k_dev_audio}")
        
        # 3. Validate master CSV files (DAC-FSQ, Encodec)
        if verbose:
            print("\n" + "="*80)
            print("📄 CODEC 1 & 3: DAC-FSQ & ENCODEC (Master CSV Filelists)")
            print("="*80)
        
        csv_files = [
            ("train", self.train_csv, self.MIN_TRAIN_FILES),
            ("val", self.val_csv, self.MIN_VAL_FILES),
            ("test", self.test_csv, 1000),
        ]
        
        for name, path, min_count in csv_files:
            is_valid, error, count = self._validate_filelist(path, name, min_count)
            
            if verbose:
                print(f"\n   {name}: {path.relative_to(self.proj_root)}")
            
            if not is_valid:
                errors.append(error)
                if verbose:
                    print(f"      ❌ {error}")
            else:
                if verbose:
                    print(f"      ✅ {count} entries")
                    # Read first path for display
                    with open(path) as f:
                        first_line = next((l.strip() for l in f if l.strip()), None)
                        if first_line:
                            print(f"      📝 Sample: {Path(first_line).name}")
        
        # 4. Validate Q2D2 filelists (CODEC 2)
        if verbose:
            print("\n" + "="*80)
            print("📄 CODEC 2: Q2D2 (Rhombic Quantization)")
            print("="*80)
        
        q2d2_files = [
            ("Q2D2 train", self.q2d2_train, self.MIN_TRAIN_FILES),
            ("Q2D2 val", self.q2d2_val, self.MIN_VAL_FILES),
        ]
        
        for name, path, min_count in q2d2_files:
            is_valid, error, count = self._validate_filelist(path, name, min_count)
            
            if verbose:
                print(f"\n   {path.relative_to(self.proj_root)}")
            
            if not is_valid:
                if path.exists():
                    errors.append(error)
                    if verbose:
                        print(f"      ❌ {error}")
                else:
                    warnings.append(f"Will be created: {path}")
                    if verbose:
                        print(f"      ⚠️  NOT FOUND (will be created from master CSV)")
            else:
                if verbose:
                    print(f"      ✅ {count} entries")
        
        # 5. Validate SpeechTokenizer filelists (CODEC 4)
        if verbose:
            print("\n" + "="*80)
            print("📄 CODEC 4: SPEECHTOKENIZER")
            print("="*80)
        
        spt_files = [
            ("SpeechTokenizer train", self.speechtok_train, self.MIN_TRAIN_FILES),
            ("SpeechTokenizer val", self.speechtok_val, self.MIN_VAL_FILES),
        ]
        
        for name, path, min_count in spt_files:
            is_valid, error, count = self._validate_filelist(path, name, min_count)
            
            if verbose:
                print(f"\n   {path.relative_to(self.proj_root)}")
            
            if not is_valid:
                if path.exists():
                    errors.append(error)
                    if verbose:
                        print(f"      ❌ {error}")
                else:
                    warnings.append(f"Will be created: {path}")
                    if verbose:
                        print(f"      ⚠️  NOT FOUND (will be created from master CSV)")
            else:
                if verbose:
                    print(f"      ✅ {count} entries")
        
        # 6. Validate HiFiCodec filelists (CODEC 5)
        if verbose:
            print("\n" + "="*80)
            print("📄 CODEC 5: HIFICODEC")
            print("="*80)
        
        hifi_files = [
            ("HiFiCodec train", self.hificodec_train, self.MIN_TRAIN_FILES),
            ("HiFiCodec val", self.hificodec_val, self.MIN_VAL_FILES),
        ]
        
        for name, path, min_count in hifi_files:
            is_valid, error, count = self._validate_filelist(path, name, min_count)
            
            if verbose:
                print(f"\n   {path.relative_to(self.proj_root)}")
            
            if not is_valid:
                if path.exists():
                    errors.append(error)
                    if verbose:
                        print(f"      ❌ {error}")
                else:
                    warnings.append(f"Will be created: {path}")
                    if verbose:
                        print(f"      ⚠️  NOT FOUND (will be created from master CSV)")
            else:
                if verbose:
                    print(f"      ✅ {count} entries")
        
        # Summary
        if verbose:
            print("\n" + "="*80)
            if errors:
                print(f"❌ VALIDATION FAILED ({len(errors)} errors, {len(warnings)} warnings)")
                print("\n🔴 ERRORS:")
                for i, err in enumerate(errors, 1):
                    print(f"   {i}. {err}")
                if warnings:
                    print("\n⚠️  WARNINGS:")
                    for i, warn in enumerate(warnings, 1):
                        print(f"   {i}. {warn}")
            else:
                print("✅ ALL 5 CODECS VALIDATED SUCCESSFULLY")
                if warnings:
                    print(f"\n⚠️  {len(warnings)} warnings (codec-specific filelists will be auto-created)")
            print("="*80 + "\n")
        
        return len(errors) == 0, errors
    
    def get_train_filelist(self, codec: str) -> Path:
        """Get training filelist path for specific codec.
        
        Args:
            codec: One of: dac_fsq, q2d2, encodec, speechtokenizer, hificodec
        
        Returns:
            Path to training filelist
        """
        codec_lower = codec.lower().replace("-", "_")
        
        mapping = {
            "dac_fsq": self.train_csv,
            "q2d2": self.q2d2_train,
            "encodec": self.train_csv,
            "speechtokenizer": self.speechtok_train,
            "hificodec": self.hificodec_train,
        }
        
        assert codec_lower in mapping, f"Unknown codec: {codec}. Must be one of {list(mapping.keys())}"
        return mapping[codec_lower]
    
    def get_val_filelist(self, codec: str) -> Path:
        """Get validation filelist path for specific codec.
        
        Args:
            codec: One of: dac_fsq, q2d2, encodec, speechtokenizer, hificodec
        
        Returns:
            Path to validation filelist
        """
        codec_lower = codec.lower().replace("-", "_")
        
        mapping = {
            "dac_fsq": self.val_csv,
            "q2d2": self.q2d2_val,
            "encodec": self.val_csv,
            "speechtokenizer": self.speechtok_val,
            "hificodec": self.hificodec_val,
        }
        
        assert codec_lower in mapping, f"Unknown codec: {codec}. Must be one of {list(mapping.keys())}"
        return mapping[codec_lower]


# Global instance
_dataset_config = None

def get_dataset_config() -> DatasetConfig:
    """Get or create global dataset configuration singleton."""
    global _dataset_config
    if _dataset_config is None:
        _dataset_config = DatasetConfig()
    return _dataset_config


def validate_audio_path(audio_path: str, source: str = "unknown") -> Path:
    """Validate and return audio file path with detailed error messages.
    
    Args:
        audio_path: Path to audio file (string or Path)
        source: Description of where this path came from (for debugging)
    
    Returns:
        Validated Path object
        
    Raises:
        FileNotFoundError: With detailed diagnostic information
        AssertionError: If path validation fails
    """
    path = Path(audio_path)
    
    # Assertion: path should be absolute or relative to a known location
    assert len(str(path)) > 0, f"Empty audio path from {source}"
    
    if not path.exists():
        # Detailed error message
        config = get_dataset_config()
        error_msg = [
            f"\n{'='*80}",
            f"❌ AUDIO FILE NOT FOUND",
            f"{'='*80}",
            f"Source: {source}",
            f"Path: {audio_path}",
            f"Absolute: {path.absolute()}",
            f"Parent exists: {path.parent.exists()}",
            f"CWD: {Path.cwd()}",
        ]
        
        # Check if it's a relative vs absolute path issue
        if not path.is_absolute():
            error_msg.append(f"⚠️  Path is RELATIVE - may depend on working directory")
            abs_path = path.absolute()
            error_msg.append(f"Resolved to: {abs_path}")
        
        # Suggest alternatives
        error_msg.extend([
            f"\n📂 Expected dataset structure:",
            f"   FSD50K audio: {config.fsd50k_dev_audio}",
            f"   Exists: {config.fsd50k_dev_audio.exists()}",
            f"\n💡 Run dataset validation:",
            f"   python validate_dataset_paths.py",
            "="*80
        ])
        
        raise FileNotFoundError("\n".join(error_msg))
    
    # Assertion: should be an audio file
    assert path.suffix.lower() in ['.wav', '.flac', '.mp3', '.ogg'], \
        f"Not an audio file ({path.suffix}): {path}"
    
    return path


if __name__ == "__main__":
    # CLI tool for quick validation
    config = get_dataset_config()
    is_valid, errors = config.validate(verbose=True)
    sys.exit(0 if is_valid else 1)

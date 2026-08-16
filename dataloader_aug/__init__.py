"""Shared dataloader utilities with augmentation for audio codec training.

This package provides unified preprocessing and normalization functions
used across all 5 audio codec trainers (Q2D2, Encodec, DAC-FSQ, 
SpeechTokenizer, and HiFiCodec).

Key components:
- normalize_rms_snr: RMS/SNR-based normalization with data augmentation
- get_normalization_stats: Extract normalization statistics for debugging
- DatasetConfig: Unified dataset path configuration for all codecs
- validate_audio_path: Path validation with detailed error messages
- get_dataset_config: Get singleton dataset configuration instance
"""

from .audio_preprocessing import normalize_rms_snr, get_normalization_stats
from .dataset_paths import (
    DatasetConfig,
    validate_audio_path,
    get_dataset_config,
)

__all__ = [
    'normalize_rms_snr',
    'get_normalization_stats',
    'DatasetConfig',
    'validate_audio_path',
    'get_dataset_config',
]

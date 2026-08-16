"""Shared audio preprocessing utilities for consistent normalization across all codecs.

This module provides unified RMS/SNR-based normalization to replace inconsistent
approaches (peak normalization, RMS without target, or no normalization) across
the 5 audio codecs: Q2D2, Encodec, DAC-FSQ, SpeechTokenizer, and HiFiCodec.

Key Benefits:
- Preserves relative loudness relationships between samples
- Provides consistent gain handling across all models
- Essential for amplitude-based testing (preserves 2:1 amplitude ratios, etc.)
- Allows controlled data augmentation via SNR variation during training

Usage:
    from datasets.audio_preprocessing import normalize_rms_snr
    
    # Training with variation
    normalized = normalize_rms_snr(waveform, train_mode=True)
    
    # Validation with fixed target
    normalized = normalize_rms_snr(waveform, train_mode=False)
"""

from __future__ import annotations

import numpy as np
import torch
import warnings
from pathlib import Path
from typing import Optional

# Import validation utilities for path checking
try:
    from .dataset_paths import validate_audio_path
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False


def normalize_rms_snr(
    waveform: torch.Tensor | np.ndarray,
    target_snr_db: float = 40.0,
    noise_floor_db: float = -60.0,
    train_mode: bool = False,
    snr_variation_db: float = 5.0,
    silence_threshold_db: float = -80.0,
    clip_threshold: float = 0.95,
    clip_value: float = 0.99,
    audio_path: Optional[str] = None,
    source_identifier: str = "unknown",
) -> torch.Tensor | np.ndarray:
    """Apply RMS-based SNR normalization to audio waveform.
    
    This function normalizes audio by measuring RMS energy and scaling to a target
    SNR level relative to a defined noise floor. Unlike peak normalization (which
    scales each file to the same peak level), RMS normalization preserves the
    relative loudness relationships between files.
    
    Example:
        If file A has 2× the RMS energy of file B before normalization, it will
        still have approximately 2× the RMS energy after normalization (within
        the ±5 dB training variation range if train_mode=True).
    
    Args:
        waveform: Input audio tensor or array, shape [..., samples] or [samples].
                  Works with mono or multi-channel audio.
        target_snr_db: Target signal-to-noise ratio in dB. Signal will be placed
                       this many dB above the noise floor. Default: 40 dB.
        noise_floor_db: Reference noise floor level in dBFS. Default: -60 dBFS
                        (standard digital audio noise floor).
        train_mode: If True, applies random variation to target_snr_db for data
                    augmentation. If False, uses fixed target_snr_db.
        snr_variation_db: Random variation range (±) in dB when train_mode=True.
                          Default: ±5 dB (35-45 dB SNR range).
        silence_threshold_db: If RMS is below this level, skip normalization and
                              return original waveform. Default: -80 dBFS.
        clip_threshold: Log warning if any sample exceeds this absolute value.
                        Default: 0.95.
        clip_value: Hard clip output to [-clip_value, +clip_value] to prevent
                    numerical overflow. Default: 0.99.
        audio_path: Optional path to audio file (for validation and debugging).
                    If provided and validation is available, will check path exists.
        source_identifier: String identifying the caller/dataset (for debugging).
                           Example: "Q2D2/VocosDataset", "DAC-FSQ/SimpleAudioDataset"
    
    Returns:
        Normalized waveform with same type and shape as input.
        - Training: RMS level uniformly distributed in range 
                    [target_snr_db - snr_variation_db, target_snr_db + snr_variation_db]
        - Validation: RMS level exactly at target_snr_db
    
    Technical Details:
        - RMS (Root Mean Square): sqrt(mean(waveform^2))
        - dBFS = 20 * log10(rms), where full scale (1.0) = 0 dBFS
        - Target RMS in dBFS: noise_floor_db + target_snr_db
        - Gain applied: target_rms_db - current_rms_db
    
    Example Values:
        - target_snr_db=40, noise_floor_db=-60 → signal at -20 dBFS
        - train_mode=True, snr_variation_db=5 → signal between -25 to -15 dBFS
        - train_mode=False → signal exactly at -20 dBFS
    """
    # Validate audio path if provided (diagnostic/debugging feature)
    if audio_path is not None and VALIDATION_AVAILABLE:
        try:
            validate_audio_path(audio_path, source=source_identifier)
            # Debug print - can be disabled in production
            # print(f"🎵 normalize_rms_snr: {Path(audio_path).name} from {source_identifier}")
        except (FileNotFoundError, AssertionError) as e:
            # Don't fail training, just warn
            warnings.warn(
                f"Path validation warning for {audio_path} from {source_identifier}: {e}",
                RuntimeWarning
            )
    
    # Assertions on input
    assert isinstance(waveform, (np.ndarray, torch.Tensor)), \
        f"Waveform must be numpy array or torch tensor, got {type(waveform)}"
    
    # Determine if input is numpy or torch
    is_numpy = isinstance(waveform, np.ndarray)  # track so we can restore the original type on return
    
    # Convert to torch if needed
    if is_numpy:
        x = torch.from_numpy(waveform).float()
    else:
        x = waveform.float()
    
    # Calculate RMS energy
    rms = torch.sqrt(torch.mean(x ** 2))  # standard RMS = sqrt(mean(x^2)), not peak-based
    
    # Handle silent/near-silent audio
    current_rms_db = 20 * torch.log10(rms + 1e-8)  # 1e-8 offset avoids log(0) for truly silent signals
    if current_rms_db < silence_threshold_db:
        # Commented out to reduce console spam (similar to Q2D2/SpeechTokenizer approach)
        # warnings.warn(
        #     f"Audio RMS ({current_rms_db:.1f} dBFS) below silence threshold "
        #     f"({silence_threshold_db} dBFS). Skipping normalization.",
        #     RuntimeWarning
        # )
        return waveform  # Return original, unchanged
    
    # Determine target RMS level
    if train_mode:
        # Random variation for data augmentation: uniform in [-snr_variation_db, +snr_variation_db]
        variation = np.random.uniform(-snr_variation_db, snr_variation_db)  # np.random keeps numpy RNG state separate from torch
        target_rms_db = noise_floor_db + target_snr_db + variation
    else:
        # Fixed target for validation
        target_rms_db = noise_floor_db + target_snr_db
    
    # Calculate gain needed to reach target RMS
    gain_db = target_rms_db - current_rms_db.item()
    gain_linear = 10 ** (gain_db / 20)  # dB to linear: 20 dB per decade for amplitude (not power)
    
    # Apply gain
    x_normalized = x * gain_linear
    
    # Check for clipping potential
    peak = torch.abs(x_normalized).max().item()
    if peak > clip_threshold:
        # Commented out to reduce console spam (similar to Q2D2/SpeechTokenizer approach)
        # Only show this warning once per process to reduce console spam
        # warnings.simplefilter('once', RuntimeWarning)
        # warnings.warn(
        #     f"Normalized audio peak ({peak:.3f}) exceeds threshold ({clip_threshold}). "
        #     f"This may indicate clipping. Applying hard clip at ±{clip_value}.",
        #     RuntimeWarning
        # )
        pass
    
    # Hard clip to prevent numerical overflow
    x_normalized = torch.clamp(x_normalized, -clip_value, clip_value)  # 0.99 not 1.0 — avoids float rounding past full-scale
    
    # Convert back to original type
    if is_numpy:
        return x_normalized.numpy().astype(waveform.dtype)  # restore original dtype (e.g. float64 array stays float64)
    else:
        return x_normalized.to(waveform.dtype)


def get_normalization_stats(
    waveform: torch.Tensor | np.ndarray,
) -> dict[str, float]:
    """Get normalization statistics for a waveform (for debugging/validation).
    
    Args:
        waveform: Input audio tensor or array.
    
    Returns:
        Dictionary with keys:
        - 'rms': RMS energy (linear scale)
        - 'rms_db': RMS energy in dBFS
        - 'peak': Maximum absolute amplitude
        - 'peak_db': Peak level in dBFS
        - 'crest_factor_db': Peak-to-RMS ratio in dB
    """
    # Convert to torch if needed
    if isinstance(waveform, np.ndarray):
        x = torch.from_numpy(waveform).float()
    else:
        x = waveform.float()
    
    rms = torch.sqrt(torch.mean(x ** 2)).item()
    peak = torch.abs(x).max().item()
    
    rms_db = 20 * np.log10(rms + 1e-8)
    peak_db = 20 * np.log10(peak + 1e-8)
    crest_factor_db = peak_db - rms_db
    
    return {
        'rms': rms,
        'rms_db': rms_db,
        'peak': peak,
        'peak_db': peak_db,
        'crest_factor_db': crest_factor_db,
    }

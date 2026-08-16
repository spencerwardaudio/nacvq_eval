"""Feature extraction from frozen codec encoders for semantic evaluation.

This module provides adapters for extracting clip-level features from Encodec and SemantiCodec
following the methodology described in the SemantiCodec paper (Table III).

Usage:
    # Extract features from Encodec
    extractor = EncodecFeatureExtractor(model_name='encodec_24khz', bandwidth=6.0, device='cuda')
    features = extractor.extract_clip_features('path/to/audio.wav')  # Shape: [feature_dim]
    
    # Extract features from SemantiCodec
    extractor = SemantiCodecFeatureExtractor(token_rate=50, semantic_vocab_size=32768, device='cuda')
    features = extractor.extract_clip_features('path/to/audio.wav')  # Shape: [feature_dim]
"""

import sys
from pathlib import Path
from typing import Optional, Literal
import warnings

import torch
import torchaudio

# Add Encodec directory to path
_ENCODEC_DIR = Path(__file__).parent.parent / "Encodec"
if str(_ENCODEC_DIR) not in sys.path:
    sys.path.insert(0, str(_ENCODEC_DIR))

from compress import MODELS
from model import EncodecModel


class FeatureExtractorBase:
    """Base class for codec feature extractors."""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
    
    def extract_clip_features(self, audio_path: str | Path) -> torch.Tensor:
        """Extract clip-level features from an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Clip-level feature vector of shape [feature_dim]
        """
        raise NotImplementedError("Subclasses must implement extract_clip_features")
    
    def _load_audio(self, audio_path: str | Path, target_sr: int) -> torch.Tensor:
        """Load and resample audio to target sample rate.
        
        Args:
            audio_path: Path to audio file
            target_sr: Target sample rate
            
        Returns:
            Audio tensor of shape [1, num_samples]
        """
        waveform, sr = torchaudio.load(str(audio_path))
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if needed
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        return waveform.to(self.device)
    
    @property
    def feature_dim(self) -> int:
        """Return the dimensionality of extracted features."""
        raise NotImplementedError("Subclasses must implement feature_dim")


class EncodecFeatureExtractor(FeatureExtractorBase):
    """Extract clip-level features from Encodec encoder.
    
    This extractor:
    1. Loads a pretrained Encodec model
    2. Passes audio through encoder to get continuous embeddings
    3. Quantizes embeddings using RVQ
    4. Reconstructs quantized embeddings (all codebook layers concatenated)
    5. Averages along time dimension to get clip-level features
    
    Args:
        model_name: Name of Encodec model ('encodec_24khz' or 'encodec_48khz')
        bandwidth: Target bandwidth in kbps (1.5, 3, 6, 12, 24)
        device: Device to run inference on ('cuda' or 'cpu')
        checkpoint_path: Optional path to custom checkpoint
    """
    
    def __init__(
        self,
        model_name: Literal['encodec_24khz', 'encodec_48khz'] = 'encodec_24khz',
        bandwidth: float = 6.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__(device)
        
        self.model_name = model_name
        self.bandwidth = bandwidth
        self.sample_rate = 24000 if '24khz' in model_name else 48000
        
        # Load model
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
        
        model_fn = MODELS[model_name]
        self.model: EncodecModel = model_fn()
        
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(device)
        self.model.eval()
        self.model.set_target_bandwidth(bandwidth)
        
        # Feature dimension is the encoder output dimension (128 for standard Encodec)
        self._feature_dim = self.model.encoder.dimension
    
    @property
    def feature_dim(self) -> int:
        """Encodec encoder dimension (typically 128)."""
        return self._feature_dim
    
    @torch.no_grad()
    def extract_clip_features(self, audio_path: str | Path) -> torch.Tensor:
        """Extract clip-level features from audio file.
        
        The extraction process follows the SemantiCodec paper methodology:
        1. Encoder forward pass: audio → continuous embeddings [B, 128, T]
        2. Vector quantization: embeddings → discrete codes [B, n_q, T]
        3. Dequantization: codes → quantized embeddings [B, 128, T]
        4. Temporal pooling: mean over time → clip-level features [128]
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Clip-level feature vector of shape [feature_dim]
        """
        # Load and preprocess audio
        waveform = self._load_audio(audio_path, self.sample_rate)
        
        # Add batch dimension if needed
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)  # [1, 1, T]
        
        # Extract encoder embeddings
        # The model.encode() method returns list of (codes, scale) tuples
        encoded_frames = self.model.encode(waveform)
        
        # For each frame, get the quantized embeddings
        quantized_embeddings = []
        for codes, scale in encoded_frames:
            # codes shape: [B, n_q, T] where n_q is number of codebooks
            # Convert to quantized embeddings
            codes_transposed = codes.transpose(0, 1)  # [n_q, B, T]
            quantized_emb = self.model.quantizer.decode(codes_transposed)  # [B, 128, T]
            quantized_embeddings.append(quantized_emb)
        
        # Concatenate all frames along time dimension
        if len(quantized_embeddings) > 1:
            all_embeddings = torch.cat(quantized_embeddings, dim=-1)  # [B, 128, T_total]
        else:
            all_embeddings = quantized_embeddings[0]
        
        # Temporal averaging to get clip-level features
        clip_features = all_embeddings.mean(dim=-1)  # [B, 128]
        
        # Remove batch dimension and return
        return clip_features.squeeze(0).cpu()  # [128]


class SemantiCodecFeatureExtractor(FeatureExtractorBase):
    """Extract clip-level features from SemantiCodec encoder.
    
    NOTE: This requires access to SemantiCodec's internal features, which are not
    exposed by the standard API. This implementation assumes you have either:
    1. Modified the semanticodec library to expose encoder features, OR
    2. Are using a forked version with a `encode_features()` method
    
    The extractor:
    1. Loads SemantiCodec model
    2. Extracts frozen AudioMAE semantic features + learnable BiLSTM acoustic features
    3. Averages along time dimension to get clip-level features
    
    Args:
        token_rate: Token rate in Hz (25, 50, or 100)
        semantic_vocab_size: Size of semantic codebook (typically 32768)
        device: Device to run inference on ('cuda' or 'cpu')
        extract_layer: Which features to extract ('all' for E=[Es, Ea], 'semantic' for Es only)
    """
    
    def __init__(
        self,
        token_rate: int = 50,
        semantic_vocab_size: int = 32768,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        extract_layer: Literal['all', 'semantic'] = 'all',
    ):
        super().__init__(device)
        
        self.token_rate = token_rate
        self.semantic_vocab_size = semantic_vocab_size
        self.extract_layer = extract_layer
        
        # Import SemantiCodec
        try:
            import semanticodec
            self.SemantiCodec = semanticodec.SemantiCodec
        except ImportError:
            raise ImportError(
                "SemantiCodec not installed. Install with:\n"
                "pip install git+https://github.com/haoheliu/SemantiCodec-inference.git"
            )
        
        # Load model (semantic_vocab_size is required by the library)
        self.model = self.SemantiCodec(
            semantic_vocab_size=semantic_vocab_size,
            token_rate=token_rate
        )
        
        # Move to device (if model supports it)
        if hasattr(self.model, 'to'):
            self.model = self.model.to(device)
        
        # Feature dimension depends on the internal architecture
        # AudioMAE typically produces 768-dim features
        # BiLSTM adds additional dimensions
        # This is a placeholder - adjust based on actual SemantiCodec implementation
        self._feature_dim = 768 if extract_layer == 'semantic' else 768 + 256
        
        warnings.warn(
            "SemantiCodecFeatureExtractor requires access to internal features. "
            "If the standard API doesn't expose them, you may need to:\n"
            "1. Fork and modify the semanticodec library, OR\n"
            "2. Use monkey-patching to intercept encoder outputs\n"
            "See documentation for details."
        )
    
    @property
    def feature_dim(self) -> int:
        """SemantiCodec feature dimension (768 for semantic only, higher for all layers)."""
        return self._feature_dim
    
    @torch.no_grad()
    def extract_clip_features(self, audio_path: str | Path) -> torch.Tensor:
        """Extract clip-level features from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Clip-level feature vector of shape [feature_dim]
            
        Raises:
            NotImplementedError: If SemantiCodec doesn't expose internal features
        """
        # Load audio (SemantiCodec typically uses 16kHz)
        waveform = self._load_audio(audio_path, target_sr=16000)
        
        # Try to extract features using different approaches
        
        # Approach 1: Check if model has encode_features method (modified library)
        if hasattr(self.model, 'encode_features'):
            features = self.model.encode_features(
                waveform,
                layer=self.extract_layer
            )  # Expected shape: [B, T, D]
            
            # Temporal averaging
            clip_features = features.mean(dim=1)  # [B, D]
            return clip_features.squeeze(0).cpu()
        
        # Approach 2: Try to access encoder directly
        elif hasattr(self.model, 'semantic_encoder'):
            # Extract semantic features from frozen AudioMAE
            semantic_features = self.model.semantic_encoder(waveform)  # [B, T, 768]
            
            if self.extract_layer == 'all' and hasattr(self.model, 'acoustic_encoder'):
                # Extract acoustic features from BiLSTM
                acoustic_features = self.model.acoustic_encoder(waveform)  # [B, T, D_a]
                # Concatenate
                features = torch.cat([semantic_features, acoustic_features], dim=-1)
            else:
                features = semantic_features
            
            # Temporal averaging
            clip_features = features.mean(dim=1)  # [B, D]
            return clip_features.squeeze(0).cpu()
        
        # Approach 3: Monkey-patch approach (advanced)
        else:
            raise NotImplementedError(
                "SemantiCodec does not expose internal encoder features. "
                "To use this extractor, you need to modify the semanticodec library to add:\n\n"
                "def encode_features(self, audio, layer='all'):\n"
                "    '''Extract encoder features before VQ'''\n"
                "    # Your implementation here\n"
                "    return features  # Shape: [B, T, D]\n\n"
                "Alternatively, fork the library and modify the encode() method to return features."
            )


def get_feature_extractor(
    codec_name: str,
    bitrate: float,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    **kwargs
) -> FeatureExtractorBase:
    """Factory function to get the appropriate feature extractor.
    
    Args:
        codec_name: Name of codec ('encodec' or 'semanticodec')
        bitrate: Target bitrate in kbps
        device: Device to run on
        **kwargs: Additional arguments for specific extractors
        
    Returns:
        Appropriate feature extractor instance
        
    Example:
        extractor = get_feature_extractor('encodec', bitrate=6.0, device='cuda')
        features = extractor.extract_clip_features('audio.wav')
    """
    if codec_name.lower() == 'encodec':
        # Choose model based on requirements (default to 24kHz)
        model_name = kwargs.get('model_name', 'encodec_24khz')
        return EncodecFeatureExtractor(
            model_name=model_name,
            bandwidth=bitrate,
            device=device,
            checkpoint_path=kwargs.get('checkpoint_path')
        )
    
    elif codec_name.lower() == 'semanticodec':
        # Map bitrate to SemantiCodec mode
        bitrate_to_mode = {
            0.35: (25, 32768),
            0.71: (50, 32768),
            1.43: (100, 32768),
        }
        
        # Find closest mode
        closest_bitrate = min(bitrate_to_mode.keys(), key=lambda x: abs(x - bitrate))
        token_rate, vocab_size = bitrate_to_mode[closest_bitrate]
        
        return SemantiCodecFeatureExtractor(
            token_rate=token_rate,
            semantic_vocab_size=vocab_size,
            device=device,
            extract_layer=kwargs.get('extract_layer', 'all')
        )
    
    else:
        raise ValueError(f"Unknown codec: {codec_name}. Supported: 'encodec', 'semanticodec'")


if __name__ == '__main__':
    """Test feature extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test codec feature extraction')
    parser.add_argument('audio_path', type=str, help='Path to audio file')
    parser.add_argument('--codec', type=str, default='encodec', choices=['encodec', 'semanticodec'])
    parser.add_argument('--bitrate', type=float, default=6.0, help='Target bitrate in kbps')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print(f"Testing {args.codec} feature extraction at {args.bitrate} kbps")
    print(f"Audio file: {args.audio_path}")
    print(f"Device: {args.device}")
    
    extractor = get_feature_extractor(args.codec, args.bitrate, args.device)
    print(f"Feature dimension: {extractor.feature_dim}")
    
    features = extractor.extract_clip_features(args.audio_path)
    print(f"Extracted features shape: {features.shape}")
    print(f"Feature statistics: mean={features.mean():.4f}, std={features.std():.4f}")
    print("✓ Feature extraction successful!")

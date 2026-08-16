"""Unified interface for encoding audio across different neural audio codecs.

Supports: Encodec, Q2D2, SpeechTokenizer, HiFiCodec, DAC-FSQ
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchaudio
from torchaudio import transforms as T

# Add codec directories to path
_HERE = Path(__file__).resolve().parent
_PROJ_ROOT = _HERE.parent
_ENCODEC_DIR = _PROJ_ROOT / "Encodec"
_Q2D2_DIR = _PROJ_ROOT / "Q2D2"
_HIFICODEC_DIR = _PROJ_ROOT / "hificodec"
_SPEECHTOKENIZER_DIR = _PROJ_ROOT / "SpeechTokenizer"
_DAC_DIR = _PROJ_ROOT / "descript-audio-codec"

for path in [_ENCODEC_DIR, _Q2D2_DIR, _HIFICODEC_DIR, _SPEECHTOKENIZER_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


class CodecEncoder:
    """Base class for codec encoders."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
    
    def encode(self, audio_path: str) -> Dict[str, np.ndarray]:
        """Encode audio file and return tokens + embeddings per layer.
        
        Returns:
            {
                "tokens": ndarray of shape [n_layers, T],
                "embeddings": list of [D, T] arrays per layer,
                "sample_rate": int
            }
        """
        raise NotImplementedError
    
    @property
    def n_layers(self) -> int:
        """Number of quantization layers."""
        raise NotImplementedError


class EncodecEncoder(CodecEncoder):
    """Encodec encoder (RVQ with 32 codebooks at 24kbps)."""
    
    def __init__(self, checkpoint: Path, bandwidth: float = 24.0, device: str = "cuda"):
        super().__init__(device)
        from compress import MODELS
        
        self.model = MODELS["multi_dataset_encodec"](str(checkpoint))
        self.model.to(device).eval()
        self.model.set_target_bandwidth(bandwidth)
        self.bandwidth = bandwidth
    
    @property
    def n_layers(self) -> int:
        return self.model.quantizer.n_q
    
    @torch.no_grad()
    def encode(self, audio_path: str) -> Dict[str, np.ndarray]:
        from utils import convert_audio
        
        wav, sr = torchaudio.load(audio_path)
        wav = convert_audio(wav, sr, self.model.sample_rate, self.model.channels)
        if wav.dim() == 2:
            wav = wav.unsqueeze(0)
        wav = wav.to(self.device)
        
        # Encode
        emb = self.model.encoder(wav)  # [B, D, T]
        
        # Quantize layer by layer
        tokens_list = []
        embeddings_list = []
        residual = emb  # each RVQ stage subtracts the previous stage’s quantised output
        
        for i in range(self.n_layers):
            quantized, indices, _ = self.model.quantizer.vq.layers[i](residual)
            tokens_list.append(indices.cpu().numpy()[0])  # [T] — single batch item
            embeddings_list.append(quantized.cpu().numpy()[0])  # [D, T]
            residual = residual - quantized  # RVQ residual: pass unrepresented signal to next codebook
        
        return {
            "tokens": np.stack(tokens_list),  # [n_layers, T]
            "embeddings": embeddings_list,
            "sample_rate": self.model.sample_rate
        }


class Q2D2Encoder(CodecEncoder):
    """Q2D2 encoder (rhombic grid quantization with 16 grid pairs at 9.8kbps)."""
    
    def __init__(self, checkpoint: Path, device: str = "cuda"):
        super().__init__(device)
        from decoder.feature_extractors import EncodecFeatures
        
        # Load Q2D2 from Lightning checkpoint
        ckpt = torch.load(checkpoint, map_location=device)
        
        self.feature_extractor = EncodecFeatures(
            encodec_model="encodec_24khz",
            bandwidths=[9.8],
            train_codebooks=True,
            num_quantizers=1,
            dowmsamples=[6, 4, 3, 1],
            vq_kmeans=200,
            vq_type='rhombic',
            codebook_dim=[9]*16 + [7]*16
        )
        
        # Load weights
        encodec_state = {
            k.replace("feature_extractor.encodec.", ""): v
            for k, v in ckpt["state_dict"].items()
            if k.startswith("feature_extractor.encodec")
        }
        self.feature_extractor.encodec.load_state_dict(encodec_state, strict=False)
        self.feature_extractor.encodec.to(device).eval()
        self._n_layers: Optional[int] = None
    
    @property
    def n_layers(self) -> int:
        # Some sanity-test checkpoints emit a single composite code stream.
        # Fall back to 1 until we infer the actual output shape at runtime.
        return self._n_layers if self._n_layers is not None else 1  # updated on first encode() call
    
    @torch.no_grad()
    def encode(self, audio_path: str) -> Dict[str, np.ndarray]:
        wav, sr = torchaudio.load(audio_path)
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, sr, 24000)
        wav = wav.mean(dim=0, keepdim=True).to(self.device)  # Mono
        
        emb = self.feature_extractor.encodec.encoder(wav.unsqueeze(0))

        # Q2D2 quantizer expects [B, D, T]. Depending on the encoder path,
        # emb may already be [B, D, T] or may arrive as [B, T, D].
        feat_dim = getattr(self.feature_extractor.encodec.quantizer, "dimension", 512)
        if emb.dim() != 3:
            raise RuntimeError(f"Unexpected Q2D2 encoder rank: {emb.dim()} from {tuple(emb.shape)}")
        if emb.shape[1] == feat_dim:
            features_for_vq = emb  # already [B, D, T]
        elif emb.shape[-1] == feat_dim:
            features_for_vq = emb.permute(0, 2, 1)  # rearrange [B, T, D] → [B, D, T]
        else:
            raise RuntimeError(
                f"Unexpected Q2D2 encoder shape {tuple(emb.shape)}; "
                f"could not align feature dim={feat_dim}."
            )

        # encode_pairs gives [n_q=1, n_pairs=16, B=1, T]; infer() gives packed [1, T]
        pair_codes = self.feature_extractor.encodec.quantizer.encode_pairs(features_for_vq)
        tokens_np = pair_codes[0, :, 0, :].cpu().numpy().astype(np.int32)  # [16, T]
        self._n_layers = int(tokens_np.shape[0])

        frame_rate = getattr(self.feature_extractor, "frame_rate", 75)
        qres = self.feature_extractor.encodec.quantizer.infer(
            features_for_vq, frame_rate=frame_rate
        )
        quantized = qres.quantized

        # Build per-layer embeddings by chunking feature dim of quantized output
        if quantized.dim() != 3:
            raise RuntimeError(
                f"Unexpected Q2D2 quantized tensor shape: {tuple(quantized.shape)}"
            )

        q0 = quantized[0]  # [T, D]
        feat_dim = q0.shape[-1]
        chunk = max(1, feat_dim // self.n_layers)
        embeddings: List[np.ndarray] = []
        for i in range(self.n_layers):
            start = i * chunk
            end = feat_dim if i == self.n_layers - 1 else min(feat_dim, (i + 1) * chunk)
            if start >= feat_dim:
                emb_i = np.zeros((1, q0.shape[0]), dtype=np.float32)
            else:
                emb_i = q0[:, start:end].transpose(0, 1).contiguous().cpu().numpy()
            embeddings.append(emb_i)
        
        return {
            "tokens": tokens_np,
            "embeddings": embeddings,
            "sample_rate": 24000
        }


class HiFiCodecEncoder(CodecEncoder):
    """HiFiCodec encoder (grouped RVQ with 4 codebooks)."""
    
    def __init__(self, checkpoint: Path, config_path: Optional[Path] = None, device: str = "cuda"):
        super().__init__(device)
        from academicodec.models.hificodec.env import AttrDict
        from academicodec.models.hificodec.models import Encoder, Quantizer
        from academicodec.utils import scan_checkpoint, load_checkpoint
        
        # Default config
        if config_path is None:
            candidates = [
                Path(checkpoint) / "config_24k_320d.json",
                Path(checkpoint) / "config.json",
                _HIFICODEC_DIR / "egs" / "hificodec_fsd50k" / "config_24k_320d.json",
                _HIFICODEC_DIR / "egs" / "HiFi-Codec-24k-320d" / "config_24k_320d.json",
            ]
            config_path = next((p for p in candidates if p.exists()), None)
            if config_path is None:
                attempted = "\n".join(f"  - {p}" for p in candidates)
                raise FileNotFoundError(
                    "Could not locate HiFiCodec config file. Tried:\n"
                    f"{attempted}"
                )
        
        with open(config_path) as f:
            self.h = AttrDict(json.load(f))
        
        self.encoder = Encoder(self.h).to(device)
        self.quantizer = Quantizer(self.h).to(device)
        
        # Use g_best (best val checkpoint)
        g_best = Path(checkpoint) / "g_best"
        if not g_best.exists():
            raise FileNotFoundError(
                f"No g_best checkpoint found in {checkpoint}. "
                "Train HiFiCodec first with train_fsd50k.sh"
            )
        
        state = load_checkpoint(str(g_best), device)
        self.encoder.load_state_dict(state["encoder"])
        self.quantizer.load_state_dict(state["quantizer"])
        
        self.encoder.eval()
        self.quantizer.eval()
    
    @property
    def n_layers(self) -> int:
        return self.h.n_code_groups
    
    @torch.no_grad()
    def encode(self, audio_path: str) -> Dict[str, np.ndarray]:
        wav, sr = torchaudio.load(audio_path)
        if sr != self.h.sampling_rate:
            wav = T.Resample(sr, self.h.sampling_rate)(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.unsqueeze(0).to(self.device)  # [1, 1, T]
        
        z = self.encoder(wav)  # [1, C, T_frames]
        # Quantizer returns (quantized_out, loss, all_indices); all_indices is a flat list of [B*T] tensors
        quantized_out, _loss, all_indices = self.quantizer(z)
        B, C, T_frames = quantized_out.shape
        n_streams = len(all_indices)
        chunk = C // n_streams
        tokens_np = np.stack(
            [idx.reshape(B, T_frames).squeeze(0).cpu().numpy() for idx in all_indices], axis=0
        )  # [n_streams, T_frames]
        embeddings = [
            quantized_out[:, i * chunk:(i + 1) * chunk, :].squeeze(0).cpu().numpy()
            for i in range(n_streams)
        ]
        return {
            "tokens": tokens_np.astype(np.int32),
            "embeddings": embeddings,
            "sample_rate": self.h.sampling_rate
        }


class DACFSQEncoder(CodecEncoder):
    """DAC-FSQ encoder (single flat FSQ codebook, 24 kHz)."""

    def __init__(self, checkpoint: Path, device: str = "cuda"):
        super().__init__(device)
        if str(_DAC_DIR) not in sys.path:
            sys.path.insert(0, str(_DAC_DIR))
        import dac
        model_path = Path(checkpoint)
        if (model_path / "latest").exists():
            model_path = model_path / "latest"
        # dac.__init__ only exports DAC, not DAC_FSQ; resolve the submodule directly.
        if hasattr(dac, "model") and hasattr(dac.model, "DAC_FSQ"):
            _cls = dac.model.DAC_FSQ
        elif hasattr(dac, "DAC_FSQ"):
            _cls = dac.DAC_FSQ
        else:
            raise AttributeError("DAC_FSQ not found in dac package")
        # load_from_folder matches train_fsq.py's save convention (model_path/dac_fsq/)
        self.model, _ = _cls.load_from_folder(
            folder=str(model_path), map_location="cpu", package=True
        )
        self.model.eval().to(device)

    @property
    def n_layers(self) -> int:
        return 1  # Single flat FSQ index per frame

    @torch.no_grad()
    def encode(self, audio_path: str) -> Dict[str, np.ndarray]:
        wav, sr = torchaudio.load(audio_path)
        if sr != self.model.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.model.sample_rate)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.unsqueeze(0).to(self.device)  # [1, 1, T]
        wav = self.model.preprocess(wav, self.model.sample_rate)
        z_out, codes, latents, _, _ = self.model.encode(wav)
        # codes: [B, 1, T]; latents: [B, fsq_dim=8, T]
        tokens_np = codes.squeeze(0).cpu().numpy().astype(np.int32)  # [1, T]
        lat = latents.squeeze(0).cpu().numpy()                        # [8, T]
        return {
            "tokens": tokens_np,
            "embeddings": [lat],
            "sample_rate": self.model.sample_rate,
        }


class SpeechTokenizerEncoder(CodecEncoder):
    """SpeechTokenizer encoder (RVQ with 8 codebooks)."""
    
    def __init__(self, checkpoint: Path, config_path: Path = None, device: str = "cuda"):
        super().__init__(device)
        from speechtokenizer.model import SpeechTokenizer
        
        # Default config (prefer the same configs used in run_pipeline sanity/scale tests)
        if config_path is None:
            cfg_candidates = [
                _SPEECHTOKENIZER_DIR / "config" / "fsd50k_cfg.json",
                _SPEECHTOKENIZER_DIR / "config" / "fsd50k_cfg_test_2ep.json",
                _SPEECHTOKENIZER_DIR / "config" / "spt_base_cfg.json",
                _SPEECHTOKENIZER_DIR / "speechtokenizer" / "config.json",
            ]
            config_path = next((p for p in cfg_candidates if p.exists()), None)
            if config_path is None:
                attempted = "\n".join(f"  - {p}" for p in cfg_candidates)
                raise FileNotFoundError(
                    "Could not locate SpeechTokenizer config file. Tried:\n"
                    f"{attempted}"
                )
        
        self.model = SpeechTokenizer.load_from_checkpoint(
            config_path=str(config_path),
            ckpt_path=str(checkpoint)
        )
        self.model.eval().to(device)
    
    @property
    def n_layers(self) -> int:
        return self.model.n_q
    
    @torch.no_grad()
    def encode(self, audio_path: str) -> Dict[str, np.ndarray]:
        wav, sr = torchaudio.load(audio_path)
        if sr != self.model.sample_rate:
            wav = T.Resample(sr, self.model.sample_rate)(wav)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.unsqueeze(0).to(self.device)  # [1, 1, T]
        
        # Encode to RVQ codes
        codes = self.model.encode(wav)  # [n_q, 1, T]
        
        # Get quantized features per layer
        quantized_list = self.model.forward_feature(wav, layers=list(range(self.n_layers)))
        
        tokens = codes.squeeze(1).cpu().numpy()  # [n_q, T]
        embeddings = [q.squeeze(0).cpu().numpy() for q in quantized_list]  # list of [D, T]
        
        return {
            "tokens": tokens.astype(np.int32),
            "embeddings": embeddings,
            "sample_rate": self.model.sample_rate
        }


def get_codec_encoder(codec_name: str, checkpoint: Path, **kwargs) -> CodecEncoder:
    """Factory function to get codec encoder by name."""
    encoders = {
        "encodec":          EncodecEncoder,
        "q2d2":             Q2D2Encoder,
        "hificodec":        HiFiCodecEncoder,
        "speechtokenizer":  SpeechTokenizerEncoder,
        "dac_fsq":          DACFSQEncoder,
    }

    if codec_name not in encoders:
        raise ValueError(f"Unknown codec: {codec_name}. Available: {list(encoders.keys())}")

    return encoders[codec_name](checkpoint, **kwargs)

#!/bin/bash
# Quick setup script for consolidated training environment
# Usage: bash setup_env.sh

set -e  # Exit on error

echo "=========================================="
echo "Setting up consolidated training environment"
echo "=========================================="

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip and install correct setuptools version FIRST
echo "Upgrading pip and installing setuptools..."
pip install --upgrade pip wheel
pip install 'setuptools>=65.0.0,<70.0.0'  # CRITICAL: Must be <70 for pkg_resources

# Verify pkg_resources is available before continuing
echo "Verifying pkg_resources..."
python -c "import pkg_resources; print('✓ pkg_resources: Available')" || {
    echo "❌ ERROR: pkg_resources not available after setuptools install"
    exit 1
}

# Install uv if not available (much faster than pip)
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    pip install uv
fi

# Install PyTorch 2.4.0 with CUDA 12.4 support FIRST
echo "Installing PyTorch 2.4.0 with CUDA 12.4 support..."
uv pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

echo "Installing additional audio codec dependencies..."
uv pip install einops

echo "Installing remaining dependencies with uv..."
uv pip install -r requirements.txt

# Editable installs for submodule packages
echo "Installing editable submodule packages..."
uv pip install -e ./SpeechTokenizer
uv pip install -e ./descript-audio-codec

# Verify all critical imports work
echo "Verifying critical imports..."
python << 'PYEOF'
errors = []
checks = [
    ("torch", "PyTorch"),
    ("torchaudio", "torchaudio"),
    ("pytorch_lightning", "PyTorch Lightning"),
    ("wandb", "W&B"),
    ("accelerate", "Accelerate"),
    ("beartype", "beartype (SpeechTokenizer)"),
    ("dac", "dac (DAC-FSQ)"),
    ("speechtokenizer", "SpeechTokenizer"),
    ("vector_quantize_pytorch", "vector-quantize-pytorch (DAC-FSQ)"),
    ("lion_pytorch", "lion_pytorch (SpeechTokenizer)"),
]
for mod, name in checks:
    try:
        __import__(mod)
        print(f"  ✓ {name}")
    except ImportError as e:
        print(f"  ❌ {name}: {e}")
        errors.append(name)

import numpy as np
if np.__version__.startswith("2."):
    print(f"  ❌ NumPy {np.__version__} — must be 1.x!")
    errors.append("numpy")
else:
    print(f"  ✓ NumPy {np.__version__} (1.x OK)")

import torch
print(f"  ✓ PyTorch {torch.__version__}")
print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  ✓ CUDA version: {torch.version.cuda}")
    print(f"  ✓ GPU device: {torch.cuda.get_device_name(0)}")

if errors:
    print(f"\n❌ {len(errors)} import(s) failed: {', '.join(errors)}")
    raise SystemExit(1)
else:
    print("\n✅ All imports OK!")
PYEOF

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Training scripts available:"
echo "  - train_speechtokenizer_fsd50k.sh"
echo "  - train_fsd50k.sh (HiFiCodec)"
echo "  - train_dac_fsq_fsd50k.sh"
echo "  - Q2D2/configs/Q2D2_fsd50k_9.8kbps_dim512_attn_b16.yaml"
echo "  - Encodec/config/config_multi_dataset.yaml"

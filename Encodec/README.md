# encodec-train

## Introduction
This repository is based on [encodec](https://github.com/facebookresearch/encodec) 

[EnCodec_Trainer](https://github.com/Mikxox/EnCodec_Trainer)

[melgan-neurips:](https://github.com/descriptinc/melgan-neurips)

[encodec-pytorch](https://github.com/ZhikangNiu/encodec-pytorch)

## Prerequisites for running `run_multi_dataset_training.sh`

### 1. Virtual Environment
- **Location**: `./venv_encodec/` (must exist)
Use Python 3.13.4, pip install -r requirements.txt to set it up. 

### 2. Dataset CSV Files

The script expects CSV files with audio file paths. 

#### Required CSV Files (from `config/config_multi_dataset.yaml`):

**Jamendo Dataset:**
- `/path/to/jamendo/jamendo_train.csv`
- `/path/to/jamendo/jamendo_val.csv`
- `/path/to/jamendo/jamendo_test.csv`

**Common Voice Dataset:**
- `/path/to/common_voice/common_voice_train.csv`
- `/path/to/common_voice/common_voice_val.csv`
- `/path/to/common_voice/common_voice_test.csv`

**FSD50K Dataset:**
- `/path/to/fsd50k/fsd50k_train.csv`
- `/path/to/fsd50k/fsd50k_val.csv`
- `/path/to/fsd50k/fsd50k_test.csv`

**DNS Challenge 4 Dataset:**
- `/path/to/dns_challenge4/dns_challenge4_train.csv`
- `/path/to/dns_challenge4/dns_challenge4_valid.csv`
- `/path/to/dns_challenge4/dns_challenge4_test.csv`


#### Generating CSV Files:
Use `datasets/generate_dataset_csvs.py`:
```bash
python datasets/generate_dataset_csvs.py \
    -i /path/to/your/dataset \
    --three_way_split \
    --train_ratio 0.995 \
    --val_ratio 0.0025 \
    --test_ratio 0.0025
```

**Note**: If a CSV file doesn't exist, the script will log a warning but continue with other datasets. At least ONE dataset must be available.

### 3. Audio Files
- Audio files referenced in CSV files must exist and be readable
- Supported formats: `.wav`, `.flac`, `.mp3` (via librosa)
- Files will be automatically resampled and converted to the format specified in config (see `model.sample_rate` and `model.channels`)

### 4. Weights & Biases (WandB) Configuration

#### WandB Setup:
1. **Login**: `wandb login` (first time only)
2. **Disable**: Set `wandb.enabled=false` in config or script
3. **Entity**: Can be set via `wandb.entity` in config (optional)

#### What's Logged to WandB:

**Training Metrics (every epoch):**
- `train/loss_g` - Generator loss
- `train/loss_w` - Quantizer commitment loss
- `train/loss_disc` - Discriminator loss (after warmup)
- `train/l_t` - Time domain loss
- `train/l_f` - Frequency domain loss
- `train/l_g` - Generator adversarial loss
- `train/l_feat` - Feature matching loss
- `train/lr_g` - Generator learning rate
- `train/lr_d` - Discriminator learning rate

**Validation Metrics (every epoch):**
- `val/loss_g` - Average generator loss across all bandwidths
- `val/loss_disc` - Average discriminator loss
- `val/si_snr` - Average SI-SNR across all bandwidths
- `val/si_snr_bw_{bandwidth}` - SI-SNR for each bandwidth (1.5, 3.0, 6.0, 12.0, 24.0)
- `val/si_snr_bw_{bandwidth}_ci_low` - Lower confidence interval
- `val/si_snr_bw_{bandwidth}_ci_high` - Upper confidence interval

**Model Artifacts (every epoch):**
- Model checkpoint files uploaded as WandB artifacts

**WandB Project Settings:**
- **Project Name**: `multi-dataset-encodec` (from script or config)
- **Run Name**: Configured in script or config (e.g., `multi_dataset_bs16_epochs300`)
  - Uses config interpolation: `multi_dataset_bs${datasets.batch_size}_epochs${common.max_epoch}`
  - **Note**: WandB name is only for logging/organization and does NOT affect checkpoint filenames or resume behavior
- **Config**: Full Hydra config is logged automatically

### 5. GPU Requirements
- **CUDA**: Script sets `CUDA_VISIBLE_DEVICES=0` (uses GPU 0)
- **Memory**: Uses `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 

### 6. Resume Training (Optional)
If `checkpoint.resume=True` in config:
- **Model checkpoint**: `checkpoint.checkpoint_path` must exist
- **Discriminator checkpoint**: `checkpoint.disc_checkpoint_path` must exist
- Training resumes from the epoch saved in checkpoint


## Running the Script

```bash
./run_multi_dataset_training.sh
```

Or make it executable first:
```bash
chmod +x run_multi_dataset_training.sh
./run_multi_dataset_training.sh
```

## Using Trained Checkpoints

### 1. Audio Reconstruction and Testing (`test.py`)

Use `test.py` to reconstruct audio from checkpoints, calculate metrics, and create spectrograms:

```bash
python test.py \
    --model_checkpoint checkpoints_multi_dataset/bs16_cut24000_length32000_epoch335_lr0.0003.pt \
    --disc_checkpoint checkpoints_multi_dataset/bs16_cut24000_length32000_epoch335_disc_lr0.0003.pt \
    --demo_dir ./demo \
    --bandwidths 1.5 3.0 6.0 12.0 24.0 \
    --device cpu
```

**Note**: Use `--device cuda` if you have CUDA-enabled PyTorch installed. On Mac or systems without CUDA, use `--device cpu`.

**Note**: `--project` is optional (defaults to `multi-dataset-encodec`). Only specify it if you want to use a different WandB project.

**What it does:**
- Loads model and discriminator from checkpoints
- Reconstructs audio at multiple bandwidths (1.5, 3.0, 6.0, 12.0, 24.0 kbps)
- Calculates SI-SNR metrics
- Creates spectrograms for visualization
- Logs results to WandB in organized tables

**Arguments:**
- `--model_checkpoint`: Path to model checkpoint (required)
- `--disc_checkpoint`: Path to discriminator checkpoint (required)
- `--demo_dir`: Directory containing demo folders with `*_gt.wav` files (default: `./demo`)
- `--bandwidths`: Bandwidths to test (default: `1.5 3.0 6.0 12.0 24.0`)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`). Use `cpu` if PyTorch is not compiled with CUDA support.
- `--project`: WandB project name (optional, default: `multi-dataset-encodec`)
- `--entity`: WandB entity/username (optional)

### 2. Audio Compression/Decompression (`main.py`)

Use `main.py` (which uses `compress.py`) to compress audio to `.ecdc` format or decompress:

**Compress audio to .ecdc format:**
```bash
python main.py input.wav output.ecdc \
    --model_name multi_dataset_encodec \
    --checkpoint checkpoints_multi_dataset/bs16_cut24000_length32000_epoch335_lr0.0003.pt \
    --bandwidth 6.0 \
    --device cpu
```

**Decompress .ecdc to .wav:**
```bash
python main.py output.ecdc output.wav \
    --model_name multi_dataset_encodec \
    --checkpoint checkpoints_multi_dataset/bs16_cut24000_length32000_epoch335_lr0.0003.pt \
    --device cpu
```

**Compress and immediately decompress (round-trip):**
```bash
python main.py input.wav output.wav \
    --model_name multi_dataset_encodec \
    --checkpoint checkpoints_multi_dataset/bs16_cut24000_length32000_epoch335_lr0.0003.pt \
    --bandwidth 6.0 \
    --device cpu
```

**Note**: Use `--device cuda` if you have CUDA-enabled PyTorch installed. On Mac or systems without CUDA, use `--device cpu` (or omit it, as `cpu` is the default).

**Arguments:**
- `input`: Input file (`.wav` for compression, `.ecdc` for decompression)
- `output`: Output file (optional, inferred from input if not provided)
- `--model_name`: Model type (`multi_dataset_encodec` for trained checkpoints)
- `--checkpoint`: Path to checkpoint file (required for `multi_dataset_encodec`)
- `--bandwidth`: Target bandwidth in kbps (1.5, 3.0, 6.0, 12.0, or 24.0, default: 6.0)
- `--device`: Device to use (`cuda` or `cpu`, default: `cuda`). Use `cpu` if PyTorch is not compiled with CUDA support.
- `--rescale`: Automatically rescale output to avoid clipping
- `--force`: Overwrite output file if it exists
- `--lm`: Use language model for better compression (slower)

**Checkpoint filename format:**
Checkpoints are saved as:
```
checkpoints_multi_dataset/bs{batch_size}_cut{tensor_cut}_length{fixed_length}_epoch{epoch}_lr{lr}.pt
```

Example: `bs16_cut24000_length32000_epoch300_lr0.0003.pt`

## Troubleshooting during training

### "CSV file not found" warnings
- Script will continue with other datasets
- At least one dataset must be available
- Check CSV paths in `config/config_multi_dataset.yaml`

### "No audio files found"
- Check CSV files contain valid paths
- Verify audio files exist and are readable
- Check file permissions

### WandB errors
- Run `wandb login` first
- Or disable WandB: set `wandb.enabled=false` in config
- Check internet connection (for cloud sync)

### Out of memory
- Reduce `datasets.batch_size` in config
- Reduce `datasets.num_workers`

### Checkpoint loading errors
- Verify checkpoint paths in config
- Check checkpoint format matches model architecture
- Ensure both model and discriminator checkpoints exist if resuming


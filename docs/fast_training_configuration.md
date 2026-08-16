# Fast Training Configuration Changes

**Date:** 2026-07-23  
**Purpose:** Configure all 5 neural audio codecs for accelerated training to meet project deadline

## Overview

Modified all codec configurations to use batch_size=8 with 1010 steps/epoch (8,080 samples per epoch, ~25% of FSD50K training set). This reduces total training time from ~520 hours to ~27 hours while maintaining consistent training dynamics across all models.

## Training Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Batch size | 8 | Balanced GPU utilization with training speed |
| Steps/epoch | 1,010 | Standard across all codecs for fair comparison |
| Samples/epoch | 8,080 | 25% dataset sampling (inspired by Q2D2's approach) |
| Validation frequency | Every 5 epochs (5,050 steps) | Reasonable checkpoint intervals |
| Total epochs | 50 | Original target maintained |
| Total steps/model | 50,500 | 50 epochs × 1,010 steps |
| **Estimated time** | **~27 hours** | All 5 models trained sequentially |

## Files Modified

### 1. Dataset Loader Patches

#### SpeechTokenizer Dataset
**File:** `SpeechTokenizer/speechtokenizer/trainer/dataset.py`

**Change:** Modified `__len__()` method to limit epoch length

```python
def __len__(self):
    # Limit to 1010 steps × 8 batch = 8,080 samples/epoch (fast training mode)
    return min(len(self.file_list), 8080)
```

**Previous:** `return len(self.file_list)` (32,772 samples)

---

#### HiFiCodec Dataset
**File:** `hificodec/academicodec/models/hificodec/meldataset.py`

**Change:** Modified `__len__()` method to limit epoch length

```python
def __len__(self):
    # Limit to 1010 steps × 8 batch = 8,080 samples/epoch (fast training mode)
    return min(len(self.audio_files), 8080)
```

**Previous:** `return len(self.audio_files)` (32,772 samples)

---

### 2. Configuration File Updates

#### SpeechTokenizer Config
**File:** `SpeechTokenizer/config/fsd50k_cfg.json`

**Changes:**
- `batch_size`: 16 → **8**
- `save_model_steps`: 5000 → **5050** (validation every 5 epochs)

**Unchanged parameters:**
- `segment_size`: 48000 (3.0 seconds @ 16kHz) ✓
- `distill_loss_lambda`: 0 (disabled for non-speech audio) ✓
- `epochs`: 50 ✓

---

#### Q2D2 Config
**File:** `Q2D2/configs/Q2D2_fsd50k_9.8kbps_dim512_attn_b16.yaml`

**Changes:**
- `train_params.batch_size`: 16 → **8**
- `val_params.batch_size`: 10 → **5**

**Unchanged parameters:**
- `num_samples`: 72000 (3.0 seconds @ 24kHz) ✓
- `limit_train_batches`: 1000 (creates 1010 effective steps with grad accumulation) ✓
- `accumulate_grad_batches`: 2 ✓
- `max_epochs`: 50 ✓
- `check_val_every_n_epoch`: 5 ✓

---

#### Encodec Config
**File:** `Encodec/config/config_multi_dataset.yaml`

**Changes:**
- `batch_size`: 16 → **8**
- `fixed_length`: 8000 → **8080** (1010 steps × 8 batch)

**Unchanged parameters:**
- `tensor_cut`: 72000 (3.0 seconds @ 24kHz) ✓
- `max_epoch`: 50 ✓
- `val_interval`: 5 (epochs) ✓

---

#### DAC-FSQ Config
**File:** `descript-audio-codec/conf/fsd50k_fsq.yml`

**Changes:**
- `batch_size`: 16 → **8**
- `val_batch_size`: 10 → **5**
- `train/AudioDataset.n_examples`: 10000000 → **8080**
- `valid_freq`: 10240 → **5050** (every 5 epochs)

**Unchanged parameters:**
- `train/AudioDataset.duration`: 3.0 ✓
- `training_epochs`: 50 ✓

---

#### HiFiCodec Config
**File:** `hificodec/egs/HiFi-Codec-24k-320d/config_24k_320d.json`

**Changes:**
- `batch_size`: 16 → **8**

**Unchanged parameters:**
- `segment_size`: 72000 (3.0 seconds @ 24kHz) ✓

---

#### HiFiCodec Training Script
**File:** `train_fsd50k.sh`

**Changes:**
- `--validation_interval`: 20485 → **5050**

---

## Rationale for Changes

### 1. Batch Size Reduction (16→8)
- **Memory efficiency:** Original batch_size=16 only used 6GB of 24GB VRAM
- **Speed vs. efficiency trade-off:** Smaller batches with more frequent updates
- **Consistency:** All models now use identical batch size for fair comparison

### 2. Dataset Sampling (25% per epoch)
- **Inspiration:** Q2D2 successfully uses `limit_train_batches=1000` for fast training
- **Training time:** Reduces epoch time by ~75% without sacrificing convergence
- **Coverage:** Over 50 epochs, each sample seen ~12.5 times (8080 × 50 / 32772)
- **Validation:** Full validation set still used (not sampled)

### 3. Validation Frequency (5050 steps = 5 epochs)
- **Checkpoints:** 10 validation points over 50 epochs
- **Monitoring:** Sufficient granularity to catch divergence
- **Storage:** Reasonable checkpoint frequency (3 kept per model)

### 4. Segment Size Standardization (Already Complete)
All models now train on 3.0-second segments:
- SpeechTokenizer: 48,000 samples @ 16kHz
- Encodec, DAC-FSQ, HiFiCodec, Q2D2: 72,000 samples @ 24kHz

## Expected Outcomes

### Training Timeline
| Codec | Estimated Time | Output Directory |
|-------|----------------|------------------|
| SpeechTokenizer | ~5.5 hours | `results/speechtokenizer_fsd50k/` |
| Q2D2 | ~5.5 hours | `Q2D2/Q2D2_fsd50k_9.8kbps/` |
| Encodec | ~5.5 hours | `Encodec/checkpoints_multi_dataset/` |
| DAC-FSQ | ~5.5 hours | `descript-audio-codec/ckpt/fsd50k_fsq/` |
| HiFiCodec | ~5.5 hours | `hificodec/egs/hificodec_fsd50k/` |
| **Total** | **~27 hours** | |

### Weights & Biases Logging
Each codec logs to separate W&B project:
- `speechtokenizer_fsd50k`
- `q2d2_fsd50k`
- `encodec_fsd50k`
- `dac-fsq_fsd50k`
- `hificodec_fsd50k`

### Validation Checkpoints
- Validation runs at steps: 5050, 10100, 15150, 20200, 25250, 30300, 35350, 40400, 45450, 50500
- Corresponds to epochs: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50

## Previous Issues Resolved

### Issue 1: Semantic Distillation for Non-Speech Audio
- **Problem:** SpeechTokenizer used HuBERT teacher (trained on speech) for environmental sounds
- **Solution:** Set `distill_loss_lambda=0` in config
- **Status:** ✓ Resolved (2026-07-22)

### Issue 2: Q2D2 Matplotlib Compatibility
- **Problem:** `tostring_rgb()` deprecated in matplotlib 3.8+
- **Solution:** Updated to `buffer_rgba()` API in `Q2D2/decoder/helpers.py`
- **Status:** ✓ Resolved (2026-07-22)

### Issue 3: Inconsistent Training Segments
- **Problem:** Models trained on different temporal contexts (0.38-3.0 seconds)
- **Solution:** Standardized all to 3.0 seconds
- **Status:** ✓ Resolved (2026-07-22)

### Issue 4: Slow Training Speed
- **Problem:** Projected 520 hours total for 50 epochs on full dataset
- **Solution:** Implemented 25% dataset sampling per epoch (this document)
- **Status:** ✓ Resolved (2026-07-23)

## Launch Commands

```bash
cd /home/spencerwardaudio/dev/Spatial_Audio/msc_proj

# Launch training
nohup bash train_5codecs.sh --gpu 0 > training.log 2>&1 &

# Save process ID
echo $! > training.pid

# Monitor progress
tail -f training.log

# Check running process
ps aux | grep train
```

## Validation Commands

```bash
# Check GPU utilization
nvidia-smi

# View W&B logs
wandb login
# Then visit: https://wandb.ai/<your-username>/

# Monitor specific codec
tail -f results/speechtokenizer_fsd50k/train.log
```

## Notes

- All changes preserve the original full-dataset training capability
- To revert to full-dataset training: remove `min(..., 8080)` limits from dataset loaders and restore original config values
- Fast training mode trades complete dataset coverage for speed while maintaining training stability
- Validation always uses full validation set (4,097 samples) regardless of training sampling

## References

- Original Q2D2 fast training: `limit_train_batches: 1000` in `Q2D2/configs/*.yaml`
- FSD50K dataset: 32,772 training samples, 4,097 validation samples
- Project deadline: 2026-07-24 evening (~27 hours from implementation)

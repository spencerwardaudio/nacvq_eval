# Change Log

## 2026-06-08 — Q²D² Quantisation + Geometry-Aware Losses

### Branch: `Q2D2_L_Mod`

---

### 1. `Encodec/quantization/core_vq.py`

#### New helper functions
- `_build_square_lattice(side)` — generates a `(side², 2)` tensor of square-grid coordinates normalised to `[-1, 1]`.
- `_build_hex_lattice(side)` — generates a `(side², 2)` tensor of hexagonal close-packed coordinates normalised to `[-1, 1]`. Alternating rows are offset by 0.5 units to achieve the hex packing.

#### New class: `Q2D2Codebook(nn.Module)`
A drop-in replacement for `EuclideanCodebook` that uses a **fixed geometric lattice** instead of EMA-learned vectors.

- Constructor takes `dim`, `codebook_size`, and `grid` (`"hex"` or `"square"`).
- The 2-D lattice coordinates are tiled across all `dim/2` dimension pairs to produce a `[codebook_size, dim]` embedding matrix stored as a non-trainable buffer.
- The grid is **never updated** during training — only the encoder learns to map into it.
- Implements the same API as `EuclideanCodebook`: `encode()`, `decode()`, `forward()` returning `(quantize, embed_ind)`.
- Requires `dim` to be even; raises `ValueError` otherwise.

#### Modified: `VectorQuantization.__init__`
- Added `use_q2d2: bool = False` and `q2d2_grid: str = "hex"` parameters.
- When `use_q2d2=True`, instantiates `Q2D2Codebook` instead of `EuclideanCodebook`.

#### Modified: `VectorQuantization.forward`
- The commitment loss block is guarded by `and not self.use_q2d2` — when the fixed lattice is active there is no EMA target to commit to, so the loss is always exactly zero.

---

### 2. `Encodec/quantization/vq.py`

#### Modified: `ResidualVectorQuantizer.__init__`
- Added `use_q2d2: bool = False` and `q2d2_grid: str = "hex"` parameters.
- Both are forwarded to `ResidualVectorQuantization(...)`, which passes them through `**kwargs` to each `VectorQuantization` layer.

---

### 3. `Encodec/model.py`

#### Modified: `EncodecModel._get_model`
- Added `use_q2d2: bool = False` and `q2d2_grid: str = "hex"` parameters.
- Both are forwarded to `qt.ResidualVectorQuantizer(...)`.

---

### 4. `Encodec/losses.py`

#### New function: `complex_stft_loss(input_wav, output_wav, n_fft=1024, hop_length=256)`
Penalises errors in both the **real** and **imaginary** components of the STFT:

$$\mathcal{L}_\text{Complex} = \|\mathcal{R}(X) - \mathcal{R}(\hat{X})\|_F^2 + \|\mathcal{I}(X) - \mathcal{I}(\hat{X})\|_F^2$$

Forces the encoder to maintain sub-frame phase alignment — directly targeting the phase insensitivity exposed by the flip-rate analysis (codebooks show ~85 % flip rate regardless of phase offset).

#### New function: `group_delay_loss(input_wav, output_wav, n_fft=1024, hop_length=256)`
Penalises mismatches in the **group delay** spectrum, approximated via finite differences along the STFT frequency axis:

$$\tau_g(\omega) \approx -\Delta_\omega\, \angle X(\omega)$$

Forces the encoder to scale codebook trajectories proportionally to absolute time offsets — targeting the temporal flatlines observed across all codebooks in the flip-rate analysis (~50 % flip rate regardless of time offset magnitude).

#### Modified: `total_loss()`
- Computes both new losses on every forward pass.
- Returns `l_complex` and `l_gd` in the loss dict alongside the existing `l_t`, `l_f`, `l_g`, `l_feat`.

---

### 5. `Encodec/train_multi_dataset.py`

#### Modified: `train_one_step()`
- Both the AMP (`scaler`) and non-AMP paths now include the new losses in the generator loss:

```
loss_g = 3·l_g + 3·l_feat + l_t/10 + l_f + 2.0·l_complex + 1.0·l_gd  [+ loss_w]
```

#### Modified: `EncodecModel._get_model` call in `main()`
- Reads `config.model.use_q2d2` and `config.model.q2d2_grid` and forwards them to `_get_model`.

---

### 6. `Encodec/config/config_multi_dataset.yaml`

- `checkpoint.resume` set to `False` and checkpoint paths cleared (fresh training run).
- Added to `model` block:
  ```yaml
  use_q2d2: True
  q2d2_grid: 'hex'
  ```

---

## 2026-07-27 — Remove experimental losses; fix Encodec crash + SpeechTokenizer validation crash

### Context
The `complex_stft_loss` / `group_delay_loss` modifications documented in the 2026-06-08 entry are excluded from the comparative analysis — they alter the original Encodec/Q2D2 design and would invalidate fair comparison against the baseline. The training script references to `l_complex` and `l_gd` were never backed by an implementation in `losses.py`, causing a crash on first training step.

### 1. `Encodec/losses.py` — no change (reverted to original design)

The two functions were not present in `losses.py`; `total_loss()` only returns `{l_t, l_f, l_g, l_feat}` matching the original Encodec design.

### 2. `Encodec/train_multi_dataset.py` — remove `l_complex` / `l_gd` from `train_one_step()`

**Root cause:** Both the AMP and non-AMP fallback paths referenced `losses_g['l_complex']` and `losses_g['l_gd']`, keys that do not exist in the `total_loss()` return dict, causing `KeyError: 'l_complex'` on the first training step.

**Fix:** Removed both keys from the generator loss formula in both paths. The AMP path now matches the loss weights used by the balancer config (`l_g×3, l_feat×3, l_t/10, l_f×1`).

### 3. `SpeechTokenizer/speechtokenizer/trainer/trainer.py` — `None` semantic feature in epoch validation

**Root cause:** The steps-based training loop already guarded `if semantic_feature is not None` before calling `self.distill_loss()`. The epoch-based validation block (every 5 epochs) was missing the same guard, causing `AttributeError: 'NoneType' object has no attribute 'size'` at epoch 5 validation when no semantic teacher model is loaded.

**Fix:** Changed epoch-validation distill line to:
```python
loss_distill = self.distill_loss(feature, semantic_feature).item() if semantic_feature is not None else 0.0
```

---

## 2026-07-27 — Fix `loss_g: nan` — fp16 zero-denominator in `l_feat` + disable AMP

### Root cause

`loss_g` logged as `nan` from epoch 1 while `loss_disc` remained valid.

The relative feature matching loss in `total_loss()` normalises each feature-map term by its own mean absolute value:

```python
l_feat += l1Loss(fmap_real[k][l], fmap_fake[k][l]) / torch.mean(torch.abs(fmap_real[k][l]))
```

Under AMP (`fp16`), early-training discriminator intermediate feature maps can underflow to exactly zero in fp16 (min normalised fp16 ≈ 6×10⁻⁵). The denominator becomes 0 → `l_feat = inf/NaN` → `loss_g = NaN`. `loss_disc` is unaffected because it uses only `logits`, not feature maps.

Additionally, `amp: True` bypasses the balancer entirely and routes to the hardcoded AMP loss formula. The model config defines `balancer.weights = {l_t, l_f, l_g, l_feat}`, making the **balancer (non-AMP) path the intended training path**. Running with `amp: True` was never the designed configuration.

### Fixes

#### 1. `Encodec/losses.py` — add epsilon to `l_feat` denominator
```python
# before
l_feat += l1Loss(...) / torch.mean(torch.abs(fmap_real[tt1][tt2]))

# after
l_feat += l1Loss(...) / (torch.mean(torch.abs(fmap_real[tt1][tt2])) + 1e-8)
```

#### 2. `Encodec/config/config_multi_dataset.yaml` — disable AMP
```yaml
# before
amp: True

# after
amp: False  # routes training through the balancer as designed
```

With `amp: False`, training uses `balancer.backward()` which handles each loss term independently — matching the original Encodec design and eliminating the fp16 instability path.

---

## 2026-07-27 — Encodec scale-test validation made proportional to train size

### Root cause

Encodec validation size was hardcoded to 10,000 segments per epoch in `multi_dataset.py`, regardless of scale-test stage size (`n=50`, `n=100`, etc.).

At the same time, `run_pipeline.py` was passing `datasets.fixed_length=<n_samples>` for Encodec. In Encodec this field was reused for two unrelated meanings:
- Train dataset size cap (count of examples)
- Validation segment duration (`fixed_length / sample_rate`)

So at `n=50` the run could unintentionally evaluate tiny ~2 ms validation segments while still evaluating against 10k segments — not proportional and not representative of the intended 8:1:1 scale-test protocol.

### Fixes

#### 1. `Encodec/multi_dataset.py`
- Added explicit dataset knobs:
  - `n_train_examples` (train count cap)
  - `n_val_segments` (validation segment count)
  - `n_test_segments` (test segment count)
  - `segment_duration_samples` (audio window length)
- Validation/test fixed-segment generation now uses configurable segment counts (`n_val_segments`, `n_test_segments`) instead of hardcoded 10k/1k.
- Segment duration is now decoupled from sample-count controls and uses `segment_duration_samples / sample_rate`.
- Train `__len__` now respects `n_train_examples` when provided.

#### 2. `Encodec/config/config_multi_dataset.yaml`
- Added defaults:
  - `n_train_examples: null`
  - `n_val_segments: 10000`
  - `n_test_segments: 1000`
  - `segment_duration_samples: 72000`
- Kept `fixed_length` as a legacy fallback for backward compatibility.

#### 3. `run_pipeline.py`
- Encodec scale-test overrides now pass:
  - `datasets.n_train_examples=<n_samples>`
  - `datasets.n_val_segments=max(1, n_samples // 8)`
- Removed overload of `datasets.fixed_length` for Encodec scale stages.

#### 4. `Encodec/train_multi_dataset.py`
- Validation log line now reports actual validation segment count from the dataloader dataset, replacing the hardcoded `10k` message.

---

## 2026-07-27 — Scale-test proportional validation applied to SpeechTokenizer and HiFiCodec

### Root cause

Scale-test was passing only `TRAIN_N_SAMPLES` to SpeechTokenizer and HiFiCodec. In both codepaths, dataset `__len__` used `TRAIN_N_SAMPLES` for training and validation datasets, so validation set size unintentionally matched train size instead of the intended `n_val = max(1, n_train // 8)`.

### Fixes

#### 1. `SpeechTokenizer/speechtokenizer/trainer/dataset.py`
- `audioDataset.__len__()` now selects env key by mode:
  - train: `TRAIN_N_SAMPLES`
  - validation (`valid=True`): `VAL_N_SAMPLES`

#### 2. `hificodec/academicodec/models/hificodec/meldataset.py`
- Added `valid: bool = False` constructor arg and stored `self.valid`.
- `MelDataset.__len__()` now selects env key by mode:
  - train: `TRAIN_N_SAMPLES`
  - validation (`valid=True`): `VAL_N_SAMPLES`

#### 3. `hificodec/academicodec/models/hificodec/train.py`
- Validation dataset instantiation now passes `valid=True` so `VAL_N_SAMPLES` cap applies only to validation loader.

#### 4. `run_pipeline.py`
- In scale-test runs, SpeechTokenizer and HiFiCodec now receive:
  - `TRAIN_N_SAMPLES = n_samples`
  - `VAL_N_SAMPLES = max(1, n_samples // 8)`

### Result

All scale-test models now use proportional validation sizing at each stage:
- DAC-FSQ: explicit `--n-val-examples`
- Q2D2: `--trainer.limit_val_batches`
- Encodec: `datasets.n_val_segments`
- SpeechTokenizer: `VAL_N_SAMPLES`
- HiFiCodec: `VAL_N_SAMPLES`

### Quick note

For any scale-test stage with train size `n`, validation now consistently uses:

`n_val = max(1, floor(n / 8))`

This aligns all five models to the same 8:1 train:val scaling policy for fair stage-by-stage comparison.

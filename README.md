# Neural Audio Codec Benchmark Framework

Trains and evaluates five neural audio codecs trained on FSD50K and their embedding latent vector selections:
**SpeechTokenizer · Encodec · DAC-FSQ · HiFiCodec · Q2D2**

---

## Quick Start

### Step 1 — Clone and set up environment 

```bash
git clone https://github.com/spencerwardaudio/nacvq_eval.git nacvq_eval && cd nacvq_eval
git submodule update --init --recursive
bash setup_env.sh
```

### Step 2 — Launch full pipeline

Runs download → splits → train (200 steps/epoch) → EGFx → analysis inside a `tmux` session (falls back to `nohup` if tmux is absent). Survives SSH disconnects.

```bash
bash launch_pipeline.sh              # FSD50K saved to ./fsd50k (default)
bash launch_pipeline.sh /data/fsd50k # or specify a path
```

Monitoring:

```bash
tmux attach -t pipeline              # reattach to live output
tmux kill-session -t pipeline        # stop the run
tail -f pipeline.log                 # nohup fallback: live log
```

Tested runtime: ~28–32 hours on a single 24 GB A40 GPU

### Step 3 — Full analysis suite (after training)

Recommended after retraining — cleans stale token/analysis dirs before re-encoding:

```bash
python3 helper_scripts/run_analysis.py --clean --yes
```

### Step 4 — Figures

Checkpoints are auto-discovered from their default output directories:

```bash
python tok_analysis/6_fig_summary_plot.py
```

Override checkpoint paths if needed:

```bash
python tok_analysis/6_fig_summary_plot.py \
    --encodec-ckpt  Encodec/outputs/.../best_model.pt \
    --q2d2-ckpt     Q2D2/outputs/.../Q2D2_best.ckpt \
    --hificodec-ckpt hificodec/egs/hificodec_fsd50k \
    --st-ckpt       results/speechtokenizer_fsd50k/SpeechTokenizer_best_dev.pt \
    --dac-ckpt      descript-audio-codec/ckpt/fsd50k_fsq/best
```

### Re-generate tables only (from existing outputs)

```bash
python3 helper_scripts/run_analysis.py --only-tables
```

---

## Manual step-by-step if the above steps more focused reruns in the pipeline

### Train (2 epoch test + 50-epoch run)

```bash
python helper_scripts/run_pipeline.py
```

This single command:
- Clears any old checkpoints
- Runs a 2-epoch sanity test per model (checks for OOM, NaN/Inf, assertion errors)
- Aborts with a clear message if any model fails
- Clears the test artefacts
- Launches `train_5codecs.sh` in the background via nohup

Monitor / control the background run:

```bash
tail -f training.log          # live output
ps -p $(cat training.pid)     # check still running
kill $(cat training.pid)      # stop if needed
tail -100 training.log        # last 100 lines
```

Expected runtime: ~28–32 hours on a single 24 GB GPU.

Checkpoints are written to:

| Model | Output directory |
|---|---|
| SpeechTokenizer | `results/speechtokenizer_fsd50k/` |
| Encodec | `checkpoints_multi_dataset` |
| DAC-FSQ | `descript-audio-codec/ckpt/fsd50k_fsq/` |
| HiFiCodec | `hificodec/egs/hificodec_fsd50k/` |
| Q2D2 | `Q2D2/outputs/` |

### Evaluate (full analysis suite)

```bash
python3 helper_scripts/run_analysis.py --clean --yes
```

1. **Discover** — auto-locate training checkpoints
2. **Encode** — tokenise amplitude / phase / temporal sinusoid test signals
3. **SVD(ΔZ)** — EVR metrics per perturbation type
4. **Report** — per-model combined sensitivity PDF (token flip rate, cosine sim, centroid magnitude per codebook)
5. **EGFx** — encode + score complex non-linear audio effect pairs
6. **Multi-codec PDF** — cross-codec comparison report
7. **Tables** — 8 analysis CSVs

**Outputs written to `datasets/analysis/`:**

| File | Contents |
|---|---|
| `{model}/combined_sensitivity_report_bw*.pdf` | Per-codebook sensitivity grids (one PDF per model) |
| `multi_codec_sensitivity.pdf` | Cross-codec comparison |
| `egfx_report.pdf` | EGFx non-linear effects report |
| `tables/table1_amplitude_response.csv` | TFR by dBFS level |
| `tables/table2_phase_sensitivity.csv` | TFR by phase angle (15°–360°, 0° excluded) |
| `tables/table3_temporal_offset.csv` | TFR by time offset |
| `tables/table4_centroid_magnitude.csv` | Codebook centroid magnitude stats |
| `tables/table5_svd_evr2.csv` | SVD(ΔZ) EVR by codec / perturbation type |
| `tables/table6_egfx_response.csv` | EGFx TFR / L2 / cosine sim per effect |
| `tables/table7_cross_codec_summary.csv` | 5-row cross-codec synthesis |
| `tables/table8_perplexity_by_unit.csv` | Perplexity by model/unit/condition |

> **`--clean` flag:** removes all stale `datasets/audio_tokens/` subdirs and old `datasets/analysis/` dirs (benchmark_runs, doa, dsp_*, q2d2_*, encodec_*, features, final_plots, etc.) before re-encoding. Use `--yes` to skip the interactive prompt when running on a server.


## 4 · Quick EGFx test (100 samples)

For rapid validation of all 5 codecs on guitar effects without running the full analysis suite:

```bash
bash helper_scripts/run_egfx_100samples.sh
```

This automated script:
- ✅ Verifies all 5 checkpoint paths exist before starting
- 📥 Downloads EGFx dataset (~5.8 GB, if not already present)
- 🎸 Encodes 100 effect pairs (25 × distortion, 25 × modulation, 25 × time_based, 25 × dynamics)
- 📊 Computes geometric metrics (token flip rate, cosine similarity, relative L2 distance)
- 📄 Generates comparative PDF report

**Expected runtime:** ~1 hour on a A40 GPU

**Outputs:**
- `datasets/egfx/effect_pairs_100samples.json` — sampled pairs list
- `datasets/audio_tokens/egfx_100samples/` — token embeddings per codec
- `datasets/analysis/egfx_metrics_100samples.json` — metrics
- `datasets/analysis/egfx_report_100samples.pdf` — visual report

**Use case:** Validate pipeline functionality and estimate timing for full dataset runs without requiring trained models on all sensitivity tests.


Configuration
Model	Config
SpeechTokenizer	fsd50k_cfg.json
Encodec	config_multi_dataset.yaml
DAC-FSQ	fsd50k_fsq.yml
HiFiCodec	config_24k_320d.json
Q2D2	Q2D2_fsd50k_9.8kbps_dim512_attn_b16.yaml
Shared training settings: batch_size=8 · 1010 steps/epoch · 50 epochs · num_workers=8

Dataset

datasets/fsd50k_train.csv
datasets/fsd50k_val.csv
datasets/fsd50k_test.csv

GPU requirements

24 GB of VRAM on single GPU as a minimum. 
CUDA required for training.
CPU supported for short tests.
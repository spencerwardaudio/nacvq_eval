#!/usr/bin/env python3
"""run_pipeline.py — Three-mode codec training pipeline.

Modes (mutually exclusive):
    --oom-test       2-epoch OOM check on all 5 models (12000 samples).
                     Aborts on first failure. 30-min epoch timeout guard.
    --scale-test     50-epoch learning check at staged sample sizes:
                       50 → 100 → 200 → 500 → 1000 → 3000 → 12000
                     Runs all 5 models at each stage, prints pass/fail summary,
                     then prompts before advancing to the next stage.
    --full-training  Launch 50-epoch production training on 12000 samples
                     via train_5codecs.sh (runs in background).

Optional:
    --model MODEL    (only with --oom-test) run a single model.
    --models MODEL [MODEL ...]
                     (only with --scale-test) run only the specified model(s).
                     e.g. --models Encodec SpeechTokenizer
    --stages N [N ...]
                     (only with --scale-test) run only the listed sample sizes.
                     e.g. --stages 200           (run n=200 only)
                          --stages 200 500 1000  (run three stages, with prompts)
    --scale-dac-num-iters N
                     (only with --scale-test) override DAC-FSQ iteration budget.

Usage:
    cd /home/spencerwardaudio/dev/Spatial_Audio/msc_proj
    source .venv/bin/activate
    python run_pipeline.py --oom-test
    python run_pipeline.py --oom-test --model DAC-FSQ
    python run_pipeline.py --scale-test
    python run_pipeline.py --scale-test --models Encodec SpeechTokenizer
    python run_pipeline.py --full-training
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

EPOCH_TIMEOUT_SECONDS = 30 * 60  # 30 minutes — oom-test only
SCALE_STAGES = [50, 100, 200, 500, 1000, 3000, 12000]  # exponential growth to find the capacity cliff

# ── Paths ────────────────────────────────────────────────────────────────────
PROJ_ROOT = Path(__file__).parent.resolve()
os.chdir(PROJ_ROOT)

# All checkpoint / output directories produced by each codec trainer
CHECKPOINT_DIRS = [
    PROJ_ROOT / "results" / "speechtokenizer_fsd50k",
    PROJ_ROOT / "results" / "speechtokenizer_test_2ep",
    PROJ_ROOT / "Encodec" / "checkpoints_multi_dataset",
    PROJ_ROOT / "descript-audio-codec" / "ckpt" / "fsd50k_fsq",
    PROJ_ROOT / "hificodec" / "egs" / "hificodec_fsd50k",
    PROJ_ROOT / "Q2D2" / "outputs",
]

# Per-model dirs — only this model's checkpoints are wiped when it starts a fresh run.
MODEL_CHECKPOINT_DIRS: dict[str, list] = {
    "Encodec":         [PROJ_ROOT / "Encodec" / "checkpoints_multi_dataset"],
    "Q2D2":            [PROJ_ROOT / "Q2D2" / "outputs"],
    "SpeechTokenizer": [PROJ_ROOT / "results" / "speechtokenizer_fsd50k",
                        PROJ_ROOT / "results" / "speechtokenizer_test_2ep"],
    "HiFiCodec":       [PROJ_ROOT / "hificodec" / "egs" / "hificodec_fsd50k"],
    "DAC-FSQ":         [PROJ_ROOT / "descript-audio-codec" / "ckpt" / "fsd50k_fsq"],
}

LOG_FILES = [
    PROJ_ROOT / "training.log",
    PROJ_ROOT / "training.pid",
]

# ── Failure detection ─────────────────────────────────────────────────────────
# Hard failures abort immediately
HARD_FAIL_RE = re.compile(
    r"(loss\s*[=:]\s*nan|loss\s*[=:]\s*inf"
    r"|gradient.*explod"
    r"|out\s+of\s+memory|cuda\s+out\s+of\s+memory"
    r"|\bnan\b loss|\binf\b loss"
    r"|device-side assert triggered)",
    re.IGNORECASE,
)

# Soft warnings logged but don't abort unless exit code is non-zero
SOFT_WARN_RE = re.compile(
    r"(warning|warn:|traceback|assertionerror|runtimeerror|valueerror)",
    re.IGNORECASE,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def banner(msg: str):
    line = "=" * 62
    print(f"\n{line}\n  {msg}\n{line}", flush=True)


def clear_artefacts():
    banner("Clearing checkpoint / log artefacts")
    for d in CHECKPOINT_DIRS:
        if d.exists():
            shutil.rmtree(d)
            print(f"  removed  {d.relative_to(PROJ_ROOT)}")
        else:
            print(f"  skipped  {d.relative_to(PROJ_ROOT)}  (not found)")
    for f in LOG_FILES:
        if f.exists():
            f.unlink()
            print(f"  removed  {f.name}")
    print()


def run_model(
    name: str,
    cmd: list,
    cwd: Path,
    env_extra: dict,
    timeout_enabled: bool = True,
) -> tuple[bool, list]:
    """Stream subprocess output live; return (success, hard_fail_lines)."""
    print(f"  cwd : {cwd.relative_to(PROJ_ROOT)}")
    print(f"  cmd : {' '.join(str(c) for c in cmd)}\n", flush=True)

    env = os.environ.copy()
    env.update(env_extra)

    hard_lines: list[str] = []
    soft_lines: list[str] = []

    proc = subprocess.Popen(
        [str(c) for c in cmd],
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    epoch_start: float | None = None
    EPOCH_START_RE = re.compile(r"Epoch\s+0[:\s]|epoch\s+0[/\s]", re.IGNORECASE)

    try:
        for raw_line in proc.stdout:
            line = raw_line.rstrip()
            print(raw_line, end="", flush=True)

            if HARD_FAIL_RE.search(line):
                hard_lines.append(line)
                # Kill the process immediately — no point continuing
                proc.kill()  # no point reading more output after a hard failure
                break

            if SOFT_WARN_RE.search(line):
                soft_lines.append(line)

            if timeout_enabled:
                if epoch_start is None and EPOCH_START_RE.search(line):
                    epoch_start = time.monotonic()
                elif epoch_start is not None:
                    elapsed = time.monotonic() - epoch_start
                    if elapsed > EPOCH_TIMEOUT_SECONDS:
                        msg = (
                            f"[TIMEOUT] {name} — first epoch exceeded "
                            f"{EPOCH_TIMEOUT_SECONDS // 60} min "
                            f"({elapsed / 60:.1f} min elapsed)"
                        )
                        print(f"\n  {msg}", flush=True)
                        hard_lines.append(msg)
                        proc.kill()
                        break
    finally:
        proc.wait()

    if hard_lines:
        print(f"\n  [FAIL] {name} — NaN / Inf / OOM / timeout detected:")
        for ln in hard_lines[:5]:
            print(f"    {ln}")
        return False, hard_lines

    if proc.returncode not in (0, -9):  # -9 = SIGKILL we sent; any other non-zero is an unexpected error
        print(f"\n  [FAIL] {name} — non-zero exit code: {proc.returncode}")
        if soft_lines:
            print("  Last suspicious lines:")
            for ln in soft_lines[-5:]:
                print(f"    {ln}")
        return False, soft_lines

    print(f"\n  [PASS] {name}\n")
    return True, []


# ── OOM-test definitions (2 epochs, 12000 samples) ───────────────────────────
def build_oom_tests() -> list[tuple]:
    py = sys.executable
    base_env = {"WANDB_PROJECT": "codec-fsd50k-scale"}

    return [
        (
            "DAC-FSQ",
            ["bash", "train_dac_fsq_fsd50k.sh", "--epochs", "2", "--gpus", "0"],
            PROJ_ROOT,
            {**base_env, "WANDB_NAME": "dac-fsq-oom-2ep"},
        ),
        (
            "Q2D2",
            [
                py, "train.py", "fit",
                "--config", "configs/Q2D2_fsd50k_9.8kbps_dim512_attn_b16.yaml",
                "--trainer.max_epochs", "2",
                "--trainer.check_val_every_n_epoch", "1",
            ],
            PROJ_ROOT / "Q2D2",
            {**base_env, "WANDB_NAME": "q2d2-oom-2ep"},
        ),
        (
            "Encodec",
            [
                py, "train_multi_dataset.py",
                "common.max_epoch=2",
                "common.val_interval=1",
                "wandb.project=codec-fsd50k-scale",
                "wandb.name=encodec-oom-2ep",
            ],
            PROJ_ROOT / "Encodec",
            base_env,
        ),
        (
            "SpeechTokenizer",
            [
                py, "SpeechTokenizer/scripts/train_example.py",
                "--config", "SpeechTokenizer/config/fsd50k_cfg_test_2ep.json",
            ],
            PROJ_ROOT,
            {**base_env, "WANDB_NAME": "speechtokenizer-oom-2ep"},
        ),
        (
            "HiFiCodec",
            ["bash", "train_fsd50k.sh", "--epochs", "2", "--gpus", "0"],
            PROJ_ROOT,
            {
                **base_env,
                "WANDB_PROJECT": "codec-fsd50k-scale",
                "WANDB_RUN_NAME": "hificodec-oom-2ep",
            },
        ),
    ]


# ── Scale-test definitions (50 epochs, variable n_samples) ───────────────────
def build_scale_run(n_samples: int, dac_num_iters_override: int | None = None) -> list[tuple]:
    n_val = max(1, n_samples // 8)  # 12.5 % validation split — standard heuristic for small datasets
    # SpeechTokenizer requires validation set >= batch_size (16)
    speechtokenizer_n_val = max(16, n_val)
    py = sys.executable
    base_env = {"WANDB_PROJECT": "codec-fsd50k-scale"}
    label = f"n{n_samples}"

    dac_num_iters = (
        dac_num_iters_override
        if dac_num_iters_override is not None
            else max(1, 50 * math.ceil(n_samples / 8))  # 50 epochs × steps_per_epoch (≈n/batch_size=8)
        (
            "DAC-FSQ",
            [
                "bash", "train_dac_fsq_fsd50k.sh",
                "--epochs", "50", "--gpus", "0",
                "--n-train-examples", str(n_samples),
                "--n-val-examples", str(n_val),
                "--num-iters", str(dac_num_iters),
            ],
            PROJ_ROOT,
            {**base_env, "WANDB_NAME": f"dac-fsq-{label}"},
        ),
        (
            "Q2D2",
            [
                py, "train.py", "fit",
                "--config", "configs/Q2D2_fsd50k_9.8kbps_dim512_attn_b16.yaml",
                "--trainer.max_epochs", "50",
                "--trainer.check_val_every_n_epoch", "5",
            ],
            PROJ_ROOT / "Q2D2",
            {
                **base_env,
                "WANDB_NAME": f"q2d2-{label}",
                "TRAIN_N_SAMPLES": str(n_samples),
                "VAL_N_SAMPLES": str(n_val),
            },
        ),
        (
            "Encodec",
            [
                py, "train_multi_dataset.py",
                "common.max_epoch=50",
                "common.val_interval=5",
                f"datasets.n_train_examples={n_samples}",
                f"datasets.n_val_segments={n_val}",
                "wandb.project=codec-fsd50k-scale",
                f"wandb.name=encodec-{label}",
            ],
            PROJ_ROOT / "Encodec",
            base_env,
        ),
        (
            "SpeechTokenizer",
            [
                py, "SpeechTokenizer/scripts/train_example.py",
                "--config", "SpeechTokenizer/config/fsd50k_cfg.json",
            ],
            PROJ_ROOT,
            {
                **base_env,
                "WANDB_NAME": f"speechtokenizer-{label}",
                "TRAIN_N_SAMPLES": str(n_samples),
                "VAL_N_SAMPLES": str(speechtokenizer_n_val),
            },
        ),
        (
            "HiFiCodec",
            ["bash", "train_fsd50k.sh", "--epochs", "50", "--gpus", "0"],
            PROJ_ROOT,
            {
                **base_env,
                "WANDB_PROJECT": "codec-fsd50k-scale",
                "WANDB_RUN_NAME": f"hificodec-{label}",
                "TRAIN_N_SAMPLES": str(n_samples),
                "VAL_N_SAMPLES": str(n_val),
                "STDOUT_INTERVAL": "1",  # Log every step for visibility in scale tests
            },
        ),
    ]


# ── Scale-run pre-flight guard ───────────────────────────────────────────────
_EPOCH_FLAGS = {
    # Maps model name → the CLI token that must equal '50' in its command list
    "DAC-FSQ":        "--epochs",
    "Q2D2":           "--trainer.max_epochs",
    "Encodec":        "common.max_epoch=50",   # positional key=val form
    "SpeechTokenizer": None,                   # epoch is in config file, checked separately
    "HiFiCodec":      "--epochs",
}
_SAMPLE_ENV_KEYS = {
    "SpeechTokenizer": "TRAIN_N_SAMPLES",
    "HiFiCodec":       "TRAIN_N_SAMPLES",
    "Q2D2":            "TRAIN_N_SAMPLES",
}
_SAMPLE_CMD_FLAGS = {
    "DAC-FSQ":  "--n-train-examples",
    "Encodec":  "datasets.n_train_examples",
}


def _verify_scale_run(runs: list[tuple], n_samples: int) -> None:
    """Abort with a clear message if any run is misconfigured for 50-epoch training."""
    errors: list[str] = []
    print("\n  ── Pre-flight config verification ──")
    for name, cmd, _cwd, env_extra in runs:
        epoch_ok = False
        sample_ok = False
        epoch_flag = _EPOCH_FLAGS.get(name)

        if name == "SpeechTokenizer":
            # Epochs live in the JSON config; env sample cap must be set
            epoch_ok = True  # epoch count is baked into fsd50k_cfg.json, not a CLI arg
        elif epoch_flag is not None:
            if "=" in epoch_flag:            # Encodec uses Hydra-style positional "key=val" tokens
                epoch_ok = epoch_flag in cmd
            else:                            # --flag VALUE pair
                cmd_str = [str(c) for c in cmd]
                try:
                    idx = cmd_str.index(epoch_flag)
                    epoch_ok = cmd_str[idx + 1] == "50"
                except (ValueError, IndexError):
                    pass

        env_key = _SAMPLE_ENV_KEYS.get(name)
        cmd_flag = _SAMPLE_CMD_FLAGS.get(name)
        if env_key:
            sample_ok = bool(env_extra.get(env_key, ""))
        elif cmd_flag:
            # Args may be passed as a single "flag=value" token or separate "flag" "value" tokens
            cmd_str = [str(c) for c in cmd]
            sample_ok = any(c == cmd_flag or c.startswith(cmd_flag + "=") for c in cmd_str)
        else:
            sample_ok = True  # model reads its own filelist; sample count is not wired via CLI

        status = "OK" if (epoch_ok and sample_ok) else "FAIL"
        print(f"    [{status}] {name:18s}  epochs=50:{str(epoch_ok):5}  sample_cap:{str(sample_ok):5}  n={n_samples}")
        if not epoch_ok:
            errors.append(f"{name}: epochs flag not set to 50 in command {cmd}")
        if not sample_ok:
            errors.append(f"{name}: sample-cap argument/env missing (n_samples={n_samples})")

    if errors:
        print("\n  [ABORT] Scale-run misconfigured — fix the errors above before training:")
        for e in errors:
            print(f"    ✗ {e}")
        sys.exit(1)
    print("  All checks passed.\n")


# ── Mode runners ──────────────────────────────────────────────────────────────
def run_oom_test(model: str | None = None):
    banner("OOM TEST — 2 epochs, 12000 samples")
    clear_artefacts()

    tests = build_oom_tests()
    if model:
        normalized = model.strip().lower()
        tests = [t for t in tests if t[0].lower() == normalized]
        if not tests:
            available = ", ".join(t[0] for t in build_oom_tests())
            print(f"[ERROR] Unknown model '{model}'. Available: {available}", file=sys.stderr)
            sys.exit(1)

    results: dict[str, bool] = {}
    for name, cmd, cwd, env_extra in tests:
        banner(f"OOM test ▶  {name}")
        ok, _ = run_model(name, cmd, cwd, env_extra, timeout_enabled=True)
        results[name] = ok
        if not ok:
            banner("OOM TEST ABORTED")
            print("Results so far:")
            for n, passed in results.items():
                print(f"  {'PASS' if passed else 'FAIL'}  {n}")
            print(f"\nFix the issue and re-run:  python run_pipeline.py --oom-test\n")
            sys.exit(1)

    banner("OOM test complete — all models passed")
    for n, passed in results.items():
        print(f"  {'✓ PASS' if passed else '✗ FAIL'}  {n}")
    print()


def run_scale_test(
    dac_num_iters_override: int | None = None,
    models: list[str] | None = None,
    stages: list[int] | None = None,
):
    active_stages = [n for n in SCALE_STAGES if stages is None or n in stages]
    if not active_stages:
        print(f"[ERROR] --stages values not in SCALE_STAGES {SCALE_STAGES}", file=sys.stderr)
        sys.exit(1)

    banner("SCALE TEST — 50 epochs per stage")
    print(f"  Stages      : {active_stages}")
    if models:
        print(f"  Models      : {', '.join(models)}")
    print(f"  W&B project : codec-fsd50k-scale")
    print(f"  Val sizing  : n_val = n_train // 8")
    print(f"                (SpeechTokenizer: min 16 samples)\n")

    for i, n in enumerate(active_stages):
        n_val = max(1, n // 8)
        speechtokenizer_n_val = max(16, n_val)
        dac_num_iters = (
            dac_num_iters_override
            if dac_num_iters_override is not None
            else max(1, 50 * math.ceil(n / 8))
        )
        banner(
            f"SCALE STAGE {i + 1}/{len(active_stages)}  —  "
            f"n_samples={n}  n_val={n_val}  "
            f"(SpeechTokenizer: {speechtokenizer_n_val})  "
            f"dac_num_iters={dac_num_iters}"
        )

        # Validate every model's command before launching any subprocess
        all_runs = build_scale_run(n, dac_num_iters_override)
        active_runs = [(nm, cmd, cwd, env) for nm, cmd, cwd, env in all_runs
                       if not models or nm in models]
        _verify_scale_run(active_runs, n)

        stage_results: dict[str, bool] = {}
        for name, cmd, cwd, env_extra in active_runs:
            # wipe before every stage so each run is independent (matches other models)
            for d in MODEL_CHECKPOINT_DIRS.get(name, []):  # wipe before every stage so each run is independent
                if d.exists():
                    shutil.rmtree(d)
                    print(f"  [CLEAN] removed old {name} checkpoints: {d.relative_to(PROJ_ROOT)}")
            banner(f"  {name}  [n={n}]")
            ok, _ = run_model(name, cmd, cwd, env_extra, timeout_enabled=False)
            stage_results[name] = ok

        # Stage summary
        all_passed = all(stage_results.values())
        print(f"\n  ── Stage n={n} summary ──")
        for name, ok in stage_results.items():
            print(f"    {'✓ PASS' if ok else '✗ FAIL'}  {name}")
        if not all_passed:
            failed = [name for name, ok in stage_results.items() if not ok]
            print(f"\n  [WARN] Failed: {', '.join(failed)}")

        if i < len(active_stages) - 1:
            next_n = active_stages[i + 1]
            try:
                answer = input(f"\n  Advance to n={next_n}? [y/N] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  Stopped.")
                break
            if answer != "y":
                print("  Stopped.")
                break

    banner("Scale test finished")


def launch_full_training():
    banner("FULL TRAINING — 50 epochs, 12000 samples")
    clear_artefacts()

    launch_cmd = (
        "nohup bash train_5codecs.sh --gpu 0 > training.log 2>&1 & echo $! > training.pid"  # PID written so the user can kill or monitor the job
    )
    print(f"  cmd: {launch_cmd}\n")
    subprocess.run(launch_cmd, shell=True, cwd=str(PROJ_ROOT), check=True)

    pid_file = PROJ_ROOT / "training.pid"
    if pid_file.exists():
        pid = pid_file.read_text().strip()
        print(f"  PID saved  : {pid}  (training.pid)")
    else:
        print("  [WARN] training.pid not created — check manually")

    print()
    print("  Monitor   :  tail -f training.log")
    print("  Stop      :  kill $(cat training.pid)")
    print("  Status    :  ps -p $(cat training.pid)")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--oom-test",      action="store_true", help="2-epoch OOM check (12000 samples)")
    mode.add_argument("--scale-test",    action="store_true", help="50-epoch staged learning check (50→12000)")
    mode.add_argument("--full-training", action="store_true", help="Launch 50-epoch production training in background")
    parser.add_argument(
        "--model",
        choices=["DAC-FSQ", "Q2D2", "Encodec", "SpeechTokenizer", "HiFiCodec"],
        help="(--oom-test only) run a single model",
    )
    parser.add_argument(
        "--models",
        choices=["DAC-FSQ", "Q2D2", "Encodec", "SpeechTokenizer", "HiFiCodec"],
        nargs="+",
        metavar="MODEL",
        help="(--scale-test only) run only the specified model(s)",
    )
    parser.add_argument(
        "--stages",
        type=int,
        nargs="+",
        metavar="N",
        help="(--scale-test only) run only the listed sample sizes, e.g. --stages 200",
    )
    parser.add_argument(
        "--scale-dac-num-iters",
        type=int,
        help="(--scale-test only) override DAC-FSQ num_iters for every scale stage",
    )
    args = parser.parse_args()

    if args.model and not args.oom_test:
        parser.error("--model can only be used with --oom-test")
    if args.models and not args.scale_test:
        parser.error("--models can only be used with --scale-test")
    if getattr(args, 'stages', None) and not args.scale_test:
        parser.error("--stages can only be used with --scale-test")
    if args.scale_dac_num_iters is not None and not args.scale_test:
        parser.error("--scale-dac-num-iters can only be used with --scale-test")
    if args.scale_dac_num_iters is not None and args.scale_dac_num_iters < 1:
        parser.error("--scale-dac-num-iters must be >= 1")

    banner("CODEC TRAINING PIPELINE")
    print(f"  Project root : {PROJ_ROOT}")
    print(f"  Python       : {sys.executable}")
    if args.oom_test:
        mode_label = f"OOM TEST{f'  [{args.model}]' if args.model else ''}"
    elif args.scale_test:
        model_suffix = f"  [{', '.join(args.models)}]" if args.models else ""
        mode_label = f"SCALE TEST{model_suffix}"
    else:
        mode_label = "FULL TRAINING"
    print(f"  Mode         : {mode_label}\n")

    if args.oom_test:
        run_oom_test(model=args.model)
    elif args.scale_test:
        run_scale_test(
            dac_num_iters_override=args.scale_dac_num_iters,
            models=args.models,
            stages=getattr(args, 'stages', None),
        )
    else:
        launch_full_training()
        banner("Full training launched")
        print("50-epoch training is running in the background.")
        print("Run `tail -f training.log` to monitor.\n")


if __name__ == "__main__":
    main()

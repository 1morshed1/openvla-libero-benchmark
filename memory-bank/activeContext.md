# Active Context

## Current focus
**Phase 1 — Arm A full frozen eval running on GPU-1 (started 2026-09-27 07:18 UTC).** Training done at 50k. Dev evals: 20k = 80%, 50k = **86%** → 50k selected (`~/vla/merged/arm-A-bf16` symlink → `arm-A-dev-50k-bf16`). Full eval = 50 trials × seeds 7, 42, 123, sequential (~1,500 episodes at ~100/h → done ~2026-09-27 22:00 UTC).

**Partial (2026-09-27 11:06 UTC):** seed 7 at 368/500 episodes, **87.5%** (REF seed 7 = 85.4%). Tasks 9 and 10 not yet run — both were weakest in dev, so expect the seed-7 number to fall. Top-drawer task (5) = 34/50, the clear weak spot so far.

## GPU constraint
- Train: **GPU-1** (`CUDA_VISIBLE_DEVICES=1`).
- Dev evals: **GPU-2** allowed (user-approved 2026-09-24) — `research/experiments/run_arm_a_dev_eval_gpu2_detached.sh`.
- Log the eval device per EXP; the final frozen eval should stay on one device across arms.

## Recent changes
- Arm A relaunched 2026-09-22 after host reboot; finished 50k on 2026-09-27 04:08 UTC (~7.4 s/step, ~5 days).
- Snapshots: 20k → `~/vla/merged/arm-A-dev-20k-bf16/` + `~/vla/adapters/arm-A-step20000/`; 50k → `~/vla/merged/arm-A-dev-50k-bf16/` + `~/vla/adapters/arm-A-step50000/`.
- Dev eval 20k (GPU-2) = 80/100; 50k (GPU-1) = 86/100; 50k ≥ 20k on every task. Details in `research/experiments/EXP-001.md`.
- Eval launcher made GPU/NOTE-parametrized (one log per NOTE).
- Added `research/experiments/measure.py` (five metrics + NVML energy).

## Monitor
```bash
pgrep -af run_libero_eval
ls -t ~/vla/openvla/experiments/logs/*armA-50k-final* | head -1 | xargs grep -c '^Success: True'
nvidia-smi -i 1
```
Per-seed rollout logs: `~/vla/openvla/experiments/logs/EVAL-libero_spatial-openvla-*--armA-50k-final-s{7,42,123}.txt`. EGL `eglMakeCurrent` traceback at process exit is harmless.

## Next
1. Wait for full eval (logs `research/experiments/arm_a_dev_eval_armA-50k-final-s{7,42,123}.log`) → fill EXP-001.
2. REF seeds 42, 123 (REF only has seed 7) for a like-for-like comparison.
3. `measure.py` on `arm-A-bf16` for perf+energy.
4. Arms B/C reuse the same merged dir + `measure.py` (precision only).

Eval launcher: `GPU=<n> CKPT=<dir> TRIALS=<n> SEED=<s> NOTE=<tag> bash research/experiments/run_arm_a_dev_eval_gpu2_detached.sh` (log per NOTE).

## Active decisions
- Base openvla, LoRA object of study, mujoco 3.3.2, W&B offline.
- Only 20k and 50k are available for checkpoint selection (see landmine below) — accept that for Arm A rather than retraining.
- **Energy (J/action) from Arm B onward** (and on Arm A once measuring) — paper differentiator.
- `research/experiments/measure.py` preferred over `scripts/metrics.py` for the paper table.

## Landmines
- **TFRecord rename:** folder/`dataset_info.name` → `libero_spatial_no_noops` **and** shards `libero_spatial-train.*` → `libero_spatial_no_noops-train.*`.
- **`save_latest_checkpoint_only=True` is the finetune.py default** — every 5k save overwrites the last. Pass `--save_latest_checkpoint_only False` or snapshot each save if intermediate checkpoints matter.
- **NVML vs torch index:** under `CUDA_VISIBLE_DEVICES=N`, torch `cuda:0` == NVML physical **N** (`measure.py` auto-resolves).
- Quant load path: eval harness uses `load_in_8bit`/`load_in_4bit` flags in `openvla_utils.py`; `measure.py` uses `BitsAndBytesConfig` — mirror eval for B/C success numbers.
- Detached shells skip `.bashrc` — launchers pin `HF_HOME` to the shared cache.

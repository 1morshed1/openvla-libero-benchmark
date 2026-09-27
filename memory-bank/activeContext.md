# Active Context

## Current focus
**Phase 1 — Arm A LoRA finishing (~49.5k / 50k on GPU-1, 2026-09-27).** Dev eval of the 20k checkpoint: **80%** (10 trials/task, seed 7) vs REF 85.4%. Next: dev-eval 50k, pick the better one, full frozen eval.

## GPU constraint
- Train: **GPU-1** (`CUDA_VISIBLE_DEVICES=1`).
- Dev evals: **GPU-2** allowed (user-approved 2026-09-24) — `research/experiments/run_arm_a_dev_eval_gpu2_detached.sh`.
- Log the eval device per EXP; the final frozen eval should stay on one device across arms.

## Recent changes
- Arm A relaunched 2026-09-22 after host reboot; ran uninterrupted to ~49.5k (~7.4 s/step).
- Snapshotted 20k → `~/vla/merged/arm-A-dev-20k-bf16/` + `~/vla/adapters/arm-A-step20000/`.
- Dev eval 20k on GPU-2 → **80/100**; weak tasks 5 (top drawer), 9 (next to plate), 10 (on wooden cabinet). Details in `research/experiments/EXP-001.md`.
- Added `research/experiments/measure.py` (five metrics + NVML energy).

## Monitor
```bash
tail -c 2000 ~/projects/openvla-libero-benchmark/research/experiments/arm_a_train_detached.log | tr '\r' '\n' | tail -3
nvidia-smi -i 1,2
```

## Next
1. When train hits 50k: snapshot run dir → `~/vla/merged/arm-A-dev-50k-bf16/`, dev-eval on GPU-2 (`CKPT=... NOTE=armA-50k-dev`).
2. Pick better of 20k / 50k → `~/vla/merged/arm-A-bf16`.
3. Full frozen eval (50 × seeds {7,42,123}) → fill EXP-001; `measure.py` for perf+energy.
4. Arms B/C reuse the same merged dir + `measure.py` (precision only).

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

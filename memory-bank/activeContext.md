# Active Context

## Current focus
**Phase 1 in progress — Arm A LoRA training running detached on GPU-1.** Eval protocol frozen. EXP-000 REF gate passed (85.4%).

## Session constraint (hard)
**GPU-1 only** — `CUDA_VISIBLE_DEVICES=1`.

## Recent changes (2026-09-21)
- Frozen `research/datasets/libero-spatial-eval.md` (10 tasks, 50 trials, seed 7 / multi-seed {7,42,123}).
- Downloaded `libero_spatial_no_noops` RLDS (~1.8G) → `~/vla/modified_libero_rlds/`.
- Patched `~/vla/openvla/vla-scripts/finetune.py` → `attn_implementation="sdpa"`.
- Patched dataset_info name → `libero_spatial_no_noops`.
- Launched detached train: `research/experiments/run_arm_a_train_detached.sh` (WANDB_MODE=offline, r=32, lr 5e-4, eff batch 128, max_steps 50k, save every 5k).
- Log: `research/experiments/arm_a_train_detached.log` | PID file: `arm_a_train.pid`

## Monitor
```bash
tail -f ~/projects/openvla-libero-benchmark/research/experiments/arm_a_train_detached.log
nvidia-smi -i 1
ls ~/vla/runs ~/vla/adapters
```

## Next after train
1. Eval intermediate adapters (dev trials) via `scripts/run_arm.sh`.
2. Merge best → `~/vla/merged/arm-A-bf16` on GPU-1.
3. Full Arm A eval → EXP-001 + five metrics.

## Active decisions
- Base openvla, LoRA object of study, GPU-1 only, mujoco 3.3.2, W&B offline.

- **Landmine (fixed 2026-09-21):** after renaming folder/`dataset_info.name` to `libero_spatial_no_noops`, also rename TFRecord shards `libero_spatial-train.*` → `libero_spatial_no_noops-train.*` or TFDS NotFoundError.

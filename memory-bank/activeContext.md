# Active Context

## Current focus
**Phase 1 — Arm A success result done; H1 supported.** Arm A (LoRA r=32, 50k) = **86.0% (430/500)** vs REF **85.4% (427/500)** on the same 500 initial states; exact McNemar p = 0.83. Next: `measure.py` perf/energy for Arm A, then Arms B (int8) / C (int4).

## GPU constraint (hard, user-set)
- **GPU-1 only, for everything** — train, eval, quantize, measure. `CUDA_VISIBLE_DEVICES=1`. Never GPU-0 or GPU-2.
- The 20k dev eval on GPU-2 (2026-09-24) was a one-off the user asked for; it is **not** a standing permission.
- GPU-1 is also used by the user's `gfm-slam` (MASt3R-SLAM) jobs; they are CPU-heavy and slow LIBERO evals (MuJoCo is CPU-bound). Don't touch them.

## Recent changes
- Arm A trained to 50k (finished 2026-09-27 04:08 UTC). Dev evals: 20k = 80/100, 50k = 86/100 → 50k selected (`~/vla/merged/arm-A-bf16` → `arm-A-dev-50k-bf16`).
- Full eval (50 trials/task, seed 7) → 86.0%. Per-task Arm A vs REF in `research/experiments/EXP-001.md`.
- **Seeds are deterministic:** seed 42 matched seed 7 on all 358 episodes compared. User approved stopping seeds 42/123. Protocol amended (`research/datasets/libero-spatial-eval.md`, Amendment 1): one 500-episode seed-7 run per arm; uncertainty via binomial CI + McNemar vs comparison arm.
- Eval launcher renamed → `research/experiments/run_eval_detached.sh` (defaults to GPU-1).

## Next
1. `measure.py` on `~/vla/merged/arm-A-bf16` (GPU-1) → fill perf/energy rows in EXP-001.
2. EXP-002 (int8) / EXP-003 (int4): same `arm-A-bf16` weights, precision only; one 500-episode run each; McNemar vs Arm A. Check the quant-load landmine first.

Eval launcher: `CKPT=<dir> TRIALS=<n> NOTE=<tag> setsid nohup bash research/experiments/run_eval_detached.sh &` (log `research/experiments/arm_a_dev_eval_<NOTE>.log`; rollout log under `~/vla/openvla/experiments/logs/`). ~100 episodes/h when the host is quiet.

## Active decisions
- Base openvla, LoRA object of study, mujoco 3.3.2, W&B offline.
- Arm A = 50k checkpoint (only 20k and 50k survived; see landmine).
- One seed per arm (Amendment 1); paired McNemar is the main significance test.
- **Energy (J/action)** measured for every arm from here on — paper differentiator.
- `research/experiments/measure.py` preferred over `scripts/metrics.py` for the paper table.

## Landmines
- **TFRecord rename:** folder/`dataset_info.name` → `libero_spatial_no_noops` **and** shards `libero_spatial-train.*` → `libero_spatial_no_noops-train.*`.
- **`save_latest_checkpoint_only=True` is the finetune.py default** — every 5k save overwrites the last. Pass `--save_latest_checkpoint_only False` or snapshot each save.
- **Seeds don't matter in the LIBERO harness** (fixed per-episode init states + greedy decoding).
- **NVML vs torch index:** under `CUDA_VISIBLE_DEVICES=N`, torch `cuda:0` == NVML physical **N** (`measure.py` auto-resolves).
- Quant load path: eval harness uses `load_in_8bit`/`load_in_4bit` flags in `openvla_utils.py`; `measure.py` uses `BitsAndBytesConfig` — mirror eval for B/C success numbers.
- Detached shells skip `.bashrc` — launchers pin `HF_HOME` to the shared cache.
- EGL `eglMakeCurrent` traceback at eval process exit is harmless.

# Active Context

## Current focus
**Phase 1 — Arm A success result done; H1 supported.** Arm A (LoRA r=32, 50k) = **86.0% (430/500)** vs REF **85.4% (427/500)** on the same 500 initial states; exact McNemar p = 0.83. Arm A perf/energy measured (177 ms/action, 50 J/action) — **EXP-001 complete.** Next: Arms B (int8) / C (int4).

## GPU constraint (hard, user-set)
- **GPU-1 only, for everything** — train, eval, quantize, measure. `CUDA_VISIBLE_DEVICES=1`. Never GPU-0 or GPU-2.
- The 20k dev eval on GPU-2 (2026-09-24) was a one-off the user asked for; it is **not** a standing permission.
- GPU-1 is also used by the user's `gfm-slam` (MASt3R-SLAM) jobs; they are CPU-heavy and slow LIBERO evals (MuJoCo is CPU-bound). Don't touch them.

## Recent changes
- Arm A trained to 50k (finished 2026-09-27 04:08 UTC). Dev evals: 20k = 80/100, 50k = 86/100 → 50k selected (`~/vla/merged/arm-A-bf16` → `arm-A-dev-50k-bf16`).
- Full eval (50 trials/task, seed 7) → 86.0%. Per-task Arm A vs REF in `research/experiments/EXP-001.md`.
- **Seeds are deterministic:** seed 42 matched seed 7 on all 358 episodes compared. User approved stopping seeds 42/123. Protocol amended (`research/datasets/libero-spatial-eval.md`, Amendment 1): one 500-episode seed-7 run per arm; uncertainty via binomial CI + McNemar vs comparison arm.
- Eval launcher renamed → `research/experiments/run_eval_detached.sh` (defaults to GPU-1).
- 2026-09-28: `measure.py` aligned to eval quant path, fixed (norm stats from checkpoint; sampler `_stop` shadowing `Thread._stop`), and run on Arm A → `research/results/arm-A-bf16.json`.

## Next
1. **EXP-002 (int8) full eval RUNNING** since 2026-09-28 05:55 UTC (`NOTE=armB-int8-final-s7`, `LOAD_IN_8BIT=True`); then `measure.py` int8, then EXP-003 (int4).
   Both arms: same `arm-A-bf16` weights, precision only; one 500-episode run each; McNemar vs Arm A. Quant-load path is aligned (see below).

Measure launcher: `ARM=<B> PRECISION=<int8|int4> SUCCESS_RATE=<pct> setsid nohup bash research/experiments/run_measure_when_idle.sh &` — waits for GPU-1 idle 5 min (perf/energy are whole-GPU).

Eval launcher: `CKPT=<dir> TRIALS=<n> NOTE=<tag> [LOAD_IN_8BIT=True|LOAD_IN_4BIT=True] setsid nohup bash research/experiments/run_eval_detached.sh &` (log `research/experiments/arm_a_dev_eval_<NOTE>.log`; rollout log under `~/vla/openvla/experiments/logs/`). ~100 episodes/h when the host is quiet.

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
- Quant load path: eval harness passes bare `load_in_8bit`/`load_in_4bit` + `torch_dtype=bf16` in `openvla_utils.py`. Under transformers 4.40.1 int4 resolves to **fp4 / no double-quant / fp32 compute** (not nf4). `measure.py` mirrors this exactly (fixed 2026-09-28). Arm C = stock OpenVLA fp4 — do not patch the external clone to nf4.
- **accelerate must stay 0.30.1** — 1.15 broke bnb int8/int4 loading under transformers 4.40.1 (`.to is not supported for 4-bit or 8-bit`).
- Detached shells skip `.bashrc` — launchers pin `HF_HOME` to the shared cache.
- EGL `eglMakeCurrent` traceback at eval process exit is harmless.

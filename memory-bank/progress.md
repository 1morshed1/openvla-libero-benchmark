# Progress

## Status
**Phase 1 — Arm A success result done; H1 supported.** Arm A **86.0%** (430/500) vs REF **85.4%** (427/500), McNemar p = 0.83. Perf/energy measurement next.

## What works
- [x] Phase 0 env + REF full **85.4%** (EXP-000).
- [x] Eval protocol frozen (`research/datasets/libero-spatial-eval.md`) + Amendment 1 (seeds deterministic → one seed-7 run per arm).
- [x] Training data `libero_spatial_no_noops` on disk (shards renamed).
- [x] Finetune SDPA patch; detached train + eval launchers (GPU-1).
- [x] Arm A LoRA trained to 50k.
- [x] Dev evals: 20k = 80/100, 50k = 86/100 → 50k selected as `arm-A-bf16`.
- [x] EXP-001 full eval: 86.0% (95% CI 83.0–89.0).
- [x] Five-metric (+energy) harness: `research/experiments/measure.py`.

## What's left
- [ ] `measure.py` on Arm A → EXP-001 perf/energy rows
- [ ] Arms B (int8) / C (int4) (+ D/E stretch)
- [ ] Results table, figures, paper, public README

## Known issues / risks
- Do not reinstall torch 2.2 / mujoco≥3.4 / flash-attn.
- Host reboots kill detached jobs.
- `save_latest_checkpoint_only=True` default lost 25k–45k checkpoints.
- CPU-heavy jobs on the host (e.g. gfm-slam) slow LIBERO evals.
- W&B offline (no cloud login); `wandb sync` later for curves.
- ~~`measure.py` quant path vs `openvla_utils.py` flags~~ — aligned 2026-09-28 (int4 = fp4 / no double-quant / fp32 compute; non-quantized layers bf16).
- Stretch (after A/B/C): nf4 + double-quant as a separate arm beside D (GPTQ/AWQ) — not a replacement for C.

## Phase map
| Phase | Goal | State |
|-------|------|-------|
| 0 | Env + REF EXP-000 | **DONE** |
| 1 | Freeze harness + Arm A | Success result **DONE** (86.0%); perf/energy pending |
| 2 | Quantize B/C(/D) | Not started |
| 3 | Write-up + figures | Not started |
| 4 | Jetson E | Stretch |
| 5 | Ship repo + paper | Not started |

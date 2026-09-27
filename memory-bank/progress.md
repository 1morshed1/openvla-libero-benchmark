# Progress

## Status
**Phase 1 — Arm A trained (50k); full frozen eval running.** Dev: 20k = 80%, 50k = **86%** (REF 85.4%). 50k selected. Full eval partial: seed 7 at 368/500 = **87.5%** (2026-09-27 11:06 UTC); seeds 42/123 queued.

## What works
- [x] Phase 0 env + REF full **85.4%** (EXP-000).
- [x] Eval protocol frozen (`research/datasets/libero-spatial-eval.md`).
- [x] Training data `libero_spatial_no_noops` on disk (shards renamed).
- [x] Finetune SDPA patch; detached train + eval launchers (GPU-1).
- [x] Arm A LoRA trained to ~50k (relaunch after reboot).
- [x] Dev eval @20k: 80/100.
- [x] Five-metric (+energy) harness: `research/experiments/measure.py`.
- [x] Dev eval @50k: 86/100 → selected as `arm-A-bf16`.
- [ ] EXP-001 full eval (running).

## What's left
- [ ] Full eval 50 × {7,42,123} → EXP-001; REF seeds 42/123
- [ ] `measure.py` five metrics (+ energy)
- [ ] Arms B/C (+ D/E stretch)
- [ ] Results table, figures, paper, public README

## Known issues / risks
- Do not reinstall torch 2.2 / mujoco≥3.4 / flash-attn.
- Host reboots kill detached trains.
- `save_latest_checkpoint_only=True` default lost 25k–45k checkpoints.
- W&B offline (no cloud login); `wandb sync` later for curves.
- `measure.py` quant path vs `openvla_utils.py` flags — align before Arm B/C.

## Phase map
| Phase | Goal | State |
|-------|------|-------|
| 0 | Env + REF EXP-000 | **DONE** |
| 1 | Freeze harness + Arm A | Trained; 50k selected (dev 86%); full eval running |
| 2 | Quantize B/C(/D) | Not started |
| 3 | Write-up + figures | Not started |
| 4 | Jetson E | Stretch |
| 5 | Ship repo + paper | Not started |

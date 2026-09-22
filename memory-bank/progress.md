# Progress

## Status
**Phase 1 — Arm A LoRA relaunched** (after host reboot). Pre-reboot 5k adapter backed up. Protocol frozen. EXP-000 PASS (85.4%).

## What works
- [x] Phase 0 env + REF full **85.4%** (EXP-000).
- [x] Eval protocol frozen (`research/datasets/libero-spatial-eval.md`).
- [x] Training data `libero_spatial_no_noops` on disk (shards renamed).
- [x] Finetune SDPA patch; Arm A detached launcher.
- [x] First LoRA ckpt @ **5k** (`~/vla/adapters/...`).
- [x] Five-metric (+energy) harness: `research/experiments/measure.py`.
- [ ] Arm A train to useful best ckpt (50k or early stop).
- [ ] Merge + EXP-001 eval.

## What's left
- [ ] Relaunch Arm A after reboot; collect more 5k-interval adapters
- [ ] Dev-eval intermediates → pick best; early-stop if plateau
- [ ] Merge bf16 → EXP-001 five metrics (+ energy)
- [ ] Arms B/C (+ D/E stretch)
- [ ] Results table, figures, paper, public README

## Known issues / risks
- Do not reinstall torch 2.2 / mujoco≥3.4 / flash-attn.
- Host reboots kill detached trains — expect to resume/relaunch; keep `save_steps=5000`.
- Shared GPU-1 contention — use detached launchers.
- W&B offline (no cloud login); `wandb sync` later for curves.
- `measure.py` quant path vs `openvla_utils.py` flags — align before Arm B/C.

## Phase map
| Phase | Goal | State |
|-------|------|-------|
| 0 | Env + REF EXP-000 | **DONE** |
| 1 | Freeze harness + Arm A | Protocol frozen; **train relaunched 2026-09-22** (pre-reboot 5k backed up) |
| 2 | Quantize B/C(/D) | Not started |
| 3 | Write-up + figures | Not started |
| 4 | Jetson E | Stretch |
| 5 | Ship repo + paper | Not started |

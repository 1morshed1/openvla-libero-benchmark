# Progress

## Status
**Phase 1 — Arm A LoRA training underway (detached, GPU-1).** Phase 0 REF gate PASS (85.4%). Eval protocol frozen.

## What works
- [x] Phase 0 env + REF full **85.4%** (EXP-000).
- [x] Eval protocol frozen (`research/datasets/libero-spatial-eval.md`).
- [x] Training data `libero_spatial_no_noops` on disk.
- [x] Finetune SDPA patch; Arm A detached launcher.
- [ ] Arm A train to completion (50k steps / early stop on best ckpt).
- [ ] Merge + EXP-001 eval.

## What's left
- [ ] Finish Arm A train; pick best adapter; merge bf16
- [ ] EXP-001 five metrics
- [ ] Arms B/C (+ D/E stretch)
- [ ] Results table, figures, paper, public README

## Known issues / risks
- Do not reinstall torch 2.2 / mujoco≥3.4 / flash-attn.
- protobuf vs TF warnings — OK for now.
- Shared GPU-1 contention — use detached launchers.
- W&B offline (no cloud login).

## Phase map
| Phase | Goal | State |
|-------|------|-------|
| 0 | Env + REF EXP-000 | **DONE** |
| 1 | Freeze harness + Arm A | Protocol frozen; **train running** |
| 2 | Quantize B/C(/D) | Not started |
| 3 | Write-up + figures | Not started |
| 4 | Jetson E | Stretch |
| 5 | Ship repo + paper | Not started |

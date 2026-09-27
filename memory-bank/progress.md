# Progress

## Status
**Phase 1 — Arm A train ~49.5k / 50k.** Dev eval @20k = **80%** (REF 85.4%). Protocol frozen. EXP-000 PASS.

## What works
- [x] Phase 0 env + REF full **85.4%** (EXP-000).
- [x] Eval protocol frozen (`research/datasets/libero-spatial-eval.md`).
- [x] Training data `libero_spatial_no_noops` on disk (shards renamed).
- [x] Finetune SDPA patch; detached train + GPU-2 dev-eval launchers.
- [x] Arm A LoRA trained to ~50k (relaunch after reboot).
- [x] Dev eval @20k: 80/100.
- [x] Five-metric (+energy) harness: `research/experiments/measure.py`.
- [ ] Dev eval @50k.
- [ ] Merge best + EXP-001 full eval.

## What's left
- [ ] Dev-eval 50k → pick 20k vs 50k
- [ ] Merge bf16 → EXP-001 five metrics (+ energy)
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
| 1 | Freeze harness + Arm A | Train ~done; dev @20k = 80%; 50k dev + full eval pending |
| 2 | Quantize B/C(/D) | Not started |
| 3 | Write-up + figures | Not started |
| 4 | Jetson E | Stretch |
| 5 | Ship repo + paper | Not started |

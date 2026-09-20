# Progress

## Status
**Phase 0 — REF smoke PASS.** Env + installs + MuJoCo EGL + released-ckpt smoke on GPU-1 done (**90%** @ 3 trials/task). Full 50-trial REF / formal ~84.7% gate and eval-protocol freeze still open.

## What works
- [x] Project plan, research scaffold, paper notes, scripts, memory bank (prior).
- [x] Office: Miniconda + `openvla` env; torch 2.11.0+cu128; gate2 on **GPU-1**.
- [x] Office: `~/vla/{openvla,openvla-oft,LIBERO}`; SDPA + Hub stats patches.
- [x] openvla `--no-deps` + curated deps; LIBERO + robosuite; mujoco **3.3.2**.
- [x] MuJoCo EGL gate5 + LIBERO env reset.
- [x] REF smoke EXP-000: 27/30 = **90%** on `libero_spatial` (center_crop True).
- [x] Rig notes: `research/hardware-office-vm-130-131.md`.
- [x] Disk expanded (~2.0 TiB).

## What's left
- [ ] Full REF: 50 trials/task → confirm ~84.7%; five metrics via `scripts/metrics.py`
- [ ] Freeze `research/datasets/libero-spatial-eval.md`
- [ ] Arm A LoRA train + merge + EXP-001
- [ ] Arms B/C (+ D/E stretch)
- [ ] Results table, figures, paper, public README

## Known issues / risks
- Do not let pip reinstall torch 2.2 / mujoco≥3.4 / flash-attn.
- protobuf 6.31.1 vs TF 2.15 pin conflict — OK for eval import path; watch Arm A TF data pipeline.
- EGL cleanup exceptions on env close — ignore if rollouts succeeded.
- Merge LoRA on GPU-1 (same as eval).

## Phase map
| Phase | Goal | State |
|-------|------|-------|
| 0 | Env + REF EXP-000 + scaffold | Smoke PASS; full REF pending |
| 1 | Freeze harness + Arm A | Not started |
| 2 | Quantize B/C(/D) | Not started |
| 3 | Write-up + figures | Not started |
| 4 | Jetson E | Stretch |
| 5 | Ship repo + paper | Not started |

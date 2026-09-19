# Progress

## Status
**Phase 0 — off-hours prep done, GPU work pending.** Scaffold + plan + all paper notes committed and pushed to `origin/main`. No EXP logs, no env verification, no model runs yet.

## What works
- [x] Project plan written (`plan/VLA-project-plan.md`).
- [x] Research README with RQ, hypotheses, metrics, matrix (`research/README.md`).
- [x] Research dirs + scaffold files, committed + pushed (`ea20a55`+).
- [x] Paper notes: OpenVLA, OpenVLA-OFT, Octo, AutoVLA (off-road), Zhang apple-harvest — grounded, with verify-from-body flags.
- [x] Frozen eval-protocol card exists (`research/datasets/libero-spatial-eval.md`) — values not yet frozen.
- [x] EXP template + weekly-log W38.
- [x] Root `.gitignore` (results JSON + figures kept tracked).
- [x] Memory bank initialized + updated 2026-09-19.
- [x] Phase-0 bring-up script `scripts/phase0_bringup.sh` (gated env setup) committed + pushed.
- [x] Metrics harness `scripts/metrics.py` (5 metrics §7) built; syntax/argparse verified on home box.
- [x] Arm launch scripts: `scripts/train_arm_a.sh` (LoRA train + merge), `scripts/run_arm.sh` (eval + metrics → JSON row).
- [x] Commit history scrubbed of AI attribution (force-pushed).

## What's left (checklist from plan)
- [ ] Run bring-up script on office box: env + torch `(12,0)` + clones + MuJoCo EGL
- [ ] EXP-000: released checkpoint ~84.7% in sim → harness validated
- [ ] Freeze eval-protocol values (tasks/seeds) in the card and never change
- [x] Metrics harness (5 metrics) built (`scripts/metrics.py`) — reuse/verify on box
- [ ] Arm A: LoRA fine-tune, merge bf16, EXP-001 (scripts ready: `train_arm_a.sh`/`run_arm.sh`)
- [ ] Arm B int8 + Arm C int4 (EXP-002/003)
- [ ] Results table A/B/C + degradation
- [ ] Arm D GPTQ/AWQ (stretch), Arm E Jetson (stretch)
- [ ] Degradation figures + demo video
- [ ] Public README + paper draft
- [ ] Weekly-log habit started

## Known issues / risks
- Blackwell vs pinned torch/flash-attn — #1 setup risk.
- MuJoCo headless classic time sink — solve before model debug.
- Device-dependent LoRA merge — merge on eval GPU.
- Forgetting `--center_crop True` silently kills success.

## Phase map
| Phase | Goal | State |
|-------|------|-------|
| 0 | Env + REF EXP-000 + scaffold + core papers | Scaffold + papers + bring-up script done; run script + EXP-000 pending |
| 1 | Freeze harness + Arm A | Not started |
| 2 | Quantize B/C(/D), fill table | Not started |
| 3 | Write-up + figures + sim demo | Not started |
| 4 | Jetson E | Stretch |
| 5 | Ship repo + paper | Not started |

# Progress

## Status
**Phase 0 — early.** Research notebook scaffold + plan committed locally; no EXP logs, no env verification, no model runs recorded in-repo.

## What works
- [x] Project plan written (`plan/VLA-project-plan.md`).
- [x] Research README with RQ, hypotheses, metrics, matrix (`research/README.md`).
- [x] Empty research dirs: papers, experiments, notes, datasets, baselines, results, figures, weekly-log.
- [x] Memory bank initialized.

## What's left (checklist from plan)
- [ ] EXP-000: released checkpoint ~84.7% in sim → harness validated
- [ ] Paper notes: OpenVLA, OpenVLA-OFT (+ Octo, off-road VLA, Zhang apple-harvest)
- [ ] Frozen eval protocol doc (`research/datasets/libero-spatial-eval.md`)
- [ ] Metrics harness (5 metrics) built + reused
- [ ] Arm A: LoRA fine-tune, merge bf16, EXP-001
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
| 0 | Env + REF EXP-000 + scaffold + core papers | Scaffold only |
| 1 | Freeze harness + Arm A | Not started |
| 2 | Quantize B/C(/D), fill table | Not started |
| 3 | Write-up + figures + sim demo | Not started |
| 4 | Jetson E | Stretch |
| 5 | Ship repo + paper | Not started |

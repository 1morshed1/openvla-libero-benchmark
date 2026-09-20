# Progress

## Status
**Phase 0 — office bring-up ready to resume.** Env + torch gate2 + clones + SDPA patches done on GPU-1. Disk expanded (~2.0 TiB / ~1.5 TiB free). Still TODO: openvla `--no-deps` install, LIBERO, MuJoCo EGL, REF/EXP-000. Rig notes: `research/hardware-office-vm-130-131.md`.

## What works
- [x] Project plan written (`plan/VLA-project-plan.md`).
- [x] Research README with RQ, hypotheses, metrics, matrix (`research/README.md`).
- [x] Research dirs + scaffold files, committed + pushed (`ea20a55`+).
- [x] Paper notes: OpenVLA, OpenVLA-OFT, Octo, AutoVLA (off-road), Zhang apple-harvest — grounded, with verify-from-body flags.
- [x] Frozen eval-protocol card exists (`research/datasets/libero-spatial-eval.md`) — values not yet frozen.
- [x] EXP template + weekly-log W38.
- [x] Root `.gitignore` (results JSON + figures kept tracked).
- [x] Memory bank initialized + updated 2026-09-19; office pause update 2026-09-20.
- [x] Phase-0 bring-up script `scripts/phase0_bringup.sh` (gated env setup) committed + pushed.
- [x] Metrics harness `scripts/metrics.py` (5 metrics §7) built; syntax/argparse verified on home box.
- [x] Arm launch scripts: `scripts/train_arm_a.sh` (LoRA train + merge), `scripts/run_arm.sh` (eval + metrics → JSON row).
- [x] Commit history scrubbed of AI attribution (force-pushed).
- [x] Office: Miniconda + `openvla` env; torch 2.11.0+cu128; gate2 `(12,0)`+matmul on **GPU-1**.
- [x] Office: `~/vla/{openvla,openvla-oft,LIBERO}` cloned; SDPA patches in openvla tree.

## What's left (checklist from plan)
- [x] Free disk (root expanded to ~2.0 TiB, 2026-09-20)
- [ ] Finish openvla `--no-deps` + curated deps (do **not** let pip install torch 2.2)
- [ ] install_libero + libero_requirements
- [ ] MuJoCo EGL gate5
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
- Disk was ~97% full morning of 2026-09-20; **expanded to ~2.0 TiB** same day — no longer blocking.
- Blind `pip install -e openvla` clobbers torch with 2.2.0 — always `--no-deps` then re-verify cu128.
- Blackwell vs flash-attn — use SDPA only (patches applied; keep them).
- MuJoCo headless classic time sink — solve before model debug.
- Device-dependent LoRA merge — merge on eval GPU (**GPU-1**).
- Forgetting `--center_crop True` silently kills success.

## Phase map
| Phase | Goal | State |
|-------|------|-------|
| 0 | Env + REF EXP-000 + scaffold + core papers | Env/torch/clones/SDPA done; disk OK; install+MuJoCo+EXP-000 next |
| 1 | Freeze harness + Arm A | Not started |
| 2 | Quantize B/C(/D), fill table | Not started |
| 3 | Write-up + figures + sim demo | Not started |
| 4 | Jetson E | Stretch |
| 5 | Ship repo + paper | Not started |

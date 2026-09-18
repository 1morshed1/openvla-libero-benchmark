# Active Context

## Current focus
Phase 0 — off-hours prep done on home box; next real work is env bring-up on the office GPU. Full execution plan in `plan/VLA-project-plan.md`.

## Recent changes (2026-09-18)
- `research/` scaffold committed + pushed to `origin/main` (initial commit `ea20a55`).
- Paper notes **written for all five reading-list papers** (grounded vs arXiv): `openvla.md`, `openvla-oft.md`, `octo.md`, `offroad-vla.md` (AutoVLA chosen over ORAD-3D), `zhang-apple-harvest.md`.
- Frozen eval-protocol **card** created: `research/datasets/libero-spatial-eval.md` (values still TODO — freeze on office box).
- `research/experiments/EXP-template.md`, `research/weekly-log/2026-W38.md` added.
- Root `.gitignore` added (weights/datasets/wandb/video ignored; `research/results/*.json` + `research/figures/*` kept tracked); redundant `research/.gitignore` removed.
- Hardware/schedule reality confirmed with user (see techContext): office box persistent, dedicated, **jobs may run unattended overnight/weekend**; home 2060 too small for 7B.

## Next steps (Phase 0, office box)
1. Confirm office box: `nvidia-smi`, `nvcc` ≥12.8, Python 3.10.
2. Conda env `openvla` + Blackwell torch cu128+; verify capability `(12, 0)`.
3. Clone openvla + openvla-oft (read) + LIBERO; install openvla with relaxed torch pin, SDPA, no flash-attn; recent bitsandbytes.
4. MuJoCo EGL smoke test (dummy env render) before touching model.
5. Tiny then full REF eval → log **EXP-000** (~84.7% gate).
6. Fill + freeze `research/datasets/libero-spatial-eval.md` exact values (tasks, seeds).

## Off-hours (home box) still open
- Verify body-only facts flagged in paper notes once PDFs read (Octo param counts, per-suite LIBERO tables, AutoVLA backbone).

## Active decisions
- Run base openvla, not OFT fork path.
- LoRA is object of study despite 96 GB card.
- Sim A–C table = complete paper; Jetson optional.
- Launch long fine-tune before leaving (Thu ~5pm) to use the unattended Fri/Sat gap; still `--save_steps 5000` for crash safety.

## Open questions / blockers
- Office env not yet verified (GPU/CUDA state unknown from repo alone).
- External `~/vla` clones don't exist yet on office box.
- Issue #1 (RQ) not yet opened on GitHub — optional, needs `gh` auth.

## Considerations
Favor correctness + logging over speed. No deadline. Cadence inverted vs plan §10: heavy GPU Sun–Thu, writing evenings/home, long trains bridge Fri–Sat.

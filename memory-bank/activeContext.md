# Active Context

## Current focus
Phase 0 — off-hours prep done on home box; next real work is env bring-up on the office GPU. Bring-up now scripted (`scripts/phase0_bringup.sh`) — office box just clones repo + sources it. Full execution plan in `plan/VLA-project-plan.md`.

## Recent changes (2026-09-19)
- **`scripts/phase0_bringup.sh` added** — function-per-section, gated (not blind): gate0 hardware, conda env, torch cu128+ with `(12,0)`+matmul assert, external clones, openvla install (re-pin transformers 4.40.1/tokenizers/timm, SDPA, no flash-attn, recent bitsandbytes, verify torch not clobbered), MuJoCo EGL render smoke, REF smoke→EXP-000. Committed + pushed.
- **`scripts/metrics.py` added** — reusable 5-metric harness (§7): loads bf16/int8/int4, warmup 10 + 100-iter median latency, peak mem (torch + nvidia-smi), du size, throughput, env provenance; writes/merges `research/results/<arm>.json`. GPU code guarded (imports on home box; py_compile + --help pass).
- **`scripts/train_arm_a.sh` added** — `train` (torchrun LoRA r=32, lr 5e-4, eff batch 128, image_aug, save_steps 5000) + `merge` (PEFT merge_and_unload → bf16, on eval GPU).
- **`scripts/run_arm.sh` added** — per-arm orchestrator: LIBERO eval (center_crop, seeds) → parse success → metrics.py → JSON row; dev/final modes.
- Git history rewritten to strip all `Co-Authored-By: Claude` trailers (force-pushed `origin/main`; new hashes). No AI attribution in commits going forward per user.

## Verify-on-box caveats (baked as script comments, can't confirm from home)
- openvla `finetune.py` flag names + `predict_action`/prompt format mirror the repo — confirm first run.
- `run_libero_eval.py` success-rate stdout parse is best-effort → `SUCCESS=<pct>` manual override in `run_arm.sh`.
- **Prereq before train:** patch `attn_implementation="flash_attention_2"` → `"sdpa"` in openvla `finetune.py` + `openvla_utils.py` (landmine #1/#3).

## Recent changes (2026-09-18)
- `research/` scaffold committed + pushed to `origin/main` (initial commit `ea20a55`).
- Paper notes **written for all five reading-list papers** (grounded vs arXiv): `openvla.md`, `openvla-oft.md`, `octo.md`, `offroad-vla.md` (AutoVLA chosen over ORAD-3D), `zhang-apple-harvest.md`.
- Frozen eval-protocol **card** created: `research/datasets/libero-spatial-eval.md` (values still TODO — freeze on office box).
- `research/experiments/EXP-template.md`, `research/weekly-log/2026-W38.md` added.
- Root `.gitignore` added (weights/datasets/wandb/video ignored; `research/results/*.json` + `research/figures/*` kept tracked); redundant `research/.gitignore` removed.
- Hardware/schedule reality confirmed with user (see techContext): office box persistent, dedicated, **jobs may run unattended overnight/weekend**; home 2060 too small for 7B.

## Next steps (Phase 0, office box)
Run `scripts/phase0_bringup.sh` section by section (stop at each GATE):
1. `gate0_check` — `nvidia-smi`, `nvcc` ≥12.8, driver recent.
2. `make_env` (Python 3.10) → `conda activate openvla` → `install_torch; gate2_check` (must assert `(12,0)`+matmul).
3. `clone_repos` (openvla run path, openvla-oft read, LIBERO) → `install_openvla` (verify torch survived) → `install_libero`.
4. `gate5_mujoco` — EGL render must pass before touching model.
5. `ref_smoke` (3 trials) then full REF eval → log **EXP-000** (~84.7% gate).
6. Fill + freeze `research/datasets/libero-spatial-eval.md` exact values (tasks, seeds).
- **Not in script:** HF auth (`huggingface-cli login`/`HF_TOKEN`) + disk for `openvla-7b-finetuned-libero-spatial` (~15GB) — REF eval pulls it at runtime.

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

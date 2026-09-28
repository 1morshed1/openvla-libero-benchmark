# Frozen eval protocol — LIBERO-Spatial

**This is the scientific control. Once filled and committed, DO NOT change it — every arm is measured against these exact settings.**

## Protocol (frozen 2026-09-21)

- **Task suite:** `libero_spatial` — all **10** tasks (no subset):
  1. pick up the black bowl between the plate and the ramekin and place it on the plate
  2. pick up the black bowl next to the ramekin and place it on the plate
  3. pick up the black bowl from table center and place it on the plate
  4. pick up the black bowl on the cookie box and place it on the plate
  5. pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate
  6. pick up the black bowl on the ramekin and place it on the plate
  7. pick up the black bowl next to the cookie box and place it on the plate
  8. pick up the black bowl on the stove and place it on the plate
  9. pick up the black bowl next to the plate and place it on the plate
  10. pick up the black bowl on the wooden cabinet and place it on the plate
- **Trials per task:** **50** for final / paper numbers (500 episodes). Dev may use 3–10.
- **Seeds:** default harness seed **`7`** (used for EXP-000 full REF). For multi-seed finals: **`{7, 42, 123}`** (3 seeds) — run when reporting ± across seeds.
- **`--center_crop`:** `True` (mandatory).
- **`unnorm_key`:**
  - REF (released HF ckpt): `libero_spatial` (key present in that checkpoint’s `dataset_statistics.json`)
  - Arms A/B/C (our LoRA on `libero_spatial_no_noops`): `libero_spatial_no_noops`
- **Eval device:** physical **GPU-1** only (`CUDA_VISIBLE_DEVICES=1`), RTX PRO 6000 Blackwell ~96GB. Log per EXP: driver, CUDA, torch, bitsandbytes, mujoco.
- **Attention impl:** SDPA (no flash-attn).
- **Repo:** base `openvla/openvla` (not the OFT fork).
- **MuJoCo:** `3.3.2` + `MUJOCO_GL=egl`.

## Provenance

- **Fine-tuning data:** `openvla/modified_libero_rlds` → `libero_spatial_no_noops` under `~/vla/modified_libero_rlds/`.
- **Reference checkpoint (REF):** `openvla/openvla-7b-finetuned-libero-spatial`.
- **EXP-000 gate:** **85.4%** (427/500) @ 50 trials/task, seed 7 — within published ~84.7% ± 0.9%.

## Sign-off

- [x] Protocol values filled and committed. Frozen on: **2026-09-21**.

## Amendment 1 (2026-09-28) — seeds are deterministic

The multi-seed rule above does not produce variance. `run_libero_eval.py` loads a fixed initial state per episode index (`task_suite.get_task_init_states`) and decodes greedily (`do_sample=False`), so the seed changes nothing: Arm A seed 42 matched seed 7 on all 358 episodes compared (0 differ). Effective protocol from now on:

- **One run per arm: 50 trials/task, seed 7** (500 episodes). Same 500 initial states for every arm, so arms are directly paired.
- **Uncertainty:** binomial 95% CI on the 500 episodes (±~3 pts at 85%), and **exact McNemar** on per-episode outcomes vs the comparison arm.
- All other settings unchanged. Approved by the user 2026-09-28.

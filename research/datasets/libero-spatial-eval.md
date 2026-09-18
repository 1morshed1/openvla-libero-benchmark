# Frozen eval protocol — LIBERO-Spatial

**This is the scientific control. Once filled and committed, DO NOT change it — every arm is measured against these exact settings.**

## Protocol (fill exact values on office box, then freeze)

- **Task suite:** `libero_spatial` (10 tasks). Subset used? — NO by default. If subsetting, list exact task names here: TODO.
- **Trials per task:** **50** for final numbers. 10–20 during dev for speed.
- **Seeds:** **3** for final numbers (list exact seeds: TODO). One seed during dev.
- **`--center_crop`:** `True` (mandatory — checkpoints trained with random-crop aug; omitting quietly tanks success).
- **`unnorm_key`:** `libero_spatial_no_noops`.
- **Eval device:** RTX PRO 6000 96GB. Record per run: driver version, CUDA version, torch version, bitsandbytes version.
- **Attention impl:** SDPA (flash-attn skipped on Blackwell).
- **Repo:** base `openvla/openvla` (not the OFT fork).

## Provenance

- **Fine-tuning data:** `openvla/modified_libero_rlds` (HF datasets) → `libero_spatial_no_noops`. ~10 GB for all four suites.
- **Reference checkpoint (REF arm):** `openvla/openvla-7b-finetuned-libero-spatial`. Anchor ≈ 84.7% ± 0.9%.

## Sign-off

- [ ] Protocol values filled and committed. Frozen on: ____

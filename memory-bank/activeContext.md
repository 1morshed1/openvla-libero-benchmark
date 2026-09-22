# Active Context

## Current focus
**Phase 1 — Arm A LoRA relaunched on GPU-1** (2026-09-22 after host reboot). Pre-reboot 5k adapter backed up. Eval protocol frozen. EXP-000 REF gate passed (85.4%).

Claude-update package reviewed (`claude-update/`): early-stop + energy column guidance accepted; `measure.py` + `EXP-001.md` installed under `research/experiments/`.

## Session constraint (hard)
**GPU-1 only** — `CUDA_VISIBLE_DEVICES=1`.

## Recent changes
- Frozen `research/datasets/libero-spatial-eval.md` (10 tasks, 50 trials, seed 7 / multi-seed {7,42,123}).
- Downloaded + fixed RLDS `libero_spatial_no_noops` (folder, `dataset_info.name`, **and** TFRecord shard names).
- Patched `~/vla/openvla/vla-scripts/finetune.py` → `attn_implementation="sdpa"`.
- Arm A train launched detached 2026-09-21; reached ~step 9006 (~18%); **host reboot 22:30 UTC killed job**.
- Surviving artifact: adapter @ step 5000 under `~/vla/adapters/...` (+ run-dir save ~15G).
- Added `research/experiments/measure.py` (latency/mem/size/throughput/**energy via NVML**) and pre-filled `EXP-001.md`.

## Monitor (after relaunch)
```bash
tail -f ~/projects/openvla-libero-benchmark/research/experiments/arm_a_train_detached.log
nvidia-smi -i 1
ls ~/vla/runs ~/vla/adapters
```

## Next
1. Relaunch Arm A on GPU-1 (preserve 5k adapter; prefer continuing for 10k/15k/… saves).
2. Dev-eval intermediates (10 trials, seed 7) → early-stop if ~84–85% flat.
3. Merge best → `~/vla/merged/arm-A-bf16` on GPU-1.
4. Full Arm A eval → fill EXP-001; `measure.py` for perf+energy.
5. Arms B/C reuse same merged dir + `measure.py` (precision only).

## Active decisions
- Base openvla, LoRA object of study, GPU-1 only, mujoco 3.3.2, W&B offline.
- **Early-stop OK** — don’t grind all 50k if success plateaus.
- **Energy (J/action) from Arm B onward** (and on Arm A once measuring) — paper differentiator.
- `scripts/metrics.py` = original 5-metric harness; `research/experiments/measure.py` = same + NVML energy (prefer for paper table).

## Landmines
- **TFRecord rename:** folder/`dataset_info.name` → `libero_spatial_no_noops` **and** shards `libero_spatial-train.*` → `libero_spatial_no_noops-train.*`.
- **NVML vs torch index:** under `CUDA_VISIBLE_DEVICES=1`, torch `cuda:0` == NVML physical **1** (`measure.py` auto-resolves).
- Quant load path: eval harness uses `load_in_8bit`/`load_in_4bit` flags in `openvla_utils.py`; `measure.py` uses `BitsAndBytesConfig` — mirror eval for B/C success numbers.

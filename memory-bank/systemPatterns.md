# System Patterns

## Architecture (intended)
```
external: openvla/openvla + LIBERO + modified_libero_rlds
     ↓
eval/train scripts (Blackwell-adapted: torch cu128+, SDPA, no flash-attn)
     ↓
checkpoints: REF (HF) | Arm A merged bf16 | B/C load-time / saved quant
     ↓
metrics harness → research/results/<arm>.json
     ↓
research/experiments/EXP-*.md + figures + paper
```

## Experiment matrix pattern
Hold **weights fixed** after Arm A; vary **only precision**. REF = harness check, not paper row.

| Arm | Precision | Role |
|-----|-----------|------|
| REF | bf16 released | Harness gate |
| A | bf16 merged LoRA | Full-precision baseline |
| B | int8 (bnb) | Cheap PTQ |
| C | int4/NF4 (bnb) | Aggressive PTQ |
| D | GPTQ/AWQ 4-bit | Stretch deployable |
| E | on-device 4-bit | Jetson stretch |

## Landmines (baked into procedure)
1. Blackwell sm_120 needs torch cu128+ / CUDA ≥12.8; skip flash-attn → SDPA.
2. Base `openvla-7b` ≈0% zero-shot on LIBERO — always use fine-tuned weights.
3. Avoid OFT transformers fork for run path; use base openvla action tokenization.
4. Eval requires `--center_crop True`.
5. Merge LoRA on same device used for eval.
6. Freeze tasks / trials / seeds before final numbers; 3 seeds for finals.
7. Fix MuJoCo headless (`MUJOCO_GL=egl`) before touching model.
8. `finetune.py` keeps only the latest checkpoint by default — snapshot each save (or pass `--save_latest_checkpoint_only False`) before relying on checkpoint selection.
9. Detached jobs die on host reboot — relaunch; long jobs need snapshot/resume discipline.

## Checkpoint selection pattern
Dev eval (10 trials/task, seed 7) on candidate checkpoints → pick best → full frozen eval (50 × 3 seeds) on the winner only. Arm A: 20k (80%) vs 50k (86%) → 50k.

## Metrics (identical recipe every arm)
Success %, median latency ms (warmup + N≥100), peak mem GB, on-disk size GB, throughput act/s, **energy J/action (NVML)**.

## Logging pattern
Copy `research/experiments/` template per run: hypothesis, config, hardware versions, five metrics, failures, next.

## Repo layout pattern
- `plan/` — execution plan.
- `research/` — RQ, papers, experiments, results, figures, weekly-log.
- `scripts/` — repo tooling: `phase0_bringup.sh` (gated env setup), `metrics.py` (5-metric harness §7), `train_arm_a.sh` (LoRA train + merge), `run_arm.sh` (eval + metrics → results JSON).
- `research/experiments/` — `EXP-*.md` logs, detached launchers, `measure.py` (five metrics + energy). Run logs (`*.log`, `*.pid`) are gitignored.
- `memory-bank/` — agent continuity docs (this tree).
- External clones (`~/vla/openvla`, LIBERO, datasets) live outside this repo unless later vendored.

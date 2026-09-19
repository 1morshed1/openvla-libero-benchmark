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

## Metrics (identical recipe every arm)
Success %, median latency ms (warmup + N≥100), peak mem GB, on-disk size GB, throughput act/s.

## Logging pattern
Copy `research/experiments/` template per run: hypothesis, config, hardware versions, five metrics, failures, next.

## Repo layout pattern
- `plan/` — execution plan.
- `research/` — RQ, papers, experiments, results, figures, weekly-log.
- `scripts/` — repo tooling: `phase0_bringup.sh` (gated env setup), `metrics.py` (5-metric harness §7), `train_arm_a.sh` (LoRA train + merge), `run_arm.sh` (eval + metrics → results JSON).
- `memory-bank/` — agent continuity docs (this tree).
- External clones (`~/vla/openvla`, LIBERO, datasets) live outside this repo unless later vendored.

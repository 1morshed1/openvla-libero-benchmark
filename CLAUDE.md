# CLAUDE.md — openvla-libero-benchmark

Agent instructions for this repo. Continuity lives in `memory-bank/`; read it at task start.

## What this repo is

Controlled experiment: **LoRA + PTQ** on OpenVLA vs **LIBERO-Spatial** success/cost trade-off. Not a deployment demo.

- **RQ:** How much can LoRA + PTQ cut latency / peak mem / model size while keeping success rate?
- **Paper row:** Arms A→B→C (precision only). REF = harness check. D/E = stretch.
- **Source of truth:** `plan/VLA-project-plan.md`, `research/README.md`, `memory-bank/`.

## Memory bank (read every session)

| File | Use |
|------|-----|
| `memory-bank/projectbrief.md` | Scope, RQ, hypotheses |
| `memory-bank/productContext.md` | Why / workflow |
| `memory-bank/systemPatterns.md` | Matrix, landmines, logging |
| `memory-bank/techContext.md` | Hardware, pins, deps |
| `memory-bank/activeContext.md` | Current focus + next |
| `memory-bank/progress.md` | Checklist + risks |

On **update memory bank**: review all six; refresh `activeContext.md` + `progress.md` especially.

## Hard rules

1. **Base `openvla/openvla` for runs** — do not use OFT transformers-fork path for the main experiment (silent ~0% risk).
2. **LoRA is the object of study** — do not switch to full fine-tune just because the 96 GB card allows it.
3. **Never install flash-attn** on Blackwell — use SDPA (`attn_implementation="sdpa"`).
4. **Keep transformers 4.40.1** (and related OpenVLA pins); only intentionally break the torch pin (cu128+ / sm_120).
5. **Eval always:** `--center_crop True`, `unnorm_key = libero_spatial_no_noops`.
6. **Base `openvla-7b` ≈0% zero-shot** — always fine-tuned / LoRA-merged weights for LIBERO success.
7. **Freeze eval protocol before training** — tasks, trials/task, seeds; log hardware versions every EXP.
8. **Merge LoRA on the same GPU used for eval.**
9. **Home RTX 2060 = no 7B** — notes/plots only. Office RTX PRO 6000 for train/eval/quantize.
10. Prefer correctness + experiment logs over speed. No speculative features beyond the plan.

## Experiment discipline

- Log every run as `research/experiments/EXP-XXX.md` (template in plan §8).
- Raw metrics → `research/results/` (one row/arm: success, latency, peak mem, size, throughput).
- Five metrics measured identically every arm (plan §7).
- REF gate: released `openvla-7b-finetuned-libero-spatial` ≈ **84.7%** before trusting harness on own weights.
- After meaningful work: update `memory-bank/activeContext.md` + `progress.md`.

## Repo layout

```
plan/           # full execution plan
research/       # RQ README, papers, EXP logs, results, figures, weekly-log
memory-bank/    # agent continuity
CLAUDE.md       # this file
```

External clones (`openvla`, `LIBERO`, datasets) typically under `~/vla/` — not necessarily in this git tree.

## Phase map (short)

| Phase | Goal |
|-------|------|
| 0 | Env + MuJoCo EGL + REF EXP-000 + paper notes |
| 1 | Freeze harness + Arm A LoRA merge + EXP-001 |
| 2 | B int8 / C int4 (/ D GPTQ-AWQ) — fill table |
| 3 | Write-up, figures, sim demo |
| 4 | Jetson E (stretch) |
| 5 | Public repo + paper |

## When editing code

- Touch only what the task needs; match existing style.
- Do not commit unless user asks.
- Do not push unless user asks.
- After env/train script changes: note landmines in EXP log + memory-bank if durable.

## Commands / env reminders

```bash
# Office GPU sanity
nvidia-smi
nvcc --version   # want >= 12.8
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"  # expect (12, 0)

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

Tiny REF smoke (after openvla install):

```bash
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 3
```

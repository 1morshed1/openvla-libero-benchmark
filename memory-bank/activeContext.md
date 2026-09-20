# Active Context

## Current focus
Full REF (**50 trials/task**) **queued** on GPU-1: waiter polls until ≥70 GiB free, then launches. GPU-1 currently held by another user’s Ray/SFPO job (~72 GiB). Smoke EXP-000 already at 90% (3 trials).

Wait/run logs under `research/experiments/ref_full50_*`.

## Session constraint (hard)
**GPU-1 only** — always `export CUDA_VISIBLE_DEVICES=1`. Never use GPU-0 or GPU-2.

## Recent changes (2026-09-20, office `vm-130-131`)
- Miniconda `~/miniconda3`, env `openvla` (Py 3.10); torch **2.11.0+cu128**; gate2 `(12,0)`+matmul on GPU-1.
- `~/vla/{openvla,openvla-oft,LIBERO}` cloned; openvla installed via **`--no-deps`** + curated pins (transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10, peft 0.11.1, bnb 0.50.2).
- SDPA patches in openvla; Hub `dataset_statistics.json` fetch in `openvla_utils.py`.
- LIBERO config `~/.libero/config.yaml`; `torch.load(..., weights_only=False)` for init states.
- **mujoco==3.3.2** (3.13 broke robosuite); opencv 4.9.0.80; numpy 1.26.4; protobuf 6.31.1.
- Disk expanded earlier (~2.0 TiB free enough).
- **EXP-000 smoke:** released spatial ckpt, 3 trials/task, center_crop True → **90%** success. Log: `research/experiments/EXP-000.md`.

## Resume / next commands
```bash
export CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$HOME/vla/openvla:$HOME/vla/LIBERO:$PYTHONPATH"
eval "$(~/miniconda3/bin/conda shell.bash hook)" && conda activate openvla
cd ~/vla/openvla
# full REF gate
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 50
```

## Active decisions
- Base openvla run path (not OFT).
- LoRA is object of study.
- GPU-1 only.
- Pin mujoco 3.3.2 for LIBERO fidelity.

## Open questions / blockers
- Full 50-trial REF not yet run (smoke only).
- Eval-protocol card values not frozen.
- HF auth optional (public REF ckpt worked without token).

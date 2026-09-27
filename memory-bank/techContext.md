# Tech Context

## Hardware
| Box | GPU | Role |
|-----|-----|------|
| Office (`vm-130-131`) | 3× RTX PRO 6000 Blackwell ~96 GB each | Train/eval/quantize — **GPU-1 only, for everything** (`CUDA_VISIBLE_DEVICES=1`). Never GPU-0 or GPU-2. |
| Home | RTX 2060, 6 GB | Notes, plots, draft only — no 7B |
| Stretch | Jetson Xavier NX 8/16 GB | Phase 4 only |

**Access:** office box = persistent disk under `/office/dev_workspace/morshed`, shared machine (other GPUs in use). Sun–Thu interactive; **jobs may run unattended overnight/weekend**. Root LV expanded to **~2.0 TiB** on 2026-09-20 (~1.5 TiB free at check); still Docker-heavy on the shared host. Home box available anytime for CPU/writing.

**Full rig notes:** `research/hardware-office-vm-130-131.md`.

## Software targets (office)
- Linux; Python 3.10 (`conda` env `openvla`).
- CUDA toolkit ≥ 12.8; driver recent (e.g. ≥570/580).
- PyTorch cu128+ (2.7+ first stable sm_120); verify capability `(12, 0)`.
- transformers **4.40.1**, tokenizers 0.19.1, timm 0.9.10 (keep OpenVLA pins except torch).
- **No flash-attn** — SDPA attention in model-load / finetune sites.
- bitsandbytes **0.50.2** (sm_120 kernels) for int8/int4.
- `nvidia-ml-py` for NVML energy in `measure.py`.
- LIBERO + `experiments/robot/libero/libero_requirements.txt`.
- MuJoCo: **pin `mujoco==3.3.2`** (3.13 breaks robosuite 1.4.x; ≥3.4 drifts libero_spatial init settle). `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`.
- Install openvla with **`pip install -e . --no-deps`** then curated deps — never let pip pull `torch==2.2.0`.
- protobuf: **6.31.1** currently needed for `tensorflow-metadata` / tfds import on this box (TF 2.15 warns).

## External repos / data
- https://github.com/openvla/openvla — run path (`~/vla/openvla`).
- https://github.com/moojink/openvla-oft — read only (`~/vla/openvla-oft`).
- https://github.com/Lifelong-Robot-Learning/LIBERO (`~/vla/LIBERO`).
- HF dataset `openvla/modified_libero_rlds` (`libero_spatial_no_noops`, …).
- Checkpoint REF: `openvla/openvla-7b-finetuned-libero-spatial`.
- Base for LoRA: `openvla/openvla-7b`.
- HF cache on this host: `$HF_HOME` → `/office/shared_cache/.cache/huggingface`.

## Eval protocol (FROZEN 2026-09-21 — `research/datasets/libero-spatial-eval.md`)
- Suite: `libero_spatial`, all 10 tasks.
- Trials/task: 50 final (10 dev); seeds: {7, 42, 123} final, 7 for dev.
- `--center_crop True`; `unnorm_key` = `libero_spatial` (REF) / `libero_spatial_no_noops` (our arms).
- Device: RTX PRO 6000; log driver/CUDA/torch/bnb every run.
- Throughput: ~100 episodes/hour per GPU (bf16) → full 3-seed eval ≈ 15 h.

## LoRA recipe (Arm A)
- r=32, lr 5e-4, effective batch 128 (bs 16 × accum 8), 50k steps, `--image_aug True`.
- ~7.4 s/step on one RTX PRO 6000, ~63 GiB → 50k ≈ 5 days.
- `finetune.py` merges the adapter into the base at each save and writes the merged bf16 model to the run dir — that dir is directly evaluable (no separate merge step needed).
- Only the latest save is kept by default (`save_latest_checkpoint_only=True`).

## This git repo
- Remote: `git@github.com:1morshed1/openvla-libero-benchmark.git`
- Launchers in `research/experiments/`: `run_ref_full50_detached.sh`, `run_arm_a_train_detached.sh`, `run_arm_a_dev_eval_gpu2_detached.sh` (generic eval on GPU-1: CKPT/TRIALS/SEED/NOTE env vars; the `gpu2` in the name is legacy — rename after the running full eval finishes, since that loop calls it by path). All detach via `setsid nohup` to survive disconnects.
- Harness: `research/experiments/measure.py` (+ energy); older `scripts/metrics.py`, `scripts/run_arm.sh`, `scripts/train_arm_a.sh`.

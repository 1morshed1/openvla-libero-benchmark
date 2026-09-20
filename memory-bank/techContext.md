# Tech Context

## Hardware
| Box | GPU | Role |
|-----|-----|------|
| Office (`vm-130-131`) | 3× RTX PRO 6000 Blackwell ~96 GB each | Train/eval/quantize — **this user: GPU-1 only** (`CUDA_VISIBLE_DEVICES=1`) |
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
- bitsandbytes recent (sm_120 kernels) for int8/int4.
- LIBERO + `experiments/robot/libero/libero_requirements.txt`.
- MuJoCo: `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl` (fallback osmesa).

## External repos / data
- https://github.com/openvla/openvla — run path.
- https://github.com/moojink/openvla-oft — read only.
- https://github.com/Lifelong-Robot-Learning/LIBERO
- HF dataset `openvla/modified_libero_rlds` (`libero_spatial_no_noops`, …).
- Checkpoint REF: `openvla/openvla-7b-finetuned-libero-spatial`.
- Base for LoRA: `openvla/openvla-7b`.

## Eval protocol (to freeze in Phase 1)
- Suite: `libero_spatial`.
- Trials/task: 50 final (10–20 dev); seeds: 3 final.
- `--center_crop True`; `unnorm_key = libero_spatial_no_noops`.
- Device: RTX PRO 6000; log driver/CUDA/torch/bnb every run.

## LoRA recipe (Arm A)
- r=32, lr 5e-4, effective batch 128 (e.g. bs 16 × accum 8).
- `--image_aug True` (pairs with center_crop at eval).
- Merge via openvla merge script on eval device.

## This git repo
- Remote: `git@github.com:1morshed1/openvla-libero-benchmark.git`
- Currently: plan + research scaffold; no train/eval code checked in yet.

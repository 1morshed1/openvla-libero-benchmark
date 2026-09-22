#!/usr/bin/env bash
# Detached Arm A LoRA train — survives laptop disconnect (setsid/nohup).
# GPU-1 only. W&B offline (no stanford-voltron login required).
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_MODE=offline
export PYTHONPATH="$HOME/vla/openvla:${PYTHONPATH:-}"
# Detached shells skip .bashrc — pin shared HF cache so we don't re-download.
export HF_HOME="${HF_HOME:-/office/shared_cache/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"

LOGDIR="$HOME/projects/openvla-libero-benchmark/research/experiments"
mkdir -p "$LOGDIR" "$HOME/vla/runs" "$HOME/vla/adapters"
LOG="$LOGDIR/arm_a_train_detached.log"

eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
conda activate openvla
cd "$HOME/vla/openvla"

{
  echo "START $(date -Is) Arm A LoRA r=32 GPU-1"
  echo "torch=$(python -c 'import torch; print(torch.__version__, torch.cuda.get_device_capability(0))')"
  nvidia-smi -i 1 --query-gpu=memory.free,memory.total --format=csv
  bash "$HOME/projects/openvla-libero-benchmark/scripts/train_arm_a.sh" train
  echo "END $(date -Is)"
  echo "ARM_A_TRAIN_DONE"
} >> "$LOG" 2>&1

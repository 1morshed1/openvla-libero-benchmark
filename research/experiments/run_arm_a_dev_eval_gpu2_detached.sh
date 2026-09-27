#!/usr/bin/env bash
# Detached Arm A eval on GPU-1.
# Protocol: libero_spatial, 10 trials/task, seed 7, center_crop True.
# Checkpoint: frozen snapshot of the step-20k merged bf16 weights.
set -euo pipefail
GPU="${GPU:-1}"
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TF_CPP_MIN_LOG_LEVEL=3
export WANDB_MODE=offline
export PYTHONPATH="$HOME/vla/openvla:$HOME/vla/LIBERO:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/office/shared_cache/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

CKPT="${CKPT:-$HOME/vla/merged/arm-A-dev-20k-bf16}"
TRIALS="${TRIALS:-10}"
SEED="${SEED:-7}"
NOTE="${NOTE:-armA-20k-dev}"

LOGDIR="$HOME/projects/openvla-libero-benchmark/research/experiments"
mkdir -p "$LOGDIR" "$HOME/vla/openvla/experiments/logs"
LOG="$LOGDIR/arm_a_dev_eval_${NOTE}.log"

eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
conda activate openvla
cd "$HOME/vla/openvla"

{
  echo "START $(date -Is) Arm A DEV eval GPU-$GPU"
  echo "ckpt=$CKPT trials=$TRIALS seed=$SEED note=$NOTE"
  echo "torch=$(python -c 'import torch; print(torch.__version__, torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))')"
  nvidia-smi -i "$GPU" --query-gpu=memory.free,memory.used,memory.total --format=csv
  python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint "$CKPT" \
    --task_suite_name libero_spatial \
    --center_crop True \
    --num_trials_per_task "$TRIALS" \
    --seed "$SEED" \
    --run_id_note "$NOTE" \
    --use_wandb False
  echo "END $(date -Is)"
  echo "ARM_A_DEV_EVAL_DONE"
} >> "$LOG" 2>&1

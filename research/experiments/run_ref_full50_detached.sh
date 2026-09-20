#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH="$HOME/vla/openvla:$HOME/vla/LIBERO:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=3

LOGDIR="$HOME/projects/openvla-libero-benchmark/research/experiments"
mkdir -p "$LOGDIR"
WAITLOG="$LOGDIR/ref_full50_wait_detached.log"
RUNLOG="$LOGDIR/ref_full50_detached.log"
NEED_FREE_MIB=70000
POLL_SEC=60

{
  echo "WAIT_START $(date -Is) (detached/nohup)"
  echo "Polling every ${POLL_SEC}s until GPU-1 free >= ${NEED_FREE_MIB} MiB"
  while true; do
    free_mib=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    used_mib=$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    echo "$(date -Is) GPU1 used=${used_mib}MiB free=${free_mib}MiB"
    if [ "$free_mib" -ge "$NEED_FREE_MIB" ]; then
      echo "GPU1 free enough — launching full REF"
      break
    fi
    sleep "$POLL_SEC"
  done
} >> "$WAITLOG" 2>&1

eval "$("$HOME/miniconda3/bin/conda" shell.bash hook)"
conda activate openvla
cd "$HOME/vla/openvla"

{
  echo "START $(date -Is) trials=50 suite=libero_spatial gpu=1"
  python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True \
    --num_trials_per_task 50
  echo "END $(date -Is)"
  echo "REF_FULL50_DONE"
} >> "$RUNLOG" 2>&1

#!/usr/bin/env bash
# Wait until GPU-1 is idle, then run measure.py (perf + NVML energy are whole-GPU,
# so any other load on the card would contaminate them).
# Env: ARM (A), PRECISION (bf16), CKPT, SUCCESS_RATE, IDLE_UTIL (5 %), IDLE_POWER (130 W),
#      IDLE_SAMPLES (10 consecutive 30 s samples = 5 min).
set -euo pipefail
GPU=1
export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME=/office/shared_cache/.cache/huggingface

ARM="${ARM:-A}"
PRECISION="${PRECISION:-bf16}"
CKPT="${CKPT:-$HOME/vla/merged/arm-A-bf16}"
SUCCESS_RATE="${SUCCESS_RATE:-86.0}"
IDLE_UTIL="${IDLE_UTIL:-5}"
IDLE_POWER="${IDLE_POWER:-130}"
IDLE_SAMPLES="${IDLE_SAMPLES:-10}"

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LOG="$REPO/research/experiments/measure_arm_${ARM}_${PRECISION}.log"
exec >>"$LOG" 2>&1
echo "START $(date -u +%FT%TZ) arm=$ARM precision=$PRECISION ckpt=$CKPT"

ok=0
while [ "$ok" -lt "$IDLE_SAMPLES" ]; do
  read -r util power < <(nvidia-smi -i "$GPU" --query-gpu=utilization.gpu,power.draw \
    --format=csv,noheader,nounits | tr -d ',')
  if [ "${util%.*}" -le "$IDLE_UTIL" ] && [ "${power%.*}" -le "$IDLE_POWER" ]; then
    ok=$((ok + 1))
  else
    ok=0
  fi
  echo "$(date -u +%T) util=${util}% power=${power}W idle_streak=$ok/$IDLE_SAMPLES"
  [ "$ok" -lt "$IDLE_SAMPLES" ] && sleep 30
done

echo "GPU-$GPU idle — launching measure.py $(date -u +%FT%TZ)"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate openvla
cd "$REPO"
python research/experiments/measure.py --arm "$ARM" --precision "$PRECISION" \
  --checkpoint "$CKPT" --success_rate "$SUCCESS_RATE"
echo "END $(date -u +%FT%TZ)"
echo MEASURE_DONE

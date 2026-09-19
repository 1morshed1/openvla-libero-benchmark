#!/usr/bin/env bash
# Run one arm end-to-end: LIBERO eval (success %) + metrics harness (latency/mem/
# size/throughput) -> research/results/<arm>.json. Same frozen protocol every arm.
#
#   scripts/run_arm.sh <ARM> <CHECKPOINT> <PRECISION> [dev|final]
#   ARM        REF | A | B | C | D
#   CHECKPOINT local dir or HF hub id
#   PRECISION  bf16 | int8 | int4   (B=int8 Arm-A ckpt, C=int4 Arm-A ckpt)
#   mode       dev  = fast (few trials, 1 seed);  final = frozen (50 trials, 3 seeds)
#
# Frozen protocol (never change once set): --center_crop True (landmine #4),
# unnorm_key libero_spatial_no_noops, suite libero_spatial. Fill exact trials/
# seeds in research/datasets/libero-spatial-eval.md and mirror here.
#
# NOTE: success-rate parsing from run_libero_eval.py stdout is best-effort — the
# harness output format may differ. Verify the grep on the first REF run; if it
# misreads, pass SUCCESS=<pct> to skip parsing and write the number by hand.
set -euo pipefail

ARM="${1:?ARM (REF|A|B|C|D)}"
CHECKPOINT="${2:?CHECKPOINT (dir or HF id)}"
PRECISION="${3:?PRECISION (bf16|int8|int4)}"
MODE="${4:-dev}"

VLA_ROOT="${VLA_ROOT:-$HOME/vla}"
OPENVLA_DIR="${OPENVLA_DIR:-$VLA_ROOT/openvla}"
SUITE="${SUITE:-libero_spatial}"
UNNORM_KEY="${UNNORM_KEY:-libero_spatial_no_noops}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${LOG_DIR:-$REPO_DIR/research/results/eval-logs}"
mkdir -p "$LOG_DIR"

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

# dev vs final trials/seeds — FREEZE final values in the eval card, then edit here once.
if [ "$MODE" = "final" ]; then
  TRIALS="${TRIALS:-50}"; SEEDS="${SEEDS:-7 42 123}"
else
  TRIALS="${TRIALS:-10}"; SEEDS="${SEEDS:-7}"
fi

echo "== Arm $ARM | $PRECISION | $MODE | suite=$SUITE | trials=$TRIALS | seeds=$SEEDS =="

# ---- 1. LIBERO eval -> success % (per seed, then averaged) -----------------
eval_one() {
  local seed="$1" log="$LOG_DIR/${ARM}_seed${seed}.log"
  cd "$OPENVLA_DIR"
  python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint "$CHECKPOINT" \
    --task_suite_name "$SUITE" \
    --center_crop True \
    --num_trials_per_task "$TRIALS" \
    --seed "$seed" 2>&1 | tee "$log"
  # best-effort parse: last "success rate" style number in the log (as %)
  grep -oiE "success[_ ]?rate[^0-9]*[0-9]+\.?[0-9]*" "$log" | tail -1 \
    | grep -oE "[0-9]+\.?[0-9]*" | tail -1
}

if [ -n "${SUCCESS:-}" ]; then
  echo "using SUCCESS=$SUCCESS (parsing skipped)"
  MEAN_SUCCESS="$SUCCESS"
else
  total=0; n=0
  for s in $SEEDS; do
    val="$(eval_one "$s" || true)"
    if [ -n "$val" ]; then
      # normalize: if <=1 treat as fraction -> percent
      val="$(python -c "v=float('$val'); print(v*100 if v<=1 else v)")"
      echo "  seed $s success=$val%"
      total="$(python -c "print($total + $val)")"; n=$((n+1))
    else
      echo "  seed $s: could not parse success (check $LOG_DIR/${ARM}_seed${s}.log)"
    fi
  done
  if [ "$n" -gt 0 ]; then
    MEAN_SUCCESS="$(python -c "print(round($total/$n, 2))")"
  else
    MEAN_SUCCESS=""
    echo "WARN: no success parsed — pass SUCCESS=<pct> and re-run, or fill the row by hand."
  fi
fi

# ---- 2. metrics harness (latency/mem/size/throughput + env) ----------------
SUCCESS_ARG=()
[ -n "${MEAN_SUCCESS:-}" ] && SUCCESS_ARG=(--success "$MEAN_SUCCESS")

python "$REPO_DIR/scripts/metrics.py" \
  --arm "$ARM" \
  --checkpoint "$CHECKPOINT" \
  --precision "$PRECISION" \
  --unnorm-key "$UNNORM_KEY" \
  "${SUCCESS_ARG[@]}"

echo ">> Row written: research/results/$ARM.json"
echo ">> Now copy research/experiments/EXP-template.md -> EXP-XXX.md and log this arm."

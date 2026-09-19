#!/usr/bin/env bash
# Arm A — LoRA r=32 fine-tune of openvla-7b on libero_spatial_no_noops, then
# merge the adapter into the base (bf16) = the Arm A deployable checkpoint.
#
# Recipe (plan §1.4/§1.5, memory-bank techContext): r=32, lr 5e-4, effective
# batch 128 (bs 16 x accum 8), image_aug True, save_steps 5000 (crash safety).
# LoRA is the OBJECT OF STUDY — do NOT switch to full fine-tune (CLAUDE.md #2).
# Merge on the SAME GPU used for eval (landmine #5).
#
# PREREQ patch (landmine #1/#3): in the openvla repo, replace
#   attn_implementation="flash_attention_2"  ->  "sdpa"
# in vla-scripts/finetune.py and experiments/robot/openvla_utils.py BEFORE running.
# Do NOT install flash-attn on Blackwell.
#
# Long run: launch before leaving to use the unattended Fri/Sat gap; checkpoints
# every 5k steps so a crash loses little. Eval intermediate ckpts, keep the best.
set -euo pipefail

VLA_ROOT="${VLA_ROOT:-$HOME/vla}"
OPENVLA_DIR="${OPENVLA_DIR:-$VLA_ROOT/openvla}"
DATA_ROOT="${DATA_ROOT:-$VLA_ROOT/modified_libero_rlds}"   # HF openvla/modified_libero_rlds
DATASET="${DATASET:-libero_spatial_no_noops}"
BASE_VLA="${BASE_VLA:-openvla/openvla-7b}"                  # base, NOT finetuned (LoRA target)
RUN_DIR="${RUN_DIR:-$VLA_ROOT/runs}"
ADAPTER_DIR="${ADAPTER_DIR:-$VLA_ROOT/adapters}"
MERGED_OUT="${MERGED_OUT:-$VLA_ROOT/merged/arm-A-bf16}"

LORA_RANK="${LORA_RANK:-32}"
LR="${LR:-5e-4}"
BATCH="${BATCH:-16}"
ACCUM="${ACCUM:-8}"          # effective batch = BATCH * ACCUM = 128
MAX_STEPS="${MAX_STEPS:-50000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"

echo "== Arm A LoRA fine-tune =="
echo "base=$BASE_VLA data=$DATA_ROOT/$DATASET  r=$LORA_RANK lr=$LR eff_batch=$((BATCH*ACCUM))"

train() {
  cd "$OPENVLA_DIR"
  # Flag names follow openvla vla-scripts/finetune.py — confirm on first run.
  torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
    --vla_path "$BASE_VLA" \
    --data_root_dir "$DATA_ROOT" \
    --dataset_name "$DATASET" \
    --run_root_dir "$RUN_DIR" \
    --adapter_tmp_dir "$ADAPTER_DIR" \
    --lora_rank "$LORA_RANK" \
    --batch_size "$BATCH" \
    --grad_accumulation_steps "$ACCUM" \
    --learning_rate "$LR" \
    --image_aug True \
    --max_steps "$MAX_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --use_lora True
  echo ">> Adapter(s) under $ADAPTER_DIR / run under $RUN_DIR."
  echo ">> Eval intermediate checkpoints with scripts/run_arm.sh; keep the best before merge."
}

# Merge chosen adapter into base -> bf16 deployable. ADAPTER=<path> merge
merge() {
  local adapter="${ADAPTER:-$1}"
  [ -n "$adapter" ] || { echo "set ADAPTER=<adapter_dir> (the best checkpoint)"; exit 1; }
  echo "== Merging $adapter into $BASE_VLA -> $MERGED_OUT (on eval GPU) =="
  mkdir -p "$MERGED_OUT"
  python - "$BASE_VLA" "$adapter" "$MERGED_OUT" <<'PY'
import sys, torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
base_id, adapter_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
base = AutoModelForVision2Seq.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    low_cpu_mem_usage=True, trust_remote_code=True)
merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
merged.save_pretrained(out)
AutoProcessor.from_pretrained(base_id, trust_remote_code=True).save_pretrained(out)
print("merged ->", out)
PY
  echo ">> Arm A checkpoint ready: $MERGED_OUT"
  echo ">> Next: scripts/run_arm.sh A $MERGED_OUT bf16   (logs EXP-001)"
}

case "${1:-}" in
  train) train ;;
  merge) shift; merge "${1:-}" ;;
  *) echo "usage: $0 train | $0 merge <adapter_dir>   (or ADAPTER=<dir> $0 merge)"; exit 1 ;;
esac

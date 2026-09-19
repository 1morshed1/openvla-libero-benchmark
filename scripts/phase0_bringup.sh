#!/usr/bin/env bash
# Phase-0 bring-up — OpenVLA + LIBERO on office RTX PRO 6000 (Blackwell sm_120).
# Run SECTION BY SECTION, not blind. Stop at each GATE and read output before continuing.
# Landmines baked in: torch cu128+, SDPA (no flash-attn), transformers 4.40.1, center_crop.
set -euo pipefail

VLA_ROOT="${VLA_ROOT:-$HOME/vla}"
ENV_NAME="${ENV_NAME:-openvla}"
PY_VER="3.10"

echo "== Phase-0 bring-up =="
echo "VLA_ROOT=$VLA_ROOT  ENV=$ENV_NAME"

# ---------------------------------------------------------------------------
# GATE 0 — hardware / toolkit sanity (read output, do NOT auto-continue)
# ---------------------------------------------------------------------------
gate0_check() {
  echo "--- nvidia-smi ---"; nvidia-smi
  echo "--- nvcc ---";       nvcc --version || echo "WARN: nvcc not on PATH (need CUDA toolkit >=12.8)"
  echo ">> Want: driver recent (>=570/580), CUDA toolkit >=12.8, Blackwell visible."
}

# ---------------------------------------------------------------------------
# 1 — conda env
# ---------------------------------------------------------------------------
make_env() {
  conda create -y -n "$ENV_NAME" python="$PY_VER"
  echo ">> Now: conda activate $ENV_NAME   (then rerun this script's later sections)"
}

# ---------------------------------------------------------------------------
# 2 — torch (Blackwell sm_120 needs cu128+, torch 2.7+). Try stable cu128 first.
# ---------------------------------------------------------------------------
install_torch() {
  pip install --upgrade pip
  # stable cu128; if sm_120 kernels missing, fall back to nightly cu128 (see note)
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  echo ">> If capability check below fails / sm_120 warning: retry with"
  echo "   pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128"
}

# GATE 2 — torch sees Blackwell
gate2_check() {
  python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("is_available", torch.cuda.is_available())
cap = torch.cuda.get_device_capability(0)
print("capability", cap, "name", torch.cuda.get_device_name(0))
assert torch.cuda.is_available(), "CUDA not available"
assert cap == (12, 0), f"expected (12,0) got {cap} — wrong torch/CUDA build"
# real kernel smoke (catches 'no kernel image' sm_120 mismatch)
x = torch.randn(2048, 2048, device="cuda"); (x @ x).sum().item()
print("matmul on GPU OK")
PY
}

# ---------------------------------------------------------------------------
# 3 — external clones (run path = base openvla; OFT read-only)
# ---------------------------------------------------------------------------
clone_repos() {
  mkdir -p "$VLA_ROOT"; cd "$VLA_ROOT"
  [ -d openvla ]     || git clone https://github.com/openvla/openvla.git
  [ -d openvla-oft ] || git clone https://github.com/moojink/openvla-oft.git   # READ ONLY, not run path
  [ -d LIBERO ]      || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
}

# ---------------------------------------------------------------------------
# 4 — install openvla WITHOUT clobbering torch; SDPA only, no flash-attn
# ---------------------------------------------------------------------------
install_openvla() {
  cd "$VLA_ROOT/openvla"
  # editable install but keep our torch: install deps, then pin the important ones
  pip install -e . || true
  # re-pin OpenVLA-critical versions (torch intentionally NOT pinned)
  pip install "transformers==4.40.1" "tokenizers==0.19.1" "timm==0.9.10"
  pip install -U bitsandbytes   # recent, needs sm_120 kernels for int8/int4 (Arms B/C)
  echo ">> DO NOT install flash-attn. Model load uses attn_implementation='sdpa'."
  # verify torch survived the openvla install
  python -c "import torch;print('torch after openvla install:',torch.__version__)"
}

install_libero() {
  cd "$VLA_ROOT/LIBERO"
  pip install -e .
  pip install -r "$VLA_ROOT/openvla/experiments/robot/libero/libero_requirements.txt"
}

# ---------------------------------------------------------------------------
# 5 — MuJoCo headless EGL smoke (FIX BEFORE touching model — landmine #7)
# ---------------------------------------------------------------------------
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
gate5_mujoco() {
  python - <<'PY'
import os
print("MUJOCO_GL", os.environ.get("MUJOCO_GL"))
import numpy as np, mujoco
xml = "<mujoco><worldbody><geom type='box' size='.1 .1 .1'/></worldbody></mujoco>"
m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
r = mujoco.Renderer(m, 128, 128); mujoco.mj_forward(m, d); r.update_scene(d)
img = r.render()
print("render OK", img.shape, "mean", float(np.asarray(img).mean()))
PY
  echo ">> If EGL fails: try PYOPENGL_PLATFORM=osmesa (fallback) or check libEGL / GPU EGL support."
}

# ---------------------------------------------------------------------------
# 6 — REF eval → EXP-000. Tiny smoke first, THEN full ~84.7% gate.
# ---------------------------------------------------------------------------
ref_smoke() {
  cd "$VLA_ROOT/openvla"
  python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True \
    --num_trials_per_task 3
  echo ">> Smoke ran end-to-end? Then run full eval (drop --num_trials or set 50) for EXP-000 gate."
  echo ">> Gate: released ckpt must hit ~84.7% before trusting harness on OUR weights."
}

cat <<'EOF'

USAGE — run in order, stop at each GATE:
  gate0_check          # hardware/toolkit — READ before continuing
  make_env             # then: conda activate openvla
  install_torch ; gate2_check    # MUST pass (12,0) + matmul before proceeding
  clone_repos
  install_openvla      # verify torch not clobbered
  install_libero
  gate5_mujoco         # MUST render before touching model
  ref_smoke            # tiny smoke, then full for EXP-000

Source this file then call functions, e.g.:
  source phase0_bringup.sh
  gate0_check
EOF

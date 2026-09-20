# Active Context

## Current focus
Phase 0 bring-up on office box — **disk blocker cleared** (root ~2.0 TiB, ~1.5 TiB free as of 2026-09-20 afternoon). Ready to resume openvla `--no-deps` install → MuJoCo → REF/EXP-000. Rig details: `research/hardware-office-vm-130-131.md`.

## Session constraint (hard)
**GPU-1 only** — always `export CUDA_VISIBLE_DEVICES=1`. Never use GPU-0 or GPU-2.

## Recent changes (2026-09-20, office `vm-130-131`)
- Miniconda installed at `~/miniconda3`; env `openvla` (Python 3.10).
- Torch **2.11.0+cu128** installed; gate2 **PASS**: capability `(12, 0)`, matmul OK on GPU-1 (physical RTX PRO 6000).
- Cloned to `~/vla/`: `openvla` (run path), `openvla-oft` (read-only), `LIBERO`.
- SDPA patches applied in `~/vla/openvla` (not flash-attn):
  - `experiments/robot/openvla_utils.py` → `attn_implementation="sdpa"`
  - LLM backbone defaults `use_flash_attention_2=False` (llama2/mistral/phi)
  - `vla-scripts/deploy.py` default → `"sdpa"`
- **Blocked:** naive `pip install -e .` tried to pull `torch==2.2.0` + TensorFlow and eat disk. Install killed; cu128 torch **survived**. Disk was ~17–22 GB free / 97% — too tight for REF.
- No system `nvcc` on PATH (CUDA toolkit not installed); PyTorch ships its own CUDA 12.8 — OK for now.
- `HF_TOKEN` unset — needed before REF download if gated.

## Resume checklist (after disk free)
1. `export CUDA_VISIBLE_DEVICES=1`
2. `eval "$(~/miniconda3/bin/conda shell.bash hook)" && conda activate openvla`
3. Confirm torch still cu128 + `(12,0)`: `python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0))"`
4. Install openvla with **`--no-deps`**, then install deps **without** re-pinning torch; re-pin `transformers==4.40.1` `tokenizers==0.19.1` `timm==0.9.10`; `pip install -U bitsandbytes`; verify torch not clobbered.
5. `install_libero` + libero_requirements.
6. `gate5_mujoco` (EGL) — must pass before model.
7. `ref_smoke` → full REF → **EXP-000** (~84.7%).
8. Freeze `research/datasets/libero-spatial-eval.md`.

## Disk notes
- Root LV expanded **501G → ~2.0 TiB** (2026-09-20); ~1.5 TiB free at afternoon check — enough for deps + REF.
- Docker images on shared box still ~200G; optional prune if pressure returns.
- Clear leftover `/tmp/pip-*` and `~/.cache/pip` if still present from aborted install.

## Active decisions
- Run base openvla, not OFT fork path.
- LoRA is object of study despite 96 GB card.
- Sim A–C table = complete paper; Jetson optional.
- GPU-1 only for this user/session.

## Open questions / blockers
- Disk no longer blocking (expanded).
- HF auth before REF pull (`HF_TOKEN` / `huggingface-cli login`).
- Optional: install CUDA toolkit 12.8 for `nvcc` (not required if torch wheels suffice).

# Office hardware — `vm-130-131`

Reference snapshot for the OpenVLA / LIBERO benchmark work. Re-check with the commands at the bottom before logging EXP metrics; numbers below are from **2026-09-20**.

## Identity

| Field | Value |
|-------|--------|
| Hostname | `vm-130-131` |
| OS | Ubuntu 24.04.4 LTS (Noble) |
| Kernel | `6.8.0-139-generic` |
| Virt | KVM guest (full virt) |
| Home / workspace | `/office/dev_workspace/morshed` |

## CPU

| Field | Value |
|-------|--------|
| Model | AMD Eng Sample: `100-000001153-07` (family 26) |
| Cores | 48 (1 thread/core, 1 socket) |
| L3 | 768 MiB |
| NUMA | 1 node |

## Memory

| Field | Value |
|-------|--------|
| RAM | **157 GiB** |
| Swap | 8 GiB |

## Storage

| Field | Value |
|-------|--------|
| Root LV | `/dev/mapper/ubuntu--vg-ubuntu--lv` |
| Size (after expand) | **2.0 TiB** (~467 GiB used, **~1.5 TiB free**, ~25%) |
| Prior state (morning) | ~501 GiB total, ~97% full (~17 GiB free) — blocked Phase 0 install / REF ckpt |

Disk was expanded on 2026-09-20 afternoon; Phase 0 can resume without a space crisis. Docker on the shared box still holds ~200 GiB of images (mostly other projects).

## GPUs (3× Blackwell)

Driver **580.178.04**. CUDA runtime via PyTorch wheels: **12.8**. Compute capability **sm_120** → `(12, 0)`.

| Index | Name | VRAM | Power limit (observed) | Notes |
|-------|------|------|------------------------|--------|
| 0 | NVIDIA RTX PRO 6000 Blackwell Server Edition | ~97887 MiB (~96 GiB) | 600 W | Shared / other users — **do not use** |
| **1** | NVIDIA RTX PRO 6000 Blackwell Server Edition | ~97887 MiB (~96 GiB) | **600 W** | **Ours — only GPU for this project** |
| 2 | NVIDIA RTX PRO 6000 Blackwell Server Edition | ~97887 MiB (~96 GiB) | 450 W (often) | Shared — **do not use** |

### Topology

`nvidia-smi topo -m` shows **PHB** between GPUs (PCIe via host bridge). **No NVLink.** Fine for single-GPU train/eval and multi-replica serving; poor for single-job tensor-parallel across cards.

### Soft allocation (hard rule for this work)

```bash
export CUDA_VISIBLE_DEVICES=1
```

- Always use **physical GPU-1**.
- With that env set, PyTorch `cuda:0` maps to physical GPU-1.
- Merge LoRA on the **same** GPU used for eval (landmine) → merge on GPU-1.
- Never schedule train/eval/quantize on GPU-0 or GPU-2.

Afternoon check (2026-09-20 ~17:08 local): GPU-1 was **idle** (0 MiB used, ~22°C). GPU-0/2 had other workloads.

## Software stack (our env)

| Piece | Value / location |
|-------|------------------|
| Miniconda | `~/miniconda3` |
| Conda env | `openvla` (Python 3.10) |
| PyTorch | **2.11.0+cu128** — gate2 pass: `(12, 0)` + matmul on GPU-1 |
| System `nvcc` | Not on PATH (no full CUDA toolkit install); torch ships CUDA 12.8 — OK for train/eval |
| External clones | `~/vla/openvla` (run), `~/vla/openvla-oft` (read-only), `~/vla/LIBERO` |
| Attention | **SDPA only** — do not install flash-attn on Blackwell; patches in `~/vla/openvla` |

Activate:

```bash
export CUDA_VISIBLE_DEVICES=1
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate openvla
```

## Landmines tied to this box

1. Blind `pip install -e ~/vla/openvla` pulls `torch==2.2.0` and breaks sm_120 — use `--no-deps`, then curated deps; re-verify cu128.
2. No flash-attn — keep `attn_implementation="sdpa"`.
3. Shared machine: other containers/jobs use GPU-0/2 and disk/Docker; always pin GPU-1.
4. Log driver / CUDA / torch / bitsandbytes versions on every EXP (plan §7).

## Sanity commands

```bash
export CUDA_VISIBLE_DEVICES=1
nvidia-smi -i 1
df -h /
free -h
eval "$(~/miniconda3/bin/conda shell.bash hook)" && conda activate openvla
python -c "import torch; print(torch.__version__, torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))"
```

## Related docs

- `memory-bank/techContext.md` — short hardware table + software pins
- `memory-bank/activeContext.md` — current Phase 0 resume checklist
- `scripts/phase0_bringup.sh` — gated bring-up
- `CLAUDE.md` — project hard rules

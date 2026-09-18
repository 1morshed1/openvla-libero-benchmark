# VLA Efficiency Project — Full Execution Plan

**Working paper title:** *Parameter-efficient adaptation and quantization of vision-language-action policies for resource-constrained robots.*

**One-line framing:** This is a controlled experiment, not a deployment demo. Every step feeds one question and one results table.

**Environment reality:** Dev box is a single **RTX PRO 6000 Blackwell, 96 GB** on **Linux**. Target (stretch) hardware is a **Jetson Xavier NX**. No time pressure — favor correctness and logging discipline over speed.

---

## 0. The research question (write this first, before any code)

> **RQ:** How much can parameter-efficient fine-tuning (LoRA) plus post-training quantization (PTQ) reduce VLA inference cost — latency, peak memory, model size — while preserving manipulation success rate on a controlled LIBERO task?

**Hypotheses (register these before you have results):**
- **H1:** LoRA (r=32) fine-tuning recovers most of the full fine-tuned success rate on LIBERO-Spatial (target: within a few points of the published ~84.7%).
- **H2:** 8-bit PTQ preserves success rate within noise while cutting peak memory and model size substantially.
- **H3:** 4-bit PTQ produces a measurable but bounded success-rate drop, and the size/latency/memory savings are large enough to make the trade-off worth it for edge hardware.
- **H4 (stretch):** The quantized policy fits and runs on Jetson Xavier NX; the cost of making a 7B VLA fit on 8–16 GB is itself a reportable result.

**The five metrics (define once, measure identically for every arm):**

| Metric | Definition | How measured |
|---|---|---|
| Success rate (%) | Fraction of successful rollouts on the frozen eval set | `run_libero_eval.py` output (successes / trials) |
| Latency (ms) | Median wall-clock per `predict_action` call, warmed up | `torch.cuda.synchronize()` + timer over N≥100 calls |
| Peak memory (GB) | Peak GPU memory during inference | `torch.cuda.max_memory_allocated()` + `nvidia-smi` for context |
| Model size (GB) | On-disk size of the deployable artifact | `du -sh` of checkpoint / merged model |
| Throughput (act/s) | Actions (or action-chunks) per second | `1000 / latency_ms`, or rollout timing |

**Anchor number:** the published OpenVLA LIBERO-Spatial result is **84.7% ± 0.9%** (avg of 3 seeds × 500 rollouts). When your eval harness runs the *released* `openvla-7b-finetuned-libero-spatial` checkpoint and returns ~84–85%, your harness is trustworthy. That is the gate before you trust it on your own weights.

---

## 1. Locked experiment matrix

You fine-tune your own policy (you have the compute), and the controlled comparison **holds the weights fixed and varies only precision**.

| ID | Arm | Weights | Precision | Purpose |
|---|---|---|---|---|
| REF | Reference | Released `openvla-7b-finetuned-libero-spatial` | bf16 | Sanity-check the harness against 84.7% |
| A | Baseline (your LoRA) | Your r=32 LoRA fine-tune, merged into base | bf16 | The honest full-precision point; proves the pipeline |
| B | int8 | Arm A weights | int8 (bitsandbytes) | Cheap PTQ point |
| C | int4 | Arm A weights | int4 / NF4 (bitsandbytes) | Aggressive PTQ point |
| D (stretch) | GPTQ/AWQ 4-bit | Arm A weights | 4-bit, calibrated | Deployment-grade PTQ; showcases the quantization skill |
| E (stretch) | On-device | Arm C or D | 4-bit | Jetson Xavier NX real-hardware numbers |

**The paper is the A→B→C(→D) row comparison** on the frozen eval harness, same device, precision the only variable. REF is a check, not a paper row. E is upside.

**Why LoRA when you have 96 GB:** LoRA is the *object of study* (parameter-efficient adaptation for edge), not a compute workaround. Do **not** switch to full fine-tuning just because the card allows it — that would undercut the thesis.

---

## 2. Landmine register (read once, they're baked into the steps below)

1. **Blackwell (sm_120) vs. pinned repos — the #1 setup risk.** The repos pin torch 2.2.x + flash-attn 2.5.5, which predate your GPU entirely. PyTorch 2.7 was the first stable release with native sm_120 support; **CUDA 12.8 is the minimum**. Resolution: newer torch (cu128+), keep the repo's other pins, and **skip flash-attn**.
2. **flash-attn does not support Blackwell** cleanly (CUDA "invalid argument" on its kernels; source builds for sm_120 are fragile). Use PyTorch **SDPA** attention instead — fine for LIBERO's short sequences, and with 96 GB the memory cost is irrelevant.
3. **Base `openvla-7b` scores ~0% zero-shot on LIBERO.** Your working "baseline" is always a *fine-tuned* checkpoint. Don't debug a 0% as if the install is broken.
4. **The OFT repo needs a mandatory transformers fork** (`transformers-openvla-oft`) or its success rate silently collapses to ~0%. **Avoid this entirely by running the experiment on the base `openvla/openvla` repo**, whose standard action-tokenization path needs no fork. Still clone + read OFT for understanding.
5. **`--center_crop True` is required at eval** because the checkpoints were trained with random-crop augmentation. Forgetting it quietly tanks success rate.
6. **Device-dependent performance.** Policies can degrade when eval'd on a different GPU than trained on; the fix is to **merge the LoRA adapter into the base model on the target device** before testing. Keep the eval device fixed per comparison and log it. Expect your Blackwell numbers to differ slightly from the paper's A100 numbers — that's fine; internal consistency across your arms is what matters.
7. **Eval is stochastic.** Freeze task subset, trials-per-task, and seeds *before* running. Use 3 seeds for final numbers.
8. **MuJoCo headless rendering** (`MUJOCO_GL`) is the classic time sink — solve it with a 3-episode dummy run before touching the model.

---

## PHASE 0 — Environment up + baseline inference + repo + papers

**Goal (end of Saturday):** the released LIBERO-Spatial checkpoint returns non-zero (ideally ~84%) success in sim, logged as EXP-000. That single number means "the stack works."

### Step 0.1 — Confirm the base system

```bash
nvidia-smi                      # driver present; note driver version (want a recent one, e.g. >=570/580)
nvcc --version                  # CUDA toolkit; want >= 12.8 for sm_120
python3 --version               # target Python 3.10
```

If `nvcc` is < 12.8, install CUDA Toolkit 12.8+ (or plan to use an NVIDIA NGC PyTorch container, which ships a Blackwell-ready CUDA + torch and is the fastest path to a working stack).

### Step 0.2 — Create the environment (Blackwell-adapted)

```bash
conda create -n openvla python=3.10 -y
conda activate openvla

# Blackwell-ready PyTorch FIRST (cu128 wheels; do NOT let the repo downgrade this later).
# Check https://pytorch.org/get-started/locally/ for the current cu128+ command.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify the GPU is actually seen as sm_120 (compute capability 12.0), not silently falling back:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
# Expect something like: 2.x.x+cu128 / NVIDIA RTX PRO 6000 / (12, 0)
```

### Step 0.3 — Clone both repos (read OFT, run the base)

```bash
mkdir -p ~/vla && cd ~/vla
git clone https://github.com/openvla/openvla.git
git clone https://github.com/moojink/openvla-oft.git   # for reading/understanding only
```

### Step 0.4 — Install the base repo without clobbering torch

The repo pins `torch==2.2.*` and `flash-attn==2.5.5`. Relax the torch pin and skip flash-attn.

```bash
cd ~/vla/openvla

# Edit pyproject.toml: relax the torch pin (e.g. "torch>=2.7" or remove the ==2.2.* constraint)
# Keep the other pins as-is: transformers 4.40.1, tokenizers 0.19.1, timm 0.9.10.
pip install -e .

# If pip tried to pull torch 2.2 back in, force the Blackwell build again:
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# DO NOT install flash-attn. Instead, switch the code to SDPA attention:
grep -rn "flash_attention_2" .    # find every hardcoded reference
# Replace occurrences of  attn_implementation="flash_attention_2"  with  attn_implementation="sdpa"
# (in vla-scripts/finetune.py, experiments/robot/openvla_utils.py, and any model-load site)
```

> Note: transformers 4.40.1 works with a modern torch; the OpenVLA custom modeling code was written against 4.40.x, so **keep that pin**. The only pin you're intentionally breaking is torch, plus dropping flash-attn.

Install a **recent bitsandbytes** (older versions lack sm_120 kernels; you'll need it for the quantization arm):

```bash
pip install -U bitsandbytes
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
```

### Step 0.5 — Install LIBERO

```bash
cd ~/vla
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .

# LIBERO's extra deps for the openvla eval harness:
cd ~/vla/openvla
pip install -r experiments/robot/libero/libero_requirements.txt
```

If you hit `tensorflow-datasets` / `dlimp` errors later during data loading, the documented fixes are:
```bash
pip install tensorflow-datasets==4.9.3
pip install --no-deps --force-reinstall git+https://github.com/moojink/dlimp_openvla
```

### Step 0.6 — Solve MuJoCo headless rendering (do this before the model)

```bash
export MUJOCO_GL=egl          # GPU headless rendering; robust default
export PYOPENGL_PLATFORM=egl
# Fallback if EGL misbehaves: export MUJOCO_GL=osmesa  (CPU, slower)
```

Smoke-test the simulator alone with a tiny throwaway script that steps a LIBERO env for a few frames and renders one image. If you get an image array back, rendering works. Fix this here, not while also debugging the model.

### Step 0.7 — Baseline inference smoke test (the EXP-000 gate)

Run a *tiny* eval first (fast feedback), then scale up:

```bash
cd ~/vla/openvla
python experiments/robot/libero/run_libero_eval.py \
  --model_family openvla \
  --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --center_crop True \
  --num_trials_per_task 3        # tiny run just to confirm non-zero success
```

- Non-zero success on this tiny run → the stack works. **Log EXP-000.**
- Then run the full protocol (`--num_trials_per_task 50`, default 500 trials) and confirm you land near **84.7%**. If you do, your harness is validated (this is REF in the matrix).
- If success is ~0%: check `--center_crop True` is set, check `MUJOCO_GL`, confirm you loaded a *fine-tuned* (not base) checkpoint, and confirm SDPA is actually being used.

### Step 0.8 — Scaffold the research notebook repo (do once, now)

```
research/
├── README.md            # RQ, hypotheses, the 5 metric definitions, the experiment matrix
├── papers/              # one .md of speak-from notes per paper
├── experiments/         # one .md per experiment, using the template in §8
├── notes/               # scratch, decisions, dead ends
├── datasets/            # dataset cards + provenance (LIBERO-Spatial subset def)
├── baselines/           # REF/anchor results and configs
├── results/             # raw metric CSVs/JSON per arm
├── figures/             # plots for the paper
└── weekly-log/          # one .md per week: what ran, what failed, next
```

Put the RQ and hypotheses at the top of `research/README.md` and open it as issue #1 in the repo. Commit the empty structure now.

### Step 0.9 — Read the two core papers (speak-from notes)

- **OpenVLA** — arXiv **2406.09246**. Notes on: the 256-bin action discretization / tokenization scheme, the DINOv2+SigLIP fused vision backbone + Llama-2 LLM, why it beats prior VLAs, and the LIBERO appendix numbers.
- **OpenVLA-OFT** — arXiv **2502.19645**. Notes on: the three OFT ingredients (parallel decoding, continuous action head via L1 regression, and the input/aug changes), and *why* they yield 25–50× faster inference. This speed/success trade-off is the direct neighbor of your paper.

Save both as `research/papers/openvla.md` and `research/papers/openvla-oft.md`.

**Phase 0 done when:** EXP-000 logged with a real success number, repo scaffolded and pushed, both paper notes written.

---

## PHASE 1 — Freeze the eval harness, then run Arm A (your LoRA fine-tune)

**Order matters: freeze the measurement protocol before you train anything.**

### Step 1.1 — Freeze the eval protocol (this is your scientific control)

Write these down in `research/datasets/libero-spatial-eval.md` and never change them again:
- Task suite: `libero_spatial` (optionally a fixed subset of the 10 tasks — if subsetting, list exactly which tasks).
- Trials per task: **50** for final numbers (use 10–20 during dev for speed).
- Seeds: **3** for final numbers (`--seed`), one during dev.
- `--center_crop True`, `unnorm_key = libero_spatial_no_noops`.
- Eval device: the RTX PRO 6000 (record driver + CUDA + torch versions).

### Step 1.2 — Build the metrics harness once, reuse for every arm

Wrap the eval so that in addition to success rate it records latency, peak memory, size, and throughput (see §7 for the exact measurement recipe). Output one JSON/CSV row per arm into `research/results/`. This script is the thing that makes the arms comparable — write it carefully, then leave it alone.

### Step 1.3 — Get the LIBERO fine-tuning data

```bash
cd ~/vla   # or your base datasets dir
git clone git@hf.co:datasets/openvla/modified_libero_rlds
# Contains libero_spatial_no_noops, libero_object_no_noops, libero_goal_no_noops, libero_10_no_noops (~10 GB total)
```

### Step 1.4 — Run the LoRA r=32 fine-tune (Arm A)

Match the published recipe (r=32, lr 5e-4, effective batch 128). On one GPU, reach effective batch 128 via gradient accumulation:

```bash
cd ~/vla/openvla
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir <PATH TO modified_libero_rlds PARENT DIR> \
  --dataset_name libero_spatial_no_noops \
  --run_root_dir ./runs \
  --adapter_tmp_dir ./adapter-tmp \
  --lora_rank 32 \
  --batch_size 16 \
  --grad_accumulation_steps 8 \       # 16 x 8 = effective 128, matching the paper
  --learning_rate 5e-4 \
  --image_aug True \                  # must pair with --center_crop True at eval
  --save_steps 5000 \
  --wandb_project vla-efficiency \
  --wandb_entity <YOUR ENTITY>
```

- Confirm `libero_spatial_no_noops` is registered in `prismatic/vla/datasets/rlds/oxe/configs.py` (it should be, since the released LIBERO checkpoints used it). If not, register it per the README's dataset-integration steps.
- **No hurry** = let it run and checkpoint every 5k steps. The paper used ~50k; success typically climbs well before that. Eval intermediate checkpoints and keep the best.
- With 96 GB and SDPA you have huge memory headroom; you can raise `--batch_size` (e.g. 32, grad_accum 4) to train faster if you want.

### Step 1.5 — Merge the adapter into the base (your bf16 deployable = Arm A)

```bash
# openvla ships a merge utility; confirm exact path/flags in the repo:
python vla-scripts/merge_lora_weights_and_save.py \
  --base_checkpoint openvla/openvla-7b \
  --lora_adapter ./adapter-tmp/<your-adapter> \
  --output_dir ./merged/arm-A-bf16
```

Merge on the **same device** you'll evaluate on (landmine 6).

### Step 1.6 — Evaluate Arm A on the frozen harness

Run your metrics harness on `./merged/arm-A-bf16` with the frozen protocol. **Log EXP-001**: hypothesis (H1), config, all five metrics, what failed, next step.

**Phase 1 done when:** Arm A exists as a merged bf16 checkpoint with a full five-metric row logged, and its success rate is sane relative to REF.

---

## PHASE 2 — Quantize and fill the trade-off table (this is the paper)

Every arm here reuses Arm A's weights and the exact same frozen harness. **Only precision changes.**

### Step 2.1 — Arm B: 8-bit (bitsandbytes)

Load Arm A with `load_in_8bit=True` (bitsandbytes). If `run_libero_eval.py` exposes `--load_in_8bit`, use it; otherwise set it at the model-load site. Run the harness. **Log EXP-002.**

### Step 2.2 — Arm C: 4-bit / NF4 (bitsandbytes)

Same, with `load_in_4bit=True` (NF4). Run the harness. **Log EXP-003.** Record the success-rate degradation vs. Arm A explicitly — this delta is the heart of the answer.

### Step 2.3 — Arm D (stretch): GPTQ or AWQ 4-bit

Calibrated PTQ (AutoGPTQ / GPTQModel, or AutoAWQ) targeting the LLM linear layers of the Llama backbone. This produces a genuinely compressed on-disk artifact and is more deployment-representative than bitsandbytes — the strongest showcase of your Shothik quantization background. More fiddly with the custom OpenVLA architecture, hence stretch. **Log EXP-004.**

### Step 2.4 — Assemble the results table

Fill this from `research/results/`:

| Arm | Precision | Success (%) | Latency (ms) | Peak mem (GB) | Size (GB) | Throughput (act/s) | Δ success vs A |
|---|---|---|---|---|---|---|---|
| A | bf16 | | | | | | 0 (ref) |
| B | int8 | | | | | | |
| C | int4 | | | | | | |
| D | GPTQ/AWQ 4-bit | | | | | | |

**This table answers the RQ.** A clean sim-only version of it is a complete paper.

**Phase 2 done when:** the table is filled for A/B/C (D optional), each backed by a logged experiment and raw results file.

---

## PHASE 3 — Write-up scaffold + repo polish (still a complete paper without Jetson)

- Draft the paper around the table while it's fresh: intro (the edge-VLA cost problem), method (LoRA + PTQ), setup (LIBERO-Spatial, frozen harness, hardware), results (the table + the degradation curve), discussion (which precision is the sweet spot).
- Make a **degradation plot**: success rate vs. model size (and vs. latency) across arms → `research/figures/`.
- Clean the repo README: problem → method → the LIBERO table → reproduction steps → (later) Jetson notes + demo video.
- **Record sim demo footage now** (successful rollouts, side-by-side precisions) so you're not scrambling later.

---

## PHASE 4 — Jetson Xavier NX (stretch, contingent on your "maybe")

Only after A–C are done and logged.

- Xavier NX is memory-bound (8 GB or 16 GB variant). A 7B VLA won't run natively — **4-bit is effectively mandatory**, and *that constraint is the result*, not a failure.
- Merge/prepare the policy **on-device** (landmine 6). Expect a deployment path via 4-bit weights + a runtime that supports ARM/Jetson (e.g. a TensorRT-LLM or llama.cpp-style route); bitsandbytes on ARM is historically painful, so budget time or pick the runtime deliberately.
- Record the **same five metrics** on-device (Arm E). If it won't fit, document exactly what you tried and where it broke — that's a legitimate experimental result about the cost of edge deployment.
- Capture the **60–90 s demo video**; upload to YouTube unlisted; embed in the README.

**The dev-card numbers (A–C) are the controlled comparison; Jetson is real-hardware validation of the same relative trade-off.** A simulation-only result is already a complete paper; the Jetson is upside, never a dependency.

---

## PHASE 5 — Ship the repo + lock the first paper

- Make the repo **public** with a clean README: the problem, the method, the LIBERO numbers in a table, the Jetson notes (if done), and the embedded demo video.
- **First paper = this VLA/edge-efficiency write-up.** It reinforces the whole narrative (mechanical → rover → perception → Jetson → VLA → robot learning); your Shothik pruning/quantization background is the method, the Jetson is the testbed.
- **Do not** lead with the prompt-injection paper or the Bangla-voice benchmark — keep them as secondary tracks for *after* this one is submitted, so the story doesn't branch into AI security.

---

## 6. Reading list, mapped to purpose

| Paper | Why | Where notes go |
|---|---|---|
| OpenVLA (2406.09246) | Core method | `papers/openvla.md` |
| OpenVLA-OFT (2502.19645) | The speed/success trade-off next door | `papers/openvla-oft.md` |
| Octo | VLA method context / baseline in the LIBERO table | `papers/octo.md` |
| One off-road/driving VLA (AutoVLA **or** the ORAD-3D dataset paper) | **For talking to Wang about her domain** — not generic manipulation | `papers/offroad-vla.md` |
| Zhang, "System design and control of an apple harvesting robot" (Mechatronics, 2021) | **His signature work** — gives you something specific of his to reference | `papers/zhang-apple-harvest.md` |

---

## 7. Metrics measurement recipe (use identically for every arm)

```python
import torch, time

# --- Peak memory ---
torch.cuda.reset_peak_memory_stats()
# ... run one full rollout / batch of predict_action calls ...
peak_gb = torch.cuda.max_memory_allocated() / 1e9
# also record nvidia-smi total (includes CUDA context / non-torch allocations)

# --- Latency (warm up first!) ---
for _ in range(10):        # warmup
    _ = vla.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
torch.cuda.synchronize()
times = []
for _ in range(100):
    t0 = time.perf_counter()
    _ = vla.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)
    torch.cuda.synchronize()
    times.append((time.perf_counter() - t0) * 1000)  # ms
latency_ms = sorted(times)[len(times)//2]            # median
throughput = 1000.0 / latency_ms                     # actions/sec
```

- **Model size:** `du -sh <checkpoint_or_merged_dir>` (for quantized, the actual saved artifact).
- **Success rate:** straight from `run_libero_eval.py`.
- Record **all five** into one row per arm in `research/results/<arm>.json`. Same GPU, same driver/CUDA/torch versions, logged every time.

---

## 8. Experiment-log template (copy per experiment into `research/experiments/`)

```markdown
# EXP-XXX — <short title>

- **Date:**
- **Hypothesis:** (e.g. H2 — 8-bit preserves success within noise)
- **Dataset / eval set:** libero_spatial (frozen: 50 trials/task, seeds {…}, center_crop True)
- **Model / arm:** (REF / A / B / C / D / E)
- **Hyperparameters:** lora_rank, lr, batch, grad_accum, steps, precision, unnorm_key
- **Hardware:** RTX PRO 6000 96GB | driver x | CUDA x | torch x | bitsandbytes x
- **Result (5 metrics):** success %, latency ms, peak mem GB, size GB, throughput act/s
- **What failed / surprised me:**
- **Next experiment:**
```

---

## 9. Definition of done / deliverables checklist

- [ ] EXP-000: released checkpoint runs in sim (~84.7% reproduced) → harness validated
- [ ] Research repo scaffolded, RQ + hypotheses + metric definitions committed
- [ ] Paper notes: OpenVLA, OpenVLA-OFT (+ Octo, off-road VLA, Zhang apple-harvest)
- [ ] Frozen eval protocol written down and never changed
- [ ] Metrics harness (5 metrics) built once and reused
- [ ] Arm A: own LoRA fine-tune, merged bf16, logged (EXP-001)
- [ ] Arm B (int8) + Arm C (int4) logged (EXP-002/003)
- [ ] Results table filled (A/B/C), degradation recorded
- [ ] Arm D GPTQ/AWQ (stretch) logged
- [ ] Arm E Jetson (stretch) logged, or a documented "what it took to fit" result
- [ ] Degradation figure(s) produced
- [ ] Demo video (60–90 s) recorded + uploaded unlisted + embedded
- [ ] Repo public with clean README
- [ ] Paper drafted around the table
- [ ] Weekly-log kept every week

---

## 10. Quick weekly rhythm (no deadlines, just cadence)

- **Weekend 1:** Phase 0 (env + baseline + repo + core papers).
- **Weekend 2–3:** Phase 1 (freeze harness, Arm A fine-tune + merge + eval).
- **Weekend 3–4:** Phase 2 (quantize, fill table). *← paper is complete here.*
- **Weekend 4–5:** Phase 3 (write-up scaffold, figures, sim demo footage).
- **Weekend 5+:** Phase 4 (Jetson, if you do it) + Phase 5 (ship + paper).

End every week with a `weekly-log/` entry: what ran, what failed, what's next. That habit is what lets you write the paper in six months.

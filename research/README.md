# VLA Efficiency Project

**Working title:** *Parameter-efficient adaptation and quantization of vision-language-action policies for resource-constrained robots.*

This is a controlled experiment, not a deployment demo. Every step feeds one research question and one results table.

---

## Research question

> **RQ:** How much can parameter-efficient fine-tuning (LoRA) plus post-training quantization (PTQ) reduce VLA inference cost — latency, peak memory, model size — while preserving manipulation success rate on a controlled LIBERO task?

## Hypotheses (registered before results exist)

- **H1** — LoRA (r=32) fine-tuning recovers most of the full fine-tuned success rate on LIBERO-Spatial (target: within a few points of the published ~84.7%).
- **H2** — 8-bit PTQ preserves success rate within noise while cutting peak memory and model size substantially.
- **H3** — 4-bit PTQ produces a measurable but bounded success-rate drop; size/latency/memory savings are large enough to make the trade-off worth it for edge hardware.
- **H4 (stretch)** — The quantized policy fits and runs on Jetson Xavier NX; the cost of making a 7B VLA fit on 8–16 GB is itself a reportable result.

## The five metrics (defined once, measured identically for every arm)

| Metric | Definition | How measured |
|---|---|---|
| Success rate (%) | Fraction of successful rollouts on the frozen eval set | `run_libero_eval.py` output (successes / trials) |
| Latency (ms) | Median wall-clock per `predict_action` call, warmed up | `torch.cuda.synchronize()` + timer over N≥100 calls |
| Peak memory (GB) | Peak GPU memory during inference | `torch.cuda.max_memory_allocated()` + `nvidia-smi` for context |
| Model size (GB) | On-disk size of the deployable artifact | `du -sh` of checkpoint / merged model |
| Throughput (act/s) | Actions (or action-chunks) per second | `1000 / latency_ms`, or rollout timing |

**Anchor number:** published OpenVLA LIBERO-Spatial result is **84.7% ± 0.9%** (avg of 3 seeds × 500 rollouts). When the eval harness runs the *released* `openvla-7b-finetuned-libero-spatial` checkpoint and returns ~84–85%, the harness is trustworthy. That gate must pass before trusting the harness on our own weights.

---

## Experiment matrix

Weights are held fixed; the controlled comparison varies **only precision**.

| ID | Arm | Weights | Precision | Purpose |
|---|---|---|---|---|
| REF | Reference | Released `openvla-7b-finetuned-libero-spatial` | bf16 | Sanity-check harness against 84.7% |
| A | Baseline (our LoRA) | Our r=32 LoRA fine-tune, merged into base | bf16 | Honest full-precision point; proves the pipeline |
| B | int8 | Arm A weights | int8 (bitsandbytes) | Cheap PTQ point |
| C | int4 | Arm A weights | int4 / NF4 (bitsandbytes) | Aggressive PTQ point |
| D (stretch) | GPTQ/AWQ 4-bit | Arm A weights | 4-bit, calibrated | Deployment-grade PTQ |
| E (stretch) | On-device | Arm C or D | 4-bit | Jetson Xavier NX real-hardware numbers |

The paper is the **A→B→C(→D)** row comparison on the frozen harness, same device, precision the only variable. REF is a check, not a paper row. E is upside.

**Why LoRA when the dev card has 96 GB:** LoRA is the *object of study* (parameter-efficient adaptation for edge), not a compute workaround. Do not switch to full fine-tuning.

---

## Hardware & schedule reality

- **Office box — RTX PRO 6000 Blackwell, 96 GB, Linux.** All GPU work runs here. Fully dedicated, persistent disk, jobs may run unattended overnight/weekend. Access: Sun–Thu 9–6 for interactive work; long trains bridge the Fri–Sat gap.
- **Home box — RTX 2060, 6 GB.** Too small for any 7B arm. Used off-hours for repo, notes, plots, paper draft, weekly-log. No GPU inference here.
- **Stretch target — Jetson Xavier NX** (8/16 GB), Phase 4 only.

**Blackwell landmines (office box only):** torch must be cu128+ (PyTorch 2.7 first stable sm_120; CUDA 12.8 minimum); skip flash-attn (no clean Blackwell support) — use SDPA attention; recent bitsandbytes for sm_120 kernels.

**Non-negotiables at eval:** `--center_crop True` (checkpoints trained with random-crop aug), `unnorm_key = libero_spatial_no_noops`, base `openvla/openvla` repo (avoids the OFT transformers-fork trap), fixed eval device logged per comparison.

---

## Repo layout

```
research/
├── README.md            # this file — RQ, hypotheses, metrics, matrix
├── papers/              # speak-from notes, one .md per paper
├── experiments/         # one .md per experiment (EXP-template.md)
├── notes/               # scratch, decisions, dead ends
├── datasets/            # dataset cards + provenance, frozen eval protocol
├── baselines/           # REF / anchor results and configs
├── results/             # raw metric CSVs/JSON per arm
├── figures/             # plots for the paper
└── weekly-log/          # one .md per week: what ran, what failed, next
```

## Definition of done

- [ ] EXP-000: released checkpoint reproduces ~84.7% in sim → harness validated
- [ ] RQ + hypotheses + metric definitions committed (this README)
- [ ] Paper notes: OpenVLA, OpenVLA-OFT (+ Octo, off-road VLA, Zhang apple-harvest)
- [ ] Frozen eval protocol written and never changed
- [ ] Metrics harness (5 metrics) built once and reused
- [ ] Arm A: own LoRA fine-tune, merged bf16, logged (EXP-001)
- [ ] Arm B (int8) + Arm C (int4) logged (EXP-002/003)
- [ ] Results table filled (A/B/C), degradation recorded
- [ ] Arm D GPTQ/AWQ (stretch), Arm E Jetson (stretch)
- [ ] Degradation figure(s) produced
- [ ] Demo video recorded + uploaded + embedded
- [ ] Repo public with clean README, paper drafted around the table

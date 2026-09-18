# Product Context

## Why this exists
7B VLAs work in sim but are too heavy for resource-constrained robots (Jetson-class memory). Need measured trade-off: LoRA adaptation + PTQ vs success rate — not vibes.

## Problem it solves
No clean, controlled numbers for: same OpenVLA weights, same LIBERO-Spatial eval, only precision changes — success / latency / peak mem / size / throughput.

## How it should work
1. Validate harness on released LIBERO-Spatial checkpoint (REF → ~84.7%).
2. Fine-tune own LoRA r=32 policy; merge to bf16 (Arm A).
3. Quantize Arm A to int8 / int4 (B/C); optional GPTQ/AWQ (D); optional Jetson (E).
4. Measure five metrics identically every arm; fill trade-off table.
5. Write paper around table; ship public repo + demo.

## User experience goals (researcher workflow)
- Office GPU (RTX PRO 6000 96GB) for all heavy work; home RTX 2060 for notes/plots only.
- Experiment logs (`EXP-XXX`) + weekly-log habit so paper writeable months later.
- Freeze eval protocol before training — scientific control first.

## Non-goals for UX
- Flashy robot demo before table exists.
- Switching to OFT transformers fork for the main run path.

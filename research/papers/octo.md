# Octo — speak-from notes

- **Cite:** Ghosh, Walke, Pertsch et al., "Octo: An Open-Source Generalist Robot Policy." arXiv **2405.12213**. Site: octo-models.github.io.
- **One-liner:** open, transformer-based **generalist** manipulation policy trained on 800k Open X-Embodiment trajectories; instructable by language *or* goal-image, fine-tunable to new robots in hours on consumer GPUs. A LIBERO-table neighbor of OpenVLA at a different point on the size/capability curve.

## Architecture

- **Large transformer policy** (not LLM-backed). Multimodal: takes language commands *or* goal images.
- **Action generation via a diffusion head** (diffusion decoding of continuous action chunks) — contrast with OpenVLA's discrete 256-bin token decode ([[openvla]]). Different mechanism, different quantization story.
- Param count: not in abstract — **fill from paper body** (Octo comes in ~27M / ~93M "small"/"base" sizes; confirm before quoting).

## Training data

- **800k trajectories** from **Open X-Embodiment** (vs OpenVLA's 970k — same source family, so a fair architecture-vs-scale contrast).

## Key claims

- Fine-tunes to new sensory inputs / action spaces "within a few hours on standard consumer GPUs."
- Serves as a versatile **policy initialization** across 9 robot platforms.

## Why it matters for our RQ

- Octo is the **non-LLM, diffusion-head** baseline. Useful framing: our edge-efficiency thesis (LoRA+PTQ on a 7B LLM-backed VLA) sits opposite Octo's "make the base model small" approach. If asked "why not just use a smaller policy like Octo?" — because OpenVLA's LLM backbone buys generality/success (see its +16.5% over RT-2-X, +20% over Diffusion Policy), and our contribution is *keeping that* while cutting deploy cost via precision, not shrinking the architecture.

## To verify on office box / from body

- [ ] Exact param counts (small vs base).
- [ ] Octo's LIBERO numbers as they appear in the OpenVLA / OFT comparison tables.
- [ ] Diffusion-head details (denoising steps → its own latency cost, relevant if comparing inference cost).

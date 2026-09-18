# OpenVLA — speak-from notes

- **Cite:** OpenVLA: An Open-Source Vision-Language-Action Model. arXiv **2406.09246**.
- **One-liner:** 7B open VLA that beats a 55B closed model (RT-2-X) on manipulation, fine-tunable on consumer GPUs via LoRA, servable via quantization. Directly names *our* two methods (LoRA + quantization) as its efficiency story — this paper is the spine of the project.

## Architecture (from abstract + paper body)

- **7B parameters** total.
- **Vision backbone:** fused features from **two** pretrained encoders — **DINOv2** (spatial/geometry) + **SigLIP** (semantic/language-aligned). Fusing both is a deliberate choice; single-encoder ablations are weaker.
- **LLM base:** **Llama 2** (7B).
- Vision features are projected into the language model's token space and the LLM autoregressively emits action tokens.

## Action representation / tokenization (paper body — verify exact line on office box)

- Robot action = continuous 7-DoF vector (Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper).
- Each dimension is **discretized into 256 bins** → one token per dimension. Bin edges set by per-dimension quantiles of the training data (robust to outliers vs uniform min/max).
- The 256 action tokens **overwrite the 256 least-used tokens** in the Llama vocabulary, so no vocab resize is needed and generation stays plain autoregressive decoding.
- **Consequence for our project:** base OpenVLA generates 7 tokens sequentially per action → this sequential decode is exactly the latency that OFT (next paper) attacks with parallel decoding, and the tokenized head is why bitsandbytes PTQ (Arms B/C) applies cleanly to the Llama linears.

## Training data

- **970k real-world robot demonstrations**, drawn from the **Open X-Embodiment** collection (many robots/embodiments).

## Headline results (from abstract)

- Beats **RT-2-X (55B)** by **+16.5%** absolute task success across **29 tasks** / multiple embodiments — with **7× fewer parameters**.
- Beats from-scratch imitation (**Diffusion Policy**) by **+20.4%** on the tasks tested.
- Explicitly: fine-tunable on consumer GPUs via **LoRA**, servable via **quantization** — the paper's own efficiency claim we are stress-testing quantitatively.

## LIBERO (the numbers we anchor to)

- Abstract does not give LIBERO figures; they are in the fine-tuning / appendix section — **pull exact table on office box.**
- Project anchor (from plan): released `openvla-7b-finetuned-libero-spatial` ≈ **84.7% ± 0.9%** (3 seeds × 500 rollouts). REF arm must reproduce this.
- **Landmine:** base `openvla-7b` scores ~**0%** zero-shot on LIBERO — always use a *fine-tuned* checkpoint; do not debug a 0% as a broken install.

## Why it matters for our RQ

- OpenVLA *asserts* LoRA + quantization make VLAs deployable but does not table the success/latency/memory/size trade-off across precisions on one frozen harness. **That empty table is our contribution.** H1 tests whether our own LoRA reproduces the anchor; H2/H3 quantify what OpenVLA only claims qualitatively.

## To verify on office box (paper body)

- [ ] Exact bin count / quantile scheme wording (256 confirmed from body — confirm quantile detail).
- [ ] Exact LIBERO-Spatial/Object/Goal/Long table for openvla-7b fine-tuned.
- [ ] Whether published LIBERO used full FT or LoRA, and the r / lr / batch recipe (plan §1.4: r=32, lr 5e-4, eff. batch 128).

# OpenVLA-OFT — speak-from notes

- **Cite:** Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success (OpenVLA-OFT). arXiv **2502.19645**.
- **One-liner:** an *Optimized Fine-Tuning* recipe that makes OpenVLA both far faster and far more accurate at inference. It is the **direct neighbor** of our paper — it trades along the speed/success axis via *architectural* changes; we trade along the size/memory axis via *precision* changes. Same benchmark (LIBERO), complementary lever.

## The three OFT ingredients (from abstract + body)

1. **Parallel decoding + action chunking.**
   - Base OpenVLA decodes action tokens **autoregressively, one at a time** (7 sequential forward passes per action). OFT predicts a whole **chunk of future actions in parallel** in a single pass.
   - This is the main throughput win: replaces N sequential decode steps with one, removing the per-token latency that dominates base inference.
2. **Continuous action representation + L1 regression objective.**
   - Drops the 256-bin discrete tokenization (see [[openvla]] notes) in favor of a **continuous action head trained with simple L1 regression**.
   - No de-tokenization, no bin quantization error; smooth actions and a cheaper head. Pairs naturally with parallel/chunked prediction.
3. **Input/output flexibility changes.**
   - Improvements to input specification and augmentation (the paper's "flexibility in input-output specifications"). Confirm exact list (e.g. multi-view / proprio inputs, aug settings) **on office box**.

## Results (from abstract)

- **Inference throughput: ~26× increase** in action generation vs base OpenVLA.
- **LIBERO average success:** base OpenVLA **76.5%** → OpenVLA-OFT **97.1%** (avg across the four suites).
- Per-suite table (Spatial / Object / Goal / Long) — **pull exact numbers on office box**; note our project's 84.7% anchor is the *Spatial* suite for the specific released fine-tuned checkpoint, so cross-check which recipe each number corresponds to before comparing.

## Why the speedup (mechanism)

- Autoregressive token decoding = latency scales with tokens-per-action and is bandwidth/sequential-bound. Parallel decoding collapses that to one pass; action chunking amortizes one inference over several control steps; continuous head removes tokenization overhead entirely. Compounded → the ~26× figure.

## Relationship to our project (important framing)

- OFT attacks latency by **changing the model** (new head, parallel decode). We attack memory/size by **changing precision** (PTQ) while **holding weights fixed** — a cleaner controlled comparison for the size/memory/success trade-off.
- **The two are orthogonal and composable:** an OFT-style policy could also be quantized. We deliberately run on the **base `openvla/openvla` repo, not the OFT repo**, because the OFT repo needs a mandatory `transformers-openvla-oft` fork or success silently collapses to ~0% (landmine 4). Clone + read OFT for understanding only.
- Good discussion-section point: our PTQ trade-off is measured on the *original tokenized* policy; noting that OFT's continuous head would interact differently with quantization is honest future work.

## To verify on office box (paper body)

- [ ] Exact per-suite LIBERO numbers, base vs OFT.
- [ ] Exact chunk size (actions per parallel decode) and how latency was measured (compare to our §7 recipe for apples-to-apples).
- [ ] Full list of the "input/output flexibility" changes.
- [ ] Whether the 26× is on the same GPU class we'll use (device-dependent — landmine 6).

# Project Brief

## Name
`openvla-libero-benchmark` — VLA efficiency study on LIBERO-Spatial.

## Working paper title
*Parameter-efficient adaptation and quantization of vision-language-action policies for resource-constrained robots.*

## One-line framing
Controlled experiment, not deployment demo. Every step feeds one RQ and one results table.

## Research question
How much can LoRA + PTQ reduce VLA inference cost (latency, peak memory, model size) while preserving manipulation success rate on a controlled LIBERO task?

## Hypotheses
- **H1:** LoRA r=32 recovers most of published LIBERO-Spatial success (~84.7%).
- **H2:** int8 PTQ preserves success within noise; big memory/size cuts.
- **H3:** int4 PTQ has bounded success drop; savings worth it for edge.
- **H4 (stretch):** Quantized policy runs on Jetson Xavier NX; fit cost is reportable.

## In scope
- Base `openvla/openvla` (not OFT runtime path).
- Own LoRA fine-tune on LIBERO-Spatial (Arm A), then PTQ arms B/C (D stretch).
- Frozen eval harness + five metrics for every arm.
- Sim-only A→B→C table = complete paper.

## Out of scope (for this paper)
- Full fine-tune instead of LoRA (undercuts thesis).
- Leading with prompt-injection / Bangla-voice tracks.
- Jetson as a dependency (stretch only).

## Success = done when
- REF harness ≈ 84.7% on released checkpoint (EXP-000).
- Arms A/B/C logged with five metrics; degradation table filled.
- Paper drafted around that table; repo public with README + demo.

## Source of truth
- `plan/VLA-project-plan.md` — full execution plan.
- `research/README.md` — RQ, hypotheses, metrics, matrix.

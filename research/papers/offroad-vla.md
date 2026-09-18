# AutoVLA (driving VLA) — speak-from notes

Chosen over ORAD-3D: AutoVLA is a full VLA (matches our object of study), driving domain for **Wang**.

- **Cite:** "AutoVLA: A Vision-Language-Action Model for End-to-End Autonomous Driving with Adaptive Reasoning and Reinforcement Fine-Tuning." arXiv **2506.13757**.
- **One-liner:** end-to-end driving VLA — single autoregressive model emits both language reasoning *and* a trajectory plan from vision + instruction. VLA framing lifted from tabletop manipulation to a vehicle.

## What it does / architecture

- **Unified autoregressive model:** reasoning + action in one generation pass (vs multi-component perception→plan→control stacks).
- **Discrete action tokenization:** continuous trajectories → discrete feasible action tokens the LLM emits — **same core trick as OpenVLA's 256-bin tokenizer** ([[openvla]]). Strong talking point: the tokenized-action idea generalizes across domains.
- **Dual thinking modes:** fast (trajectory-only) vs slow (chain-of-thought). **Adaptive reasoning** picks per-scene → cuts compute on easy scenes.
- **Reinforcement fine-tuning** via **GRPO** to suppress unnecessary reasoning.

## Datasets / benchmarks

- nuPlan, nuScenes, Waymo, CARLA. Open-loop + closed-loop eval.

## Angle for talking to Wang (her domain)

- Bridge: VLA + tokenized actions transfer manipulation → driving; the open question on a vehicle is **compute budget**. AutoVLA's *adaptive reasoning* cuts cost by skipping thinking; **our lever is orthogonal — cut cost by precision (LoRA+PTQ) without changing what the model computes.** The two compose: a driving VLA could be both adaptively-reasoned *and* quantized. That's the natural "your domain × my method" pitch.
- Action-space contrast to prep: manipulation = 7-DoF Δpose+gripper; driving = future trajectory waypoints. Both tokenized; eval differs (task success vs open/closed-loop driving metrics).

## To verify / fill from body

- [ ] Backbone LLM + param count.
- [ ] How much adaptive reasoning saves (latency / token count).
- [ ] Whether any quantization/edge deployment is discussed (likely not → our gap).

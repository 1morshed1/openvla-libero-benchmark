# Active Context

## Current focus
Phase 0 start: memory-bank created; research scaffold dirs exist; full execution plan in `plan/VLA-project-plan.md`.

## Recent changes
- Repo scaffold: `research/` (empty subdirs + README), `plan/VLA-project-plan.md`.
- `memory-bank/` core files created (2026-09-18).
- Root `CLAUDE.md` added — agent rules + landmines + phase map.

## Next steps (Phase 0)
1. Confirm office box: `nvidia-smi`, `nvcc` ≥12.8, Python 3.10.
2. Conda env + Blackwell torch cu128+; verify sm_120.
3. Clone openvla + openvla-oft (read) + LIBERO; install openvla with relaxed torch, SDPA, no flash-attn; recent bitsandbytes.
4. Fix MuJoCo EGL smoke test (dummy env render).
5. Tiny then full REF eval → log EXP-000 (~84.7% gate).
6. Paper notes: OpenVLA + OpenVLA-OFT speak-from md files.
7. Commit/push scaffold; open RQ as issue #1 if desired.

## Active decisions
- Run base openvla, not OFT fork path.
- LoRA is object of study despite 96 GB card.
- Sim A–C table = complete paper; Jetson optional.

## Open questions / blockers
- Office env not yet verified in this workspace (GPU/CUDA state unknown from repo alone).
- External `~/vla` clones may not exist yet on this machine.
- research/README present; paper notes and EXP logs still empty.

## Considerations
Favor correctness + logging over speed. No deadline pressure — weekend cadence in plan §10.

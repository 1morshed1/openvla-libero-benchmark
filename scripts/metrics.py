#!/usr/bin/env python3
"""Metrics harness — measure the 5 project metrics identically for every arm.

Metrics (plan §7): success %, median latency ms, peak mem GB, on-disk size GB,
throughput act/s. Success comes from run_libero_eval.py; this script measures the
other four by timing `vla.predict_action` and writes one JSON row per arm to
research/results/<arm>.json (success merged in via --success or run_arm.sh).

Build once, reuse unchanged for REF/A/B/C so arms stay comparable. Same GPU,
same driver/CUDA/torch/bnb versions, logged every run.

Precision:
  bf16  — REF and Arm A (merged LoRA)
  int8  — Arm B (bitsandbytes load_in_8bit)
  int4  — Arm C (bitsandbytes NF4)

Landmines honored: attn_implementation='sdpa' (no flash-attn on Blackwell),
center_crop handled at eval (not here), unnorm_key required.

NOTE: predict_action signature / prompt format mirror the openvla eval harness.
Verify against experiments/robot/openvla_utils.py on the office box the first run.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# openvla instruction prompt (matches openvla eval harness).
PROMPT_TMPL = "In: What action should the robot take to {instr}?\nOut:"
DEFAULT_INSTR = "pick up the black bowl and place it on the plate"


def _run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def env_versions():
    """Record hardware/software provenance — logged every run (plan requirement)."""
    info = {}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["capability"] = list(torch.cuda.get_device_capability(0))
    except Exception as e:
        info["torch_error"] = repr(e)
    try:
        import bitsandbytes as bnb
        info["bitsandbytes"] = bnb.__version__
    except Exception:
        info["bitsandbytes"] = None
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        pass
    info["driver"] = _run("nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits")
    info["nvcc"] = _run("nvcc --version | grep release") or None
    return info


def dir_size_gb(path):
    """On-disk size of the checkpoint artifact (du -sb). None for a bare HF hub id."""
    p = Path(path)
    if not p.exists():
        return None
    out = _run(f"du -sb {p}")
    if not out:
        return None
    try:
        return round(int(out.split()[0]) / 1e9, 3)
    except Exception:
        return None


def load_model(checkpoint, precision):
    """Load OpenVLA at the given precision. bf16 | int8 | int4(NF4)."""
    import torch
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)

    kwargs = dict(
        attn_implementation="sdpa",   # landmine #1/#3: never flash-attn on Blackwell
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    if precision == "bf16":
        kwargs["torch_dtype"] = torch.bfloat16
    elif precision in ("int8", "int4"):
        from transformers import BitsAndBytesConfig
        if precision == "int8":
            qcfg = BitsAndBytesConfig(load_in_8bit=True)
        else:
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        kwargs["quantization_config"] = qcfg
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        raise ValueError(f"unknown precision {precision!r}")

    vla = AutoModelForVision2Seq.from_pretrained(checkpoint, **kwargs)
    if precision == "bf16":
        vla = vla.to("cuda")   # bnb-quantized models are already device-placed
    vla.eval()
    return processor, vla


def build_inputs(processor, instr):
    import torch
    from PIL import Image
    import numpy as np
    # Dummy 256x256 image — processor center-crops/resizes. Latency is
    # input-content-independent; success rate (real images) comes from eval.
    img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    prompt = PROMPT_TMPL.format(instr=instr)
    inputs = processor(prompt, img).to("cuda", dtype=torch.bfloat16)
    return inputs


def measure(processor, vla, unnorm_key, instr, warmup=10, iters=100):
    """Median latency (ms), throughput (act/s), peak GB — §7 recipe exactly."""
    import torch
    inputs = build_inputs(processor, instr)

    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(warmup):
            _ = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    times.sort()
    latency_ms = times[len(times) // 2]
    peak_torch_gb = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    smi = _run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
    peak_smi_gb = round(int(smi) / 1000.0, 3) if smi else None
    return {
        "latency_ms_median": round(latency_ms, 2),
        "latency_ms_p10": round(times[len(times) // 10], 2),
        "latency_ms_p90": round(times[(len(times) * 9) // 10], 2),
        "throughput_act_s": round(1000.0 / latency_ms, 2),
        "peak_mem_gb_torch": peak_torch_gb,
        "peak_mem_gb_nvidia_smi": peak_smi_gb,
        "latency_iters": iters,
        "latency_warmup": warmup,
    }


def main():
    ap = argparse.ArgumentParser(description="Measure latency/mem/size for one arm.")
    ap.add_argument("--arm", required=True, help="REF|A|B|C|D (output file name)")
    ap.add_argument("--checkpoint", required=True, help="local dir or HF hub id")
    ap.add_argument("--precision", required=True, choices=["bf16", "int8", "int4"])
    ap.add_argument("--unnorm-key", default="libero_spatial_no_noops")
    ap.add_argument("--instruction", default=DEFAULT_INSTR)
    ap.add_argument("--success", type=float, default=None,
                    help="success %% from run_libero_eval.py (merged into the row)")
    ap.add_argument("--size-path", default=None,
                    help="dir to du for on-disk size (default: --checkpoint if local)")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--results-dir", default=None,
                    help="default: <repo>/research/results")
    args = ap.parse_args()

    results_dir = Path(args.results_dir) if args.results_dir else \
        Path(__file__).resolve().parent.parent / "research" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{args.arm}.json"

    row = {
        "arm": args.arm,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "precision": args.precision,
        "unnorm_key": args.unnorm_key,
        "success_rate_pct": args.success,
        "size_gb": dir_size_gb(args.size_path or args.checkpoint),
        "env": env_versions(),
    }

    # Merge into an existing row (e.g. success written first, metrics second).
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
            if row.get("success_rate_pct") is None:
                row["success_rate_pct"] = prev.get("success_rate_pct")
            if row.get("size_gb") is None:
                row["size_gb"] = prev.get("size_gb")
        except Exception:
            pass

    try:
        import torch
        assert torch.cuda.is_available(), "CUDA not available — run on the office GPU"
        processor, vla = load_model(args.checkpoint, args.precision)
        row.update(measure(processor, vla, args.unnorm_key, args.instruction,
                            warmup=args.warmup, iters=args.iters))
    except Exception as e:
        row["measure_error"] = repr(e)
        print(f"MEASURE FAILED: {e!r}", file=sys.stderr)

    out_path.write_text(json.dumps(row, indent=2) + "\n")
    print(json.dumps(row, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

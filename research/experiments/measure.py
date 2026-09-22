#!/usr/bin/env python3
"""
measure.py — five-metric harness for the OpenVLA-LIBERO efficiency experiment.

Measures, identically for every arm (A/B/C/D):
    success rate (%)      <- passed in from run_libero_eval (--success_rate)
    latency (ms)          <- median per predict_action() call
    peak memory (GB)      <- torch peak + NVML total (incl. CUDA context)
    model size (GB)       <- on-disk size of the checkpoint dir
    throughput (act/s)    <- 1000 / median_latency_ms
    energy (J/action)     <- NVML power integrated over a fixed workload  [NEW]
    avg power (W)

Design notes
------------
* Runs on GPU-1 only, matching the session constraint. Launch with:
      CUDA_VISIBLE_DEVICES=1 python measure.py --arm A ...
  Under CUDA_VISIBLE_DEVICES=1, torch sees the card as "cuda:0" but NVML uses
  *physical* indices, so NVML must read physical index 1. This script resolves
  that automatically from CUDA_VISIBLE_DEVICES (override with --nvml_index).
* Perf metrics are measured on a FIXED, deterministic observation so numbers are
  comparable across arms (compute depends on input *shape*, not content). Swap in
  a real captured LIBERO frame via --image if you want realism; results won't
  differ meaningfully for latency/energy.
* Success rate comes from the real frozen eval (run_libero_eval.py); pass it in so
  the JSON row is complete. Keeps the expensive rollout eval separate from the
  cheap perf loop.

Deps:  pip install nvidia-ml-py    (provides the `pynvml` module)
"""

import argparse, json, os, threading, time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

try:
    import pynvml
    _NVML = True
except Exception:
    _NVML = False


# ----------------------------------------------------------------------------- #
# NVML power sampler
# ----------------------------------------------------------------------------- #
def _resolve_nvml_index(override: int | None) -> int:
    """Physical NVML index. Under CUDA_VISIBLE_DEVICES=1, torch's cuda:0 == phys 1."""
    if override is not None:
        return override
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    return int(vis) if vis.isdigit() else 0


class PowerSampler(threading.Thread):
    """Samples GPU power (W) in the background; integrates to Joules (trapezoidal)."""

    def __init__(self, nvml_index: int, interval_s: float = 0.02):
        super().__init__(daemon=True)
        self.interval = interval_s
        self._stop = threading.Event()
        self.samples: list[tuple[float, float]] = []  # (t, watts)
        self.ok = _NVML
        if self.ok:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(nvml_index)

    def run(self):
        if not self.ok:
            return
        while not self._stop.is_set():
            try:
                w = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW -> W
                self.samples.append((time.perf_counter(), w))
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        self.join(timeout=2.0)

    def energy_joules(self) -> float | None:
        if len(self.samples) < 2:
            return None
        e = 0.0
        for (t0, w0), (t1, w1) in zip(self.samples, self.samples[1:]):
            e += 0.5 * (w0 + w1) * (t1 - t0)  # W * s = J
        return e

    def avg_watts(self) -> float | None:
        if not self.samples:
            return None
        return sum(w for _, w in self.samples) / len(self.samples)


# ----------------------------------------------------------------------------- #
# Model loading (precision is the only thing that changes across arms)
# ----------------------------------------------------------------------------- #
def load_policy(checkpoint: str, precision: str, device: str = "cuda:0"):
    common = dict(trust_remote_code=True, low_cpu_mem_usage=True,
                  attn_implementation="sdpa")  # <- Blackwell/SDPA path, no flash-attn
    if precision == "bf16":
        vla = AutoModelForVision2Seq.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, **common).to(device)
    elif precision == "int8":
        qc = BitsAndBytesConfig(load_in_8bit=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            checkpoint, quantization_config=qc, device_map={"": 0}, **common)
    elif precision == "int4":
        qc = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        vla = AutoModelForVision2Seq.from_pretrained(
            checkpoint, quantization_config=qc, device_map={"": 0}, **common)
    else:
        raise ValueError(f"unknown precision {precision!r} (bf16|int8|int4)")
    vla.eval()
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    return vla, processor


def dir_size_gb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / 1e9


# ----------------------------------------------------------------------------- #
# The perf loop (latency + energy + throughput + peak memory) on a fixed input
# ----------------------------------------------------------------------------- #
def measure_perf(vla, processor, unnorm_key, instruction, image,
                 nvml_index, n_warmup=10, n_iters=50, device="cuda:0"):
    # OpenVLA prompt template (verify against your patched repo if you changed it).
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(n_warmup):
        _ = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    sampler = PowerSampler(nvml_index)
    sampler.start()

    lat_ms = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _ = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        torch.cuda.synchronize()
        lat_ms.append((time.perf_counter() - t0) * 1000.0)

    sampler.stop()

    lat_ms.sort()
    median_ms = lat_ms[len(lat_ms) // 2]
    energy = sampler.energy_joules()
    nvml_mem = None
    if _NVML:
        try:
            mi = pynvml.nvmlDeviceGetMemoryInfo(sampler.handle)
            nvml_mem = mi.used / 1e9
        except Exception:
            pass

    return {
        "latency_ms_median": round(median_ms, 2),
        "latency_ms_p90": round(lat_ms[int(0.9 * len(lat_ms))], 2),
        "throughput_act_s": round(1000.0 / median_ms, 2),
        "peak_mem_torch_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "peak_mem_nvml_gb": round(nvml_mem, 3) if nvml_mem else None,
        "energy_j_per_action": round(energy / n_iters, 3) if energy else None,
        "avg_power_w": round(sampler.avg_watts(), 1) if sampler.avg_watts() else None,
        "n_iters": n_iters,
    }


# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="A | B | C | D")
    ap.add_argument("--precision", required=True, choices=["bf16", "int8", "int4"])
    ap.add_argument("--checkpoint", required=True,
                    help="e.g. ~/vla/merged/arm-A-bf16 (int8/int4 load the SAME bf16 dir)")
    ap.add_argument("--unnorm_key", default="libero_spatial_no_noops")
    ap.add_argument("--instruction", default="pick up the black bowl and place it on the plate")
    ap.add_argument("--image", default=None, help="optional real LIBERO frame; else a dummy 224x224")
    ap.add_argument("--success_rate", type=float, default=None,
                    help="from run_libero_eval on the frozen protocol (50 trials x seeds)")
    ap.add_argument("--nvml_index", type=int, default=None,
                    help="physical GPU index for NVML; auto from CUDA_VISIBLE_DEVICES")
    ap.add_argument("--n_iters", type=int, default=50)
    ap.add_argument("--out_dir", default="research/results")
    args = ap.parse_args()

    ckpt = os.path.expanduser(args.checkpoint)
    nvml_index = _resolve_nvml_index(args.nvml_index)
    image = (Image.open(os.path.expanduser(args.image)).convert("RGB")
             if args.image else Image.new("RGB", (224, 224), (127, 127, 127)))

    print(f"[measure] arm={args.arm} precision={args.precision} "
          f"nvml_index={nvml_index} nvml={'on' if _NVML else 'OFF (pip install nvidia-ml-py)'}")

    vla, processor = load_policy(ckpt, args.precision)
    perf = measure_perf(vla, processor, args.unnorm_key, args.instruction, image,
                        nvml_index, n_iters=args.n_iters)

    row = {
        "arm": args.arm,
        "precision": args.precision,
        "checkpoint": ckpt,
        "success_rate_pct": args.success_rate,     # fill from run_libero_eval
        "model_size_gb": round(dir_size_gb(ckpt), 3),
        **perf,
        "device": torch.cuda.get_device_name(0),
        "unnorm_key": args.unnorm_key,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    os.makedirs(os.path.expanduser(args.out_dir), exist_ok=True)
    out = os.path.join(os.path.expanduser(args.out_dir), f"arm-{args.arm}-{args.precision}.json")
    with open(out, "w") as f:
        json.dump(row, f, indent=2)
    print(json.dumps(row, indent=2))
    print(f"[measure] wrote {out}")


if __name__ == "__main__":
    main()

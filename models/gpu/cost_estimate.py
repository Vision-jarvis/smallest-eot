"""GPU cost estimate for the Smart Turn Hindi fine-tune, from measured numbers where possible.

    python -m models.gpu.cost_estimate [--n-train 12000]

Measured here: training FLOPs per sample (torch FlopCounterMode on the real model, forward x3
for forward+backward) and log-mel + augmentation time per sample on this CPU (emulated x86, so
pessimistic). Assumed (stated in the output): achieved GPU throughput as a fraction of the
L4's dense bf16 peak, container start-up and data upload time. Prices: modal.com/pricing,
fetched 2026-10-20. The Modal function's timeout is the hard ceiling on spend.
Writes results/gpu_cost_estimate.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
L4_BF16_DENSE_TFLOPS = 121.0  # NVIDIA L4 datasheet: 242 TFLOPS bf16 with sparsity, 121 dense
MFU = 0.15                    # assumed achieved fraction for a small 4-layer encoder at batch 64
STARTUP_H = 0.15              # image build/pull + volume mount + model load (assumed)
CPU_PRICE, MEM_PRICE = 0.04716, 0.007992


def flops_per_sample() -> float:
    from torch.utils.flop_counter import FlopCounterMode

    from models.gpu.smart_turn_torch import SmartTurnV3

    m = SmartTurnV3().eval()
    with FlopCounterMode(display=False) as fc:
        m(torch.zeros(1, 80, 800))
    return float(fc.get_total_flops())


def logmel_seconds_per_sample(n: int = 20) -> float:
    from core.audio import mulaw_roundtrip
    from core.encoders import log_mel

    x = np.random.default_rng(0).standard_normal(8 * 16000).astype(np.float32) * 0.05
    t = time.perf_counter()
    for _ in range(n):
        log_mel(mulaw_roundtrip(x, 16000), normalise=True)
    return (time.perf_counter() - t) / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path(__file__).with_name("finetune_smart_turn_hi.yaml"))
    ap.add_argument("--n-train", type=int, default=None, help="training windows (default: Chirp3 + human slice + 4 x 6000 IndicVoices pauses)")
    a = ap.parse_args()
    cfg = yaml.safe_load(a.config.read_text())
    c, tr = cfg["cost"], cfg["train"]
    n_train = a.n_train or (2196 + 1500 + 4 * 6000 * 7 // 10)  # train split gets 4 offsets per pause, 70% of pauses
    n_dev = 6000 // 10 + 1165
    f_fwd = flops_per_sample()
    samples = tr["epochs"] * n_train + (tr["epochs"] + 1) * n_dev  # train + one dev pass per epoch and at epoch 0
    gpu_s = (tr["epochs"] * n_train * 3 * f_fwd + (tr["epochs"] + 1) * n_dev * f_fwd) / (L4_BF16_DENSE_TFLOPS * 1e12 * MFU)
    mel = logmel_seconds_per_sample()
    data_s = samples * mel / c["cpu_cores"]
    run_h = max(gpu_s, data_s) / 3600 + STARTUP_H
    rate = c["usd_per_hour"] + c["cpu_cores"] * CPU_PRICE + c["mem_gib"] * MEM_PRICE
    out = {
        "gpu": c["gpu"], "usd_per_hour_all_in": rate,
        "measured": {"train_flops_per_sample": 3 * f_fwd, "forward_gflops_per_sample": f_fwd / 1e9,
                     "logmel_plus_mulaw_ms_per_sample_this_cpu": 1000 * mel},
        "assumed": {"mfu": MFU, "peak_tflops": L4_BF16_DENSE_TFLOPS, "startup_h": STARTUP_H, "n_train": n_train, "n_dev": n_dev,
                    "epochs": tr["epochs"]},
        "gpu_compute_h": gpu_s / 3600, "cpu_data_h": data_s / 3600, "bottleneck": "cpu data loading" if data_s > gpu_s else "gpu",
        "estimated_run_h": run_h, "estimated_usd": run_h * rate,
        "ceiling_usd_at_timeout": c["timeout_h"] * rate,
        "with_two_restarts_usd": 3 * run_h * rate,
    }
    p = ROOT / "results/gpu_cost_estimate.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

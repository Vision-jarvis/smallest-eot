"""Single-core efficiency benchmark (points-matrix row 7).

    python -m eval.bench_cpu --detector pulse_eot_hi --turns 40 --calls 50

* ms per 100 ms hop: one process, onnxruntime pinned to 1 intra-op thread, every hop scored
  (need=True), Vosk partials replayed from the cache. Includes VAD, prosody, lexical head and
  the ONNX hazard head; excludes the ASR itself (Pulse runs server-side).
* calls per core = 100 ms / mean ms per hop.
* memory: RSS of one process holding `--calls` independent detector states (concurrent calls)
  after each has consumed a turn.
* model size: bytes of the shipped artifacts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from eval import asr_cache
from eval.data import load_eot_bench_hi
from eval.harness import replay
from eval.registry import make

ROOT = Path(__file__).resolve().parents[1]


def rss_mb() -> float:
    import psutil

    return psutil.Process(os.getpid()).memory_info().rss / 2**20


def artifact_mb(detector: str) -> float:
    if detector.startswith("pulse_eot_hi"):
        variant = detector.split(":", 1)[1] if ":" in detector else "full"
        files = list((ROOT / "models/artifacts").glob(f"pulse_eot_hi_{variant}.*")) + list((ROOT / "models/artifacts").glob(f"pulse_eot_hi_{variant}_lexical.npz"))
        files.append(ROOT / "data/hf/onnx-community__silero-vad/onnx/model.onnx")
    elif detector.startswith("smart_turn"):
        files = [ROOT / "data/hf/pipecat-ai__smart-turn-v3/smart-turn-v3.2-cpu.onnx"]
    else:
        files = [ROOT / "data/hf/onnx-community__silero-vad/onnx/model.onnx"]
    return sum(f.stat().st_size for f in files) / 2**20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default="pulse_eot_hi")
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--calls", type=int, default=50)
    a = ap.parse_args()
    turns = load_eot_bench_hi(a.turns)
    parts = asr_cache.load("eot_bench_hi_clean")
    base = rss_mb()
    dets = [make(a.detector) for _ in range(a.calls)]
    ms = []
    for i, det in enumerate(dets):
        t = turns[i % len(turns)]
        tr = replay(det, t, parts.get(t.id, []), lazy=False)
        ms.append(tr.ms)
    ms = np.concatenate(ms)
    out = {
        "detector": a.detector,
        "hops": int(len(ms)),
        "ms_per_hop_mean": float(ms.mean()),
        "ms_per_hop_p50": float(np.percentile(ms, 50)),
        "ms_per_hop_p95": float(np.percentile(ms, 95)),
        "ms_per_hop_p99": float(np.percentile(ms, 99)),
        "calls_per_core": float(100.0 / ms.mean()),
        "rss_mb_process": rss_mb(),
        "rss_mb_for_calls": rss_mb() - base,
        "concurrent_calls_held": a.calls,
        "model_mb": artifact_mb(a.detector),
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER", ""),
    }
    dst = ROOT / "results/efficiency"
    dst.mkdir(parents=True, exist_ok=True)
    (dst / f"{a.detector.replace(':', '__')}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

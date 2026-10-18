"""Run one detector over one test set / condition through the streaming harness.

    python -m eval.run_bench --detector silero_timer --set eot_bench_hi --condition clean

Writes results/<set>/<condition>/<detector>/{predictions.parquet, traces.parquet,
sweep.parquet, summary.json}. predictions.parquet uses eot-bench's exact schema.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from eval import asr_cache, conditions, metrics
from eval.data import Turn, load_eot_bench_hi
from eval.harness import bench_rows, replay, to_frame
from eval.registry import make

ROOT = Path(__file__).resolve().parents[1]
PAD_S = 3.0  # silence tail after the recording so deployed (VAD-timed) policies can fire

_DET = None


def _init(name: str) -> None:
    global _DET
    _DET = make(name)


def _run_one(args) -> tuple[list[dict], dict]:
    turn, partials = args
    tr = replay(_DET, turn, partials, pad_s=PAD_S, lazy=True)
    rows = bench_rows(turn, tr, _DET.score_point)
    trace = {"id": turn.id, "t": tr.t.tolist(), "p": tr.p.tolist(), "sil": tr.sil.tolist(), "ms": tr.ms.tolist(), "audio_end": tr.audio_end}
    return rows, trace


def load_set(name: str, limit: int | None) -> list[Turn]:
    if name == "eot_bench_hi":
        return load_eot_bench_hi(limit)
    if name.startswith("dev_"):
        from models.episodes import load_episodes

        return load_episodes(name, limit)
    raise ValueError(name)


def run(detector: str, set_name: str, condition: str, limit: int | None = None, workers: int | None = None) -> dict:
    turns = [conditions.apply(t, condition) for t in load_set(set_name, limit)]
    cache_name = f"{set_name}_{condition}" + (f"_n{limit}" if limit else "")
    asr_cache.build(turns, cache_name, PAD_S)
    parts = asr_cache.load(cache_name)
    det = make(detector)
    out = ROOT / "results" / set_name / condition / detector.replace(":", "__")
    out.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    t0 = time.time()
    rows, traces = [], []
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(detector,)) as ex:
        for r, tr in ex.map(_run_one, [(t, parts.get(t.id, [])) for t in turns], chunksize=2):
            rows.extend(r)
            traces.append(tr)
    pred = to_frame(rows)
    pred.to_parquet(out / "predictions.parquet")
    pd.DataFrame(traces).to_parquet(out / "traces.parquet")
    sw, summary = metrics.summarise(pred, det.score_point)
    sw.to_parquet(out / "sweep.parquet")
    ms = np.concatenate([np.asarray(t["ms"]) for t in traces])
    p = np.concatenate([np.asarray(t["p"]) for t in traces])
    computed = np.isfinite(p)
    summary.update(
        {
            "detector": detector,
            "set": set_name,
            "condition": condition,
            "n_turns": len(turns),
            "score_point": det.score_point,
            "uses_text": det.uses_text,
            "wall_s": round(time.time() - t0, 1),
            "ms_per_hop_p50": float(np.percentile(ms, 50)),
            "ms_per_hop_p95": float(np.percentile(ms, 95)),
            "ms_per_hop_max": float(ms.max()),
            "ms_per_scored_hop_p95": float(np.percentile(ms[computed], 95)) if computed.any() else None,
            "note_timing": f"wall clock with {workers} parallel worker processes on one machine; see eval/bench_cpu.py for the single-core benchmark",
        }
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--set", default="eot_bench_hi")
    ap.add_argument("--condition", default="clean", choices=conditions.CONDITIONS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    s = run(a.detector, a.set, a.condition, a.limit, a.workers)
    print(json.dumps({k: s[k] for k in ("detector", "condition", "FCR@300", "FCR@600", "Lat@5%", "Lat@10%", "Detect@5%", "auc", "ap", "wall_s")}, indent=1))


if __name__ == "__main__":
    main()

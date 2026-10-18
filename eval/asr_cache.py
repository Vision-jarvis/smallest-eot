"""Run the streaming ASR replay once per dataset and cache partials with arrival times.

The cache is a parquet of (turn_id, t_arrival, text, is_final). Arrival times include the
measured decode time on this machine, so a slower or faster CPU produces a slightly
different (but still causal) log. The detector never sees partials before `t_arrival`.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from core.asr import Partial, records_to_partials, transcribe_stream
from core.audio import pad_silence
from eval.data import Turn

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache"


def _work(args: tuple[str, np.ndarray, int, float]) -> list[dict]:
    tid, x, sr, pad_s = args
    x = pad_silence(x, sr, pad_s, np.random.default_rng(0)) if pad_s > 0 else x
    return [{"turn_id": tid, "t_arrival": p.t_arrival, "text": p.text, "is_final": p.is_final} for p in transcribe_stream(x, sr)]


def build(turns: list[Turn], name: str, pad_s: float, workers: int | None = None) -> Path:
    """Transcribe every turn (in parallel processes) and write data/cache/asr_<name>.parquet."""
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"asr_{name}.parquet"
    if out.exists():
        return out
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    rows: list[dict] = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_work, [(t.id, t.audio, t.sr, pad_s) for t in turns], chunksize=4):
            rows.extend(r)
    pd.DataFrame(rows, columns=["turn_id", "t_arrival", "text", "is_final"]).to_parquet(out)
    return out


def load(name: str) -> dict[str, list[Partial]]:
    df = pd.read_parquet(CACHE / f"asr_{name}.parquet")
    return {tid: records_to_partials(g.to_dict("records")) for tid, g in df.groupby("turn_id", sort=False)}

"""Streaming replay harness: real-time-ordered 20 ms PCM16 frames, a decision every 100 ms.

`replay()` is the only place audio and transcripts meet a detector. It enforces causality by
construction: at hop t the detector has received exactly the frames covering [0, t) and the
partials whose logged arrival time is <= t. Nothing else is passed in.

The per-hop trace is then read out two ways:

* `bench_rows()`: eot-bench's span grid (every 0.1 s inside each labelled silence span, plus
  the span end) with its exact column schema, so `eval/metrics.py` and LiveKit's own
  `eot_harness.metrics` give identical numbers (tests/test_metrics_parity.py).
* the full trace, including a padded silence tail, for the deployed policy in
  `eval/deployed.py` where the detector's own VAD (not oracle spans) times the decision.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from core.asr import Partial
from core.audio import FRAME_MS, HOP_MS, frames, pad_silence
from eval.data import Turn

FRAMES_PER_HOP = HOP_MS // FRAME_MS
EPS = 1e-6


@dataclass
class Trace:
    turn_id: str
    t: np.ndarray  # hop end times (s)
    p: np.ndarray  # p_eot at each hop (nan where a lazy detector was not asked)
    sil: np.ndarray  # detector's own trailing-silence estimate (s)
    ms: np.ndarray  # wall-clock compute per hop (ms), audio push + score
    audio_end: float  # end of the real (unpadded) audio


def grid_times(start: float, end: float, step: float = 0.1) -> list[float]:
    """eot-bench `_time_grid`: start, start+step, ..., plus the span end if off-grid."""
    n = max(0, int(math.floor((end - start) / step + EPS)))
    vals = [round(start + i * step, 6) for i in range(n + 1)]
    if vals and vals[-1] < end - EPS:
        vals.append(round(end, 6))
    return vals


def needed_hops(turn: Turn, score_point: float | None) -> set[int]:
    """Hop indices the benchmark will read (lets lazy detectors skip the rest)."""
    need: set[int] = set()
    for s in turn.spans:
        ts = [s.start + score_point] if score_point is not None else grid_times(s.start, s.end)
        for t in ts:
            if t <= s.end + EPS:
                need.add(hop_index(t))
    return need


def hop_index(t: float) -> int:
    """Latest hop whose end time is <= t (causal read-out for off-grid span ends)."""
    return int(math.floor(t / (HOP_MS / 1000) + EPS))


def replay(det, turn: Turn, partials: list[Partial], pad_s: float = 0.0, lazy: bool = True, seed: int = 0) -> Trace:
    rng = np.random.default_rng(seed)
    x = pad_silence(turn.audio, turn.sr, pad_s, rng) if pad_s > 0 else turn.audio
    need = needed_hops(turn, getattr(det, "score_point", None)) if lazy else None
    parts = sorted(partials, key=lambda p: p.t_arrival)
    det.start_turn(turn.sr, turn.agent_text)
    ts, ps, sils, mss = [], [], [], []
    k = 0
    acc_ms = 0.0
    for i, f in enumerate(frames(x, turn.sr)):
        t0 = time.perf_counter()
        det.push_audio(f)
        acc_ms += (time.perf_counter() - t0) * 1000
        if (i + 1) % FRAMES_PER_HOP:
            continue
        hop = (i + 1) // FRAMES_PER_HOP
        t = round(hop * HOP_MS / 1000, 6)
        while k < len(parts) and parts[k].t_arrival <= t + EPS:
            det.push_partial(parts[k])
            k += 1
        t0 = time.perf_counter()
        p = det.score(t, need is None or hop in need)
        acc_ms += (time.perf_counter() - t0) * 1000
        ts.append(t)
        ps.append(p)
        sils.append(det.silence_s(t))
        mss.append(acc_ms)
        acc_ms = 0.0
    return Trace(turn.id, np.array(ts), np.array(ps, dtype=float), np.array(sils), np.array(mss), turn.duration)


def bench_rows(turn: Turn, trace: Trace, score_point: float | None) -> list[dict]:
    """Trace -> eot-bench predictions.parquet rows for this turn."""
    rows = []
    for s in turn.spans:
        sp_t = None if score_point is None else round(s.start + score_point, 6)
        for t in grid_times(s.start, s.end):
            h = hop_index(t) - 1  # trace[0] is hop 1 (t = 0.1)
            p = float(trace.p[h]) if 0 <= h < len(trace.p) else 0.0
            needed = score_point is None or abs(t - sp_t) < EPS
            if not np.isfinite(p):
                if needed:
                    raise ValueError(f"{turn.id}: detector returned no score at needed t={t}")
                p = 0.0  # never read: score-point metrics use only the score-point row
            rows.append(
                {
                    "id": turn.id,
                    "language": turn.language,
                    "span_index": s.index,
                    "timestamp": t,
                    "silence_dur": round(t - s.start, 6),
                    "p_eot": p,
                    "label": s.label,
                }
            )
    return rows


def to_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["id", "language", "span_index", "timestamp", "silence_dur", "p_eot", "label"])

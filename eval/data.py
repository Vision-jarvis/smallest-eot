"""Turn records: the one data shape every harness, model and benchmark in this repo uses.

A `Turn` is one user turn: causal audio, labelled silence spans (`hold` / `eot`), the last
agent utterance (dialogue context) and, for labelling only, reference words. eot-bench rows,
synthetic dev episodes and (once IndicVoices access is granted) Bharat-Turn-8k spans are all
converted into this shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from core.audio import decode, resample

ROOT = Path(__file__).resolve().parents[1]
EOT_BENCH_HI = ROOT / "data/hf/livekit__eot-bench-data/data/hi/validation-00000-of-00001.parquet"
MIN_SILENCE_SPAN = 0.1  # eot-bench --min-silence-span default


@dataclass
class Span:
    start: float
    end: float
    label: str  # "hold" | "eot"
    index: int  # position in the source row's silence_spans (eot-bench span_index)

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass
class Turn:
    id: str
    audio: np.ndarray
    sr: int
    spans: list[Span]
    agent_text: str | None = None
    words: list[dict] = field(default_factory=list)  # reference words: labels / slicing only
    language: str = "hi"
    meta: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sr


def spans_from_row(silence_spans) -> list[Span]:
    """eot-bench convention: the final silence span is `eot`, every earlier one `hold`."""
    out = []
    n = len(silence_spans)
    for i, s in enumerate(silence_spans):
        start, end = float(s["start"]), float(s["end"])
        if end - start < MIN_SILENCE_SPAN - 1e-6:
            continue
        out.append(Span(start, end, "eot" if i == n - 1 else "hold", i))
    return out


def last_agent_text(messages) -> str | None:
    for m in reversed(list(messages if messages is not None else [])):
        if m.get("role") == "assistant" and m.get("content"):
            return str(m["content"])
    return None


def load_eot_bench_hi(limit: int | None = None, sr: int = 16000) -> list[Turn]:
    """The 400-turn Hindi validation split of livekit/eot-bench-data (TEST ONLY, never train)."""
    df = pd.read_parquet(EOT_BENCH_HI)
    if limit:
        df = df.iloc[:limit]
    turns = []
    for r in df.itertuples(index=False):
        x, s = decode(r.audio)
        turns.append(
            Turn(
                id=str(r.id),
                audio=resample(x, s, sr),
                sr=sr,
                spans=spans_from_row(r.silence_spans),
                agent_text=last_agent_text(r.messages),
                words=[dict(w) for w in (r.words if r.words is not None else [])],
                language=str(r.language),
            )
        )
    return turns


def word_before(turn: Turn, t: float) -> str | None:
    """Reference word that ends last at or before time t (used to slice filler-preceded holds)."""
    best = None
    for w in turn.words:
        if float(w["end"]) <= t + 0.05:
            best = str(w["word"]).strip()
    return best

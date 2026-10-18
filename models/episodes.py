"""Turn Smart Turn v3.2 Hindi clips into eot-bench-shaped single-pause episodes.

Each clip of pipecat-ai/smart-turn-data-v3.2-{train,test} (CC BY 4.0) is one user utterance
labelled `endpoint_bool` (complete) or not (cut mid-thought). We make it one causal turn:

    [left pad so speech ends on the 100 ms grid][clip speech][silence of duration d]

with a single labelled span starting at the (Silero-detected) end of speech: `eot` if the
clip is complete, else `hold`. Hold durations d are drawn from the measured hold-span
histogram of livekit/eot-bench-data **English** (0.2 to 5 s, as eot-bench filters), and EoT
span durations from its English EoT spans, so the synthetic pause statistics are realistic
but no Hindi test statistic is used. The silence after the clip keeps the clip's own trailing
audio where it has some, then continues with a matched noise floor.

Splits (honesty rule 3): `train` episodes come from the Smart Turn *train* split; `dev`
episodes from its *test* split. eot-bench Hindi is never touched here.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from core.audio import pad_silence, resample
from core.vad import speech_prob_trace
from eval.data import Span, Turn

ROOT = Path(__file__).resolve().parents[1]
ST_TRAIN = ROOT / "data/smart_turn_hi"
ST_TEST = ROOT / "data/smart_turn_hi_test"
DURS = ROOT / "data/cache/en_span_durations.json"
EPISODES = ROOT / "data/cache/episodes"


def _durations() -> tuple[np.ndarray, np.ndarray]:
    d = json.loads(DURS.read_text())
    hold = np.array(d["hold"])
    hold = hold[(hold >= 0.2) & (hold <= 5.0)]
    return hold, np.array(d["eot"])


def speech_end(x: np.ndarray, sr: int) -> float | None:
    """End time of the last Silero speech window (prob >= 0.5) in the clip."""
    tr = speech_prob_trace(x, sr)
    sp = tr[tr[:, 1] >= 0.5]
    return float(sp[-1, 0]) if len(sp) else None


def build_split(split: str, max_clips: int | None = None, seed: int = 0, pattern: str = "hi_*.parquet") -> pd.DataFrame:
    """Read Smart Turn parquet shards (Hindi by default) and return the clip table (audio as bytes)."""
    src = ST_TRAIN if split == "train" else ST_TEST
    files = sorted(src.glob(pattern))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if max_clips and len(df) > max_clips:
        df = df.sample(max_clips, random_state=seed).reset_index(drop=True)
    return df


def make_episode(row, rng: np.random.Generator, hold_d: np.ndarray, eot_d: np.ndarray, sr: int = 16000) -> Turn | None:
    x, s = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = resample(x, s, sr)
    end = speech_end(x, sr)
    if end is None or end < 0.5:
        return None
    lead = (np.ceil(end * 10 - 1e-6) / 10) - end  # shift so the pause starts on the 0.1 s grid
    x = np.concatenate([np.zeros(int(round(lead * sr)), np.float32), x])
    start = round(end + lead, 6)
    label = "eot" if bool(row["endpoint_bool"]) else "hold"
    dur = float(rng.choice(eot_d if label == "eot" else hold_d))
    keep = x[: int((start + dur) * sr)]
    keep = pad_silence(keep, sr, start + dur - len(keep) / sr, rng)
    return Turn(
        id=str(row["id"]),
        audio=keep,
        sr=sr,
        spans=[Span(start, round(start + dur, 6), label, 0)],
        meta={"dataset": row["dataset"], "synthetic": bool(row["synthetic"]), "midfiller": bool(row["midfiller"]), "endfiller": bool(row["endfiller"])},
    )


def build_human(name: str, max_clips: int | None, seed: int = 0) -> Path:
    """Real-human (non-synthetic) Smart Turn train clips in other languages: prosody-only examples."""
    return build_and_save(name, "train", max_clips, seed, pattern="human_*.parquet")


def build_and_save(name: str, split: str, max_clips: int | None, seed: int = 0, pattern: str = "hi_*.parquet") -> Path:
    EPISODES.mkdir(parents=True, exist_ok=True)
    out = EPISODES / f"{name}.parquet"
    df = build_split(split, max_clips, seed, pattern)
    hold_d, eot_d = _durations()
    rng = np.random.default_rng(seed)
    recs = []
    for r in df.to_dict("records"):
        t = make_episode(r, rng, hold_d, eot_d)
        if t is None:
            continue
        buf = io.BytesIO()
        sf.write(buf, t.audio, t.sr, format="FLAC")
        sp = t.spans[0]
        recs.append({"id": t.id, "flac": buf.getvalue(), "start": sp.start, "end": sp.end, "label": sp.label, **t.meta})
    pd.DataFrame(recs).to_parquet(out)
    return out


def load_episodes(name: str, limit: int | None = None) -> list[Turn]:
    df = pd.read_parquet(EPISODES / f"{name}.parquet")
    if limit:
        df = df.iloc[:limit]
    out = []
    for r in df.itertuples(index=False):
        x, sr = sf.read(io.BytesIO(r.flac), dtype="float32")
        meta = {"dataset": r.dataset, "synthetic": bool(r.synthetic), "midfiller": bool(r.midfiller), "endfiller": bool(r.endfiller)}
        out.append(Turn(id=str(r.id), audio=x, sr=sr, spans=[Span(float(r.start), float(r.end), r.label, 0)], meta=meta))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--max-clips", type=int, default=None)
    a = ap.parse_args()
    print(build_and_save(a.name, a.split, a.max_clips))

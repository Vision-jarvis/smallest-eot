"""Proxy overlap set for the barge-in arbiter (until the IndicVoices overlap set is licensed).

Label rules (problem statement Section 5, rules 3 and 4) define a backchannel as other-speaker
speech < 1 s made only of {haan, hmm, achha, ji, ...}, and a barge-in as other-speaker speech
> 1 s. Without two-party Hindi calls we build the *acoustic* proxy of those rules from
single-speaker clips:

* backchannel: an isolated inter-pausal unit (IPU) of 0.12 to 0.8 s, with >= 300 ms of
  silence on both sides, cropped and followed by silence (Silero IPUs on Smart Turn clips).
  When the streaming ASR transcript of the IPU is made only of backchannel tokens it is also
  flagged `lexical_bc` (a stricter subset reported separately).
* barge_in: the onset of an IPU longer than 1.2 s.
* noise: MUSAN noise (free-sound + sound-bible), cropped at energy onsets.

Clips are split by source split (Smart Turn train -> train, Smart Turn test -> test) and MUSAN
by file hash, so no audio file contributes to two splits.
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from core.audio import resample
from core.vad import speech_prob_trace
from eval.conditions import MUSAN_NOISE

ROOT = Path(__file__).resolve().parents[1]
SR = 16000
PRE_S = 0.3
POST_S = 0.9  # >= the 600 ms streaming-update horizon plus margin


def ipus(x: np.ndarray, sr: int, min_gap: float = 0.2) -> list[tuple[float, float]]:
    tr = speech_prob_trace(x, sr)
    if len(tr) == 0:
        return []
    win = tr[1, 0] - tr[0, 0] if len(tr) > 1 else 0.032
    sp = tr[:, 1] >= 0.5
    runs, start = [], None
    for (t, _), s in zip(tr, sp):
        if s and start is None:
            start = t - win
        if not s and start is not None:
            runs.append([start, t - win])
            start = None
    if start is not None:
        runs.append([start, tr[-1, 0]])
    merged: list[list[float]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < min_gap:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged]


def _floor(rng: np.random.Generator, n: int) -> np.ndarray:
    return (rng.standard_normal(n) * 3e-4).astype(np.float32)


def speech_samples(x: np.ndarray, sr: int, clip_id: str, rng: np.random.Generator) -> list[dict]:
    out = []
    segs = ipus(x, sr)
    for k, (a, b) in enumerate(segs):
        gap_before = a - (segs[k - 1][1] if k else 0.0)
        gap_after = (segs[k + 1][0] if k + 1 < len(segs) else len(x) / sr) - b
        dur = b - a
        lo = max(0, int((a - 0.05) * sr))
        if 0.12 <= dur <= 0.8 and gap_before >= 0.3 and gap_after >= 0.3:
            body = x[lo : int(b * sr) + int(0.03 * sr)]
            label = "backchannel"
        elif dur > 1.2:
            body = x[lo : lo + int(POST_S * sr)]
            label = "barge_in"
        else:
            continue
        audio = np.concatenate([_floor(rng, int(PRE_S * sr)), body, _floor(rng, int(POST_S * sr))])[: int((PRE_S + POST_S) * sr)]
        out.append({"id": f"{clip_id}#{k}", "source": clip_id, "label": label, "ipu_dur": dur, "audio": audio})
    return out


def noise_samples(split: str, rng: np.random.Generator, per_file: int = 2) -> list[dict]:
    out = []
    for f in sorted(MUSAN_NOISE.rglob("*.wav")):
        h = int(hashlib.md5(f.name.encode()).hexdigest()[:4], 16) % 100
        if (split == "train") != (h < 70):
            continue
        x, s = sf.read(f, dtype="float32")
        x = resample(x.mean(axis=1) if x.ndim > 1 else x, s, SR)
        for j in range(per_file):
            st = int(rng.integers(0, max(1, len(x) - int(POST_S * SR))))
            body = x[st : st + int(POST_S * SR)]
            body = body / (np.max(np.abs(body)) + 1e-6) * float(rng.uniform(0.05, 0.5))
            audio = np.concatenate([_floor(rng, int(PRE_S * SR)), body])
            out.append({"id": f"{f.stem}#{j}", "source": f.stem, "label": "noise", "ipu_dur": np.nan, "audio": audio})
    return out


def build(split: str, max_clips: int | None = None, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    src = ROOT / ("data/smart_turn_hi" if split == "train" else "data/smart_turn_hi_test")
    files = sorted(src.glob("hi_*.parquet")) + (sorted(src.glob("human_*.parquet")) if split == "train" else [])
    df = pd.concat([pd.read_parquet(f, columns=["id", "audio", "language"]) for f in files], ignore_index=True)
    if max_clips and len(df) > max_clips:
        df = df.sample(max_clips, random_state=seed)
    out = []
    for r in df.itertuples(index=False):
        x, s = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32")
        x = resample(x.mean(axis=1) if x.ndim > 1 else x, s, SR)
        for smp in speech_samples(x, SR, str(r.id), rng):
            smp["language"] = r.language
            out.append(smp)
    for smp in noise_samples(split, rng):
        smp["language"] = "none"
        out.append(smp)
    return out

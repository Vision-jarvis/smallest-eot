"""Open streaming ASR (Vosk, Kaldi-based, Apache-2.0) replayed causally with logged arrival times.

Honesty rule 2 of the problem statement forbids oracle transcripts at inference: test-time
text must come from a streaming ASR run live or replayed with logged arrival times. This
module is that replay. Audio is fed in 100 ms chunks in real-time order; after each chunk we
record whatever the recogniser emits, stamped with

    arrival = chunk_end_time + measured_decode_time

so a detector can only see a partial once it would really have arrived. Pulse partials will
slot into the same `Partial` record once a Smallest API key is available (see
adapters/pulse_endpointing.py).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.audio import to_pcm16

ROOT = Path(__file__).resolve().parents[1]
VOSK_HI = ROOT / "data/vosk/vosk-model-small-hi-0.22"
CHUNK_S = 0.1

_MODEL = None


@dataclass(frozen=True)
class Partial:
    """One transcript message as it arrived: (t_arrival, text, is_final).

    `text` is the text of the current utterance only (finals are not repeated in later
    partials), matching how Pulse and most streaming ASRs emit interim results.
    """

    t_arrival: float
    text: str
    is_final: bool


def _model():
    global _MODEL
    if _MODEL is None:
        from vosk import Model, SetLogLevel

        SetLogLevel(-1)
        _MODEL = Model(str(VOSK_HI))
    return _MODEL


def transcribe_stream(x: np.ndarray, sr: int) -> list[Partial]:
    """Replay `x` through Vosk in 100 ms chunks and return partials with arrival times."""
    from vosk import KaldiRecognizer

    rec = KaldiRecognizer(_model(), sr)
    pcm = to_pcm16(x).tobytes()
    step = int(CHUNK_S * sr) * 2
    out: list[Partial] = []
    last_partial = ""
    for i in range(0, len(pcm), step):
        t_end = min(len(pcm), i + step) / 2 / sr
        t0 = time.perf_counter()
        is_final = rec.AcceptWaveform(pcm[i : i + step])
        text = json.loads(rec.Result())["text"] if is_final else json.loads(rec.PartialResult())["partial"]
        arrival = t_end + (time.perf_counter() - t0)
        if is_final:
            if text:
                out.append(Partial(arrival, text, True))
            last_partial = ""
        elif text and text != last_partial:
            out.append(Partial(arrival, text, False))
            last_partial = text
    return out


def visible_text(partials: list[Partial], t: float) -> tuple[str, bool]:
    """Transcript visible at time t: all finals so far plus the latest partial after them.

    Returns (text, ends_with_final) where the flag says the ASR has closed the utterance.
    """
    finals: list[str] = []
    current = ""
    closed = False
    for p in partials:
        if p.t_arrival > t:
            break
        if p.is_final:
            finals.append(p.text)
            current = ""
            closed = True
        else:
            current = p.text
            closed = False
    text = " ".join(finals + ([current] if current else []))
    return text, closed


def partials_to_records(partials: list[Partial]) -> list[dict]:
    return [{"t_arrival": p.t_arrival, "text": p.text, "is_final": p.is_final} for p in partials]


def records_to_partials(records) -> list[Partial]:
    return [Partial(float(r["t_arrival"]), str(r["text"]), bool(r["is_final"])) for r in records]

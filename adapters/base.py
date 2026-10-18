"""Detector interface shared by every baseline and by Pulse-EoT-hi.

A detector is a causal state machine. The streaming harness (eval/harness.py) drives it:

    det.start_turn(sr, agent_text)
    for every 20 ms frame:       det.push_audio(frame)
    whenever a partial arrives:  det.push_partial(partial)     # only after its arrival time
    every 100 ms hop t:          p = det.score(t, need)         # p_eot(t) in [0, 1]

`need` tells lazy detectors (Whisper-sized encoders that cost ~100 ms per call on one core)
whether this hop is scored by the benchmark; cheap detectors ignore it and always compute.
`silence_s(t)` is the detector's own causal estimate of the user's trailing silence, used by
the deployed policy (eval/deployed.py) where no oracle silence spans exist.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from core.asr import Partial


class StreamingDetector(Protocol):
    name: str
    score_point: float | None  # eot-bench score point (e.g. 0.2) or None for full traces
    uses_text: bool

    def start_turn(self, sr: int, agent_text: str | None) -> None: ...

    def push_audio(self, frame: np.ndarray) -> None: ...

    def push_partial(self, partial: Partial) -> None: ...

    def score(self, t: float, need: bool) -> float: ...

    def silence_s(self, t: float) -> float: ...


class EotBenchBatchAdapter:
    """Wrap a StreamingDetector in eot-bench's `predict_batch` adapter contract.

    The statement's model API (Section 6) is eot-bench's: each item carries the causal audio
    prefix, the messages, and optionally `partials` [(t_arrival, text, is_final), ...]. We
    replay the prefix through a fresh detector in 20 ms frames and return its p_eot at the
    end of the prefix, so the same object can be dropped into `eot-harness predict`.
    """

    def __init__(self, factory, adapter_id: str) -> None:
        self.factory = factory
        self.adapter_id = adapter_id
        self.score_point = None

    def supports_language(self, lang_code: str) -> bool:
        return lang_code.startswith("hi")

    def predict_batch(self, batch: list[dict]) -> list[float]:
        from core.audio import frames

        out = []
        for item in batch:
            audio = item["audio"]
            x = np.asarray(audio["array"], dtype=np.float32)
            sr = int(audio["sampling_rate"])
            agent = next((m.get("content") for m in reversed(item.get("messages") or []) if m.get("role") == "assistant"), None)
            det = self.factory()
            det.start_turn(sr, agent)
            partials = sorted((Partial(float(a), str(b), bool(c)) for a, b, c in item.get("partials") or []), key=lambda p: p.t_arrival)
            t_end = len(x) / sr
            k = 0
            frame_s = 0.02
            for i, f in enumerate(frames(x, sr)):
                t = (i + 1) * frame_s
                while k < len(partials) and partials[k].t_arrival <= min(t, t_end):
                    det.push_partial(partials[k])
                    k += 1
                det.push_audio(f)
            out.append(float(det.score(t_end, True)))
        return out

"""Causal streaming Silero VAD (v5 ONNX) on onnxruntime CPU.

Silero consumes 512-sample windows at 16 kHz (256 at 8 kHz) with a 64 (32) sample context
carried from the previous window and a recurrent state. Our frames are 20 ms, so we buffer
and run the network whenever a full window is available. Everything is strictly causal:
the state after `push(frame)` depends only on audio pushed so far.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
SILERO_PATH = ROOT / "data/hf/onnx-community__silero-vad/onnx/model.onnx"

_SESSION: ort.InferenceSession | None = None


def _session() -> ort.InferenceSession:
    global _SESSION
    if _SESSION is None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        _SESSION = ort.InferenceSession(str(SILERO_PATH), sess_options=so, providers=["CPUExecutionProvider"])
    return _SESSION


@dataclass
class SileroStream:
    """Streaming VAD with hysteresis.

    `onset` / `offset` are the speech-probability thresholds to enter / leave speech
    (Silero's own defaults are 0.5 / 0.35). `min_speech_ms` suppresses blips shorter
    than that (a cough or click), which is how production VAD front-ends behave.
    """

    sr: int = 16000
    onset: float = 0.5
    offset: float = 0.35
    min_speech_ms: float = 64.0
    _buf: np.ndarray = field(default_factory=lambda: np.zeros(0, np.float32))
    _ctx: np.ndarray | None = None
    _state: np.ndarray | None = None
    t: float = 0.0  # seconds of audio consumed by the network
    in_speech: bool = False
    speech_run: float = 0.0
    last_speech_end: float | None = None  # time when the last confirmed speech ended
    speech_start: float | None = None
    probs: list[tuple[float, float]] = field(default_factory=list)  # (window end time, prob)

    def __post_init__(self) -> None:
        self.win = 512 if self.sr == 16000 else 256
        self.ctx_len = 64 if self.sr == 16000 else 32
        self.reset()

    def reset(self) -> None:
        self._buf = np.zeros(0, np.float32)
        self._ctx = np.zeros(self.ctx_len, np.float32)
        self._state = np.zeros((2, 1, 128), np.float32)
        self.t = 0.0
        self.in_speech = False
        self.speech_run = 0.0
        self.last_speech_end = None
        self.speech_start = None
        self.probs = []

    def push(self, frame: np.ndarray) -> None:
        self._buf = np.concatenate([self._buf, frame.astype(np.float32)])
        while len(self._buf) >= self.win:
            chunk, self._buf = self._buf[: self.win], self._buf[self.win :]
            x = np.concatenate([self._ctx, chunk])[None, :]
            out, self._state = _session().run(
                None, {"input": x, "state": self._state, "sr": np.array(self.sr, dtype=np.int64)}
            )
            self._ctx = chunk[-self.ctx_len :]
            self.t += self.win / self.sr
            self._update(float(out.reshape(-1)[0]))

    def _update(self, p: float) -> None:
        dt = self.win / self.sr
        self.probs.append((self.t, p))
        if p >= self.onset:
            self.speech_run += dt
            if not self.in_speech and self.speech_run * 1000 >= self.min_speech_ms:
                self.in_speech = True
                self.speech_start = self.t - self.speech_run
        elif p < self.offset:
            if self.in_speech:
                self.in_speech = False
                self.last_speech_end = self.t - dt
            self.speech_run = 0.0

    def trailing_silence(self, now: float) -> float:
        """Seconds since confirmed speech ended (0 while in speech, `now` if never spoke)."""
        if self.in_speech:
            return 0.0
        if self.last_speech_end is None:
            return now
        return max(0.0, now - self.last_speech_end)


def speech_prob_trace(x: np.ndarray, sr: int) -> np.ndarray:
    """Offline convenience: (time, prob) for every Silero window of a whole clip (still causal)."""
    v = SileroStream(sr=sr)
    v.push(x)
    return np.array(v.probs, dtype=np.float64).reshape(-1, 2)

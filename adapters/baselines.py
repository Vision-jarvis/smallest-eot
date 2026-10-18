"""CPU baselines run through the same streaming harness as Pulse-EoT-hi.

* `SileroTimer`: the production default for a Hindi agent today (a trailing-silence timer on
  a real VAD). p_eot(t) = min(1, VAD trailing silence / 2 s), so a threshold th in the
  eot-bench sweep is exactly a silence timer of 2000*th ms, and the deployed sweep over
  300..1200 ms in eval/deployed.py is the `eou_timeout_ms`-style curve.
* `SmartTurnDetector`: Smart Turn v3.2 CPU ONNX, scored once per pause at 0.2 s of silence
  (eot-bench's score point for SmartTurn), last 8 s of audio.
* `HinglishTurnDetector`: deveshu/hinglish-turn-detector int8, same protocol.
"""

from __future__ import annotations

import numpy as np

from core.asr import Partial
from core.audio import resample
from core.encoders import HinglishTurn, SmartTurnV32
from core.vad import SileroStream

TIMER_SCALE_S = 2.0


class AudioRing:
    """Last `seconds` of audio at 16 kHz (the encoders' rate)."""

    def __init__(self, seconds: float = 8.0) -> None:
        self.n = int(16000 * seconds)
        self.buf = np.zeros(0, np.float32)
        self.sr = 16000

    def reset(self, sr: int) -> None:
        self.buf = np.zeros(0, np.float32)
        self.sr = sr

    def push(self, frame: np.ndarray) -> None:
        f = resample(frame, self.sr, 16000) if self.sr != 16000 else frame
        self.buf = np.concatenate([self.buf, f])[-self.n :]


class SileroTimer:
    name = "silero-vad-timer"
    score_point = None
    uses_text = False

    def __init__(self) -> None:
        self.vad = SileroStream()

    def start_turn(self, sr: int, agent_text: str | None) -> None:
        self.vad = SileroStream(sr=sr)

    def push_audio(self, frame: np.ndarray) -> None:
        self.vad.push(frame)

    def push_partial(self, partial: Partial) -> None:
        return None

    def silence_s(self, t: float) -> float:
        return self.vad.trailing_silence(t)

    def score(self, t: float, need: bool) -> float:
        return float(min(1.0, self.silence_s(t) / TIMER_SCALE_S))


class _EncoderDetector:
    score_point = 0.2
    uses_text = False

    def __init__(self, encoder) -> None:
        self.enc = encoder
        self.ring = AudioRing()
        self.vad = SileroStream()
        self.calls_ms: list[float] = []

    def start_turn(self, sr: int, agent_text: str | None) -> None:
        self.ring.reset(sr)
        self.vad = SileroStream(sr=sr)

    def push_audio(self, frame: np.ndarray) -> None:
        self.ring.push(frame)
        self.vad.push(frame)

    def push_partial(self, partial: Partial) -> None:
        return None

    def silence_s(self, t: float) -> float:
        return self.vad.trailing_silence(t)

    def score(self, t: float, need: bool) -> float:
        if not need:
            return float("nan")
        p = self.enc.prob(self.ring.buf)
        self.calls_ms.append(self.enc.last_ms)
        return p


class SmartTurnDetector(_EncoderDetector):
    name = "smart-turn-v3.2"

    def __init__(self) -> None:
        super().__init__(SmartTurnV32())


class SmartTurnFp32Detector(_EncoderDetector):
    """The fp32 graph (`smart-turn-v3.2-gpu.onnx`) eot-bench's published row used, run on CPU."""

    name = "smart-turn-v3.2-fp32"

    def __init__(self) -> None:
        super().__init__(SmartTurnV32(fp32=True))


class HinglishTurnDetector(_EncoderDetector):
    name = "hinglish-turn-detector"

    def __init__(self) -> None:
        super().__init__(HinglishTurn())

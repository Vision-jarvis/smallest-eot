"""Whisper-tiny based open turn classifiers on onnxruntime CPU.

* Smart Turn v3.2 (pipecat-ai/smart-turn-v3, BSD-2): `smart-turn-v3.2-cpu.onnx`, input
  Whisper log-mel (B, 80, 800) of the last 8 s at 16 kHz, left zero-padded, waveform
  zero-mean / unit-variance normalised first (as in eot-bench's SmartTurnAudioAdapter, which
  calls WhisperFeatureExtractor with do_normalize=True). Output is already a probability.
* deveshu/hinglish-turn-detector (MIT): `model_int8.onnx`, same 8 s right-aligned log-mel,
  output a logit (sigmoid -> P(turn complete)).

The log-mel is a numpy port of Whisper's feature math (n_fft 400, hop 160, periodic Hann,
reflect-padded centred STFT, power spectrum, log10 clamp, dynamic range 8, (x+4)/4); it is
tested against transformers.WhisperFeatureExtractor in tests/test_encoders.py.
"""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
SMART_TURN_PATH = ROOT / "data/hf/pipecat-ai__smart-turn-v3/smart-turn-v3.2-cpu.onnx"
HINGLISH_PATH = ROOT / "data/hf/deveshu__hinglish-turn-detector/model_int8.onnx"
MEL_PATH = ROOT / "data/hf/deveshu__hinglish-turn-detector/mel_filters.npz"
SR = 16000
N_SAMPLES = 8 * SR
N_FFT, HOP = 400, 160


@lru_cache(maxsize=1)
def mel_filters() -> np.ndarray:
    """(80, 201) Slaney mel filterbank, the same matrix Whisper uses."""
    return np.load(MEL_PATH)["filters"].T.astype(np.float32)


@lru_cache(maxsize=1)
def _window() -> np.ndarray:
    n = np.arange(N_FFT)
    return (0.5 - 0.5 * np.cos(2 * np.pi * n / N_FFT)).astype(np.float32)  # periodic Hann


def right_align(x: np.ndarray, n: int = N_SAMPLES) -> np.ndarray:
    """Keep the last n samples, left-pad with zeros if shorter (the 8 s window convention)."""
    if len(x) >= n:
        return x[-n:].astype(np.float32)
    return np.pad(x.astype(np.float32), (n - len(x), 0))


def log_mel(x: np.ndarray, normalise: bool) -> np.ndarray:
    """Whisper log-mel (80, 800) of exactly 8 s of 16 kHz audio."""
    x = right_align(x).astype(np.float64)
    if normalise:
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    xp = np.pad(x, (N_FFT // 2, N_FFT // 2), mode="reflect")
    n_frames = 1 + (len(xp) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    spec = np.fft.rfft(xp[idx] * _window(), axis=1)
    power = (np.abs(spec) ** 2).T[:, :-1]  # (201, 800)
    mel = mel_filters().astype(np.float64) @ power
    lm = np.log10(np.maximum(mel, 1e-10))
    lm = np.maximum(lm, lm.max() - 8.0)
    return ((lm + 4.0) / 4.0).astype(np.float32)


def _session(path: Path, threads: int = 1) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


class SmartTurnV32:
    """P(turn complete) from Smart Turn v3.2 (CPU build)."""

    name = "smart-turn-v3.2-cpu"

    def __init__(self, threads: int = 1) -> None:
        self.sess = _session(SMART_TURN_PATH, threads)
        self.last_ms = 0.0

    def prob(self, x16k: np.ndarray) -> float:
        t0 = time.perf_counter()
        feats = log_mel(x16k, normalise=True)[None]
        p = float(self.sess.run(None, {"input_features": feats})[0].reshape(-1)[0])
        self.last_ms = (time.perf_counter() - t0) * 1000
        return p


class HinglishTurn:
    """P(turn complete) from deveshu/hinglish-turn-detector (int8)."""

    name = "deveshu-hinglish-turn-int8"

    def __init__(self, threads: int = 1, normalise: bool = False) -> None:
        self.sess = _session(HINGLISH_PATH, threads)
        self.normalise = normalise
        self.last_ms = 0.0

    def prob(self, x16k: np.ndarray) -> float:
        t0 = time.perf_counter()
        feats = log_mel(x16k, normalise=self.normalise)[None]
        logit = float(self.sess.run(None, {"mel": feats})[0].reshape(-1)[0])
        self.last_ms = (time.perf_counter() - t0) * 1000
        return float(1.0 / (1.0 + np.exp(-logit)))

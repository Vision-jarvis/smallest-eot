"""Audio utilities: decoding, resampling, telephony codec simulation, noise mixing, framing.

All audio inside the project is float32 mono in [-1, 1]. The streaming contract from the
problem statement is PCM16 mono at 8 or 16 kHz in 20 ms frames, so `frames()` yields
exactly that (after an int16 round trip) and every detector consumes those frames.
"""

from __future__ import annotations

import io
from collections.abc import Iterator
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

FRAME_MS = 20
HOP_MS = 100


def decode(audio: dict) -> tuple[np.ndarray, int]:
    """Decode a Hugging Face Audio dict ({"bytes"} or {"array", "sampling_rate"})."""
    if "array" in audio and audio.get("array") is not None:
        return np.asarray(audio["array"], dtype=np.float32), int(audio["sampling_rate"])
    arr, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return np.asarray(arr, dtype=np.float32), int(sr)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    g = gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def to_pcm16(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)


def from_pcm16(x: np.ndarray) -> np.ndarray:
    return (x.astype(np.float32) / 32768.0).astype(np.float32)


def mulaw_roundtrip(x: np.ndarray, sr: int, mu: int = 255) -> np.ndarray:
    """Simulate a G.711 mu-law 8 kHz phone leg and return audio at the input rate.

    Band-limits to 8 kHz (the >4 kHz band disappears), applies 8-bit mu-law companding
    and quantisation, then resamples back so downstream models see the same rate.
    """
    nb = resample(x, sr, 8000)
    nb = np.clip(nb, -1.0, 1.0)
    comp = np.sign(nb) * np.log1p(mu * np.abs(nb)) / np.log1p(mu)
    q = np.round((comp + 1.0) / 2.0 * mu) / mu * 2.0 - 1.0
    dec = np.sign(q) * ((1.0 + mu) ** np.abs(q) - 1.0) / mu
    return resample(dec.astype(np.float32), 8000, sr)


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add `noise` (looped / randomly offset) to `speech` at a target SNR over active speech.

    Speech power is measured on samples above -40 dB of the peak frame so that long
    silences do not deflate it (which would make the effective SNR far worse than stated).
    """
    if len(noise) < len(speech):
        noise = np.tile(noise, int(np.ceil(len(speech) / max(1, len(noise)))) + 1)
    off = int(rng.integers(0, len(noise) - len(speech) + 1))
    n = noise[off : off + len(speech)].astype(np.float64)
    s = speech.astype(np.float64)
    frame = 320
    nfr = max(1, len(s) // frame)
    e = np.array([np.mean(s[i * frame : (i + 1) * frame] ** 2) for i in range(nfr)]) + 1e-12
    active = e > e.max() * 1e-4
    p_s = float(e[active].mean())
    p_n = float(np.mean(n**2)) + 1e-12
    gain = np.sqrt(p_s / (p_n * 10 ** (snr_db / 10.0)))
    out = s + gain * n
    peak = np.max(np.abs(out))
    if peak > 0.999:
        out = out / peak * 0.999
    return out.astype(np.float32)


def pad_silence(x: np.ndarray, sr: int, seconds: float, rng: np.random.Generator) -> np.ndarray:
    """Append `seconds` of silence carrying a noise floor matched to the clip's quietest frames.

    Used to continue a user's trailing silence past the end of a recording so that a
    deployed (VAD-timed) policy can actually fire; digital zeros would be unrealistically
    easy for a VAD.
    """
    if seconds <= 0:
        return x
    frame = int(0.02 * sr)
    nfr = max(1, len(x) // frame)
    rms = np.sqrt(np.array([np.mean(x[i * frame : (i + 1) * frame] ** 2) for i in range(nfr)]) + 1e-12)
    floor = float(np.percentile(rms, 5))
    floor = min(max(floor, 1e-4), 3e-3)
    tail = (rng.standard_normal(int(seconds * sr)) * floor).astype(np.float32)
    return np.concatenate([x, tail])


def frames(x: np.ndarray, sr: int, frame_ms: int = FRAME_MS) -> Iterator[np.ndarray]:
    """Yield PCM16-quantised float frames of `frame_ms` (the last partial frame is zero-padded)."""
    n = int(sr * frame_ms / 1000)
    q = from_pcm16(to_pcm16(x))
    for i in range(0, len(q), n):
        f = q[i : i + n]
        if len(f) < n:
            f = np.pad(f, (0, n - len(f)))
        yield f

"""Causal features for the barge-in arbiter over the first `window_ms` after user onset.

The arbiter runs while the agent is speaking, on the echo-cancelled user channel. Onset is
an energy trigger (a 20 ms frame above -45 dBFS), deliberately cruder than a VAD so that
coughs, horns and TV also reach the arbiter, as they do in production when a VAD fires.
Everything is computed from audio in [onset - 100 ms, onset + window] only.
"""

from __future__ import annotations

import numpy as np

from core.prosody import frame_f0
from core.vad import SileroStream

ONSET_DBFS = -45.0
PRE_MS = 100


def energy_onset(x: np.ndarray, sr: int) -> float | None:
    n = int(0.02 * sr)
    for i in range(0, len(x) - n, n):
        if 20 * np.log10(np.sqrt(np.mean(x[i : i + n] ** 2)) + 1e-9) > ONSET_DBFS:
            return i / sr
    return None


def window_features(x: np.ndarray, sr: int, onset: float, window_ms: int) -> dict[str, float]:
    a = max(0, int((onset - PRE_MS / 1000) * sr))
    b = int((onset + window_ms / 1000) * sr)
    seg = x[a:b]
    vad = SileroStream(sr=sr)
    vad.push(seg)
    probs = np.array([p for _, p in vad.probs]) if vad.probs else np.zeros(1)
    n = int(0.02 * sr)
    frames = [seg[i : i + n] for i in range(0, len(seg) - n + 1, n)]
    e = np.array([20 * np.log10(np.sqrt(np.mean(f**2)) + 1e-9) for f in frames]) if frames else np.full(1, -90.0)
    f0s, corr = [], []
    prev = None
    for f in frames:
        w = f if prev is None else np.concatenate([prev, f])
        prev = f
        hz, c = frame_f0(w, sr)
        f0s.append(hz)
        corr.append(c)
    f0s = np.array(f0s)
    voiced = np.isfinite(f0s)
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) + 1e-9 if len(seg) > 16 else np.ones(8)
    freqs = np.fft.rfftfreq(max(len(seg), 16), 1 / sr)[: len(spec)]
    tail = probs[-3:]
    return {
        "vad_mean": float(probs.mean()),
        "vad_max": float(probs.max()),
        "vad_frac": float((probs >= 0.5).mean()),
        "vad_tail": float(tail.mean()),  # is the user still talking at the decision point?
        "vad_in_speech": float(vad.in_speech),
        "e_mean": float(e.mean()),
        "e_max": float(e.max()),
        "e_tail_minus_max": float(e[-5:].mean() - e.max()),
        "e_slope": float(np.polyfit(np.arange(len(e)), e, 1)[0]) if len(e) > 2 else 0.0,
        "voiced_frac": float(voiced.mean()),
        "corr_mean": float(np.mean(corr)) if corr else 0.0,
        "f0_range_st": float(12 * np.log2(np.nanmax(f0s[voiced]) / np.nanmin(f0s[voiced]))) if voiced.sum() > 1 else 0.0,
        "f0_tail_st": float(12 * np.log2(np.nanmean(f0s[voiced][-3:]) / 100.0)) if voiced.sum() > 0 else 0.0,
        "spec_flatness": float(np.exp(np.mean(np.log(spec))) / np.mean(spec)),
        "spec_centroid": float(np.sum(freqs * spec) / np.sum(spec)),
    }


FEATURES = (
    "vad_mean", "vad_max", "vad_frac", "vad_tail", "vad_in_speech", "e_mean", "e_max", "e_tail_minus_max",
    "e_slope", "voiced_frac", "corr_mean", "f0_range_st", "f0_tail_st", "spec_flatness", "spec_centroid",
)

"""Causal frame-level prosody (log energy, f0, voicing) and turn-level pause features.

Per 20 ms frame we compute log RMS energy and an autocorrelation f0 estimate over a 40 ms
window (this frame plus the previous one), which is causal and costs ~20 us per frame.
The f0 search range 70 to 400 Hz is valid at both 8 and 16 kHz because the fundamental of
speech sits far below the 4 kHz telephone cut-off, which is why pitch survives narrowband
better than the breathiness / fricative cues audio models lean on (Section 3.3).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

F0_MIN, F0_MAX = 70.0, 400.0
VOICING_MIN_CORR = 0.45


def frame_f0(window: np.ndarray, sr: int) -> tuple[float, float]:
    """Return (f0 Hz or nan, normalised autocorrelation peak) for one analysis window."""
    w = window - window.mean()
    energy = float(np.dot(w, w))
    if energy < 1e-7:
        return float("nan"), 0.0
    n = len(w)
    spec = np.fft.rfft(w * np.hanning(n), 2 * n)
    ac = np.fft.irfft(np.abs(spec) ** 2)[:n]
    ac = ac / (ac[0] + 1e-12)
    lo, hi = int(sr / F0_MAX), min(n - 1, int(sr / F0_MIN))
    lag = lo + int(np.argmax(ac[lo:hi]))
    peak = float(ac[lag])
    if peak < VOICING_MIN_CORR:
        return float("nan"), peak
    return sr / lag, peak


@dataclass
class ProsodyTracker:
    """Keeps the last few seconds of frame features and summarises them at a pause."""

    sr: int = 16000
    history_s: float = 4.0
    _prev: np.ndarray | None = None
    energy: deque = field(default_factory=deque)
    f0: deque = field(default_factory=deque)
    all_f0: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.maxlen = int(self.history_s / 0.02)
        self.energy = deque(maxlen=self.maxlen)
        self.f0 = deque(maxlen=self.maxlen)

    def push(self, frame: np.ndarray) -> None:
        win = frame if self._prev is None else np.concatenate([self._prev, frame])
        self._prev = frame
        rms = float(np.sqrt(np.mean(frame.astype(np.float64) ** 2)) + 1e-8)
        self.energy.append(20.0 * np.log10(rms))
        f0, _ = frame_f0(win, self.sr)
        self.f0.append(f0)
        if np.isfinite(f0):
            self.all_f0.append(f0)

    def features(self, silence_s: float) -> dict[str, float]:
        """Prosodic summary of the speech that preceded the current pause.

        `silence_s` (from the VAD) tells us how many trailing frames are silence, so the
        "last speech" window excludes them.
        """
        e = np.array(self.energy)
        f = np.array(self.f0)
        n_sil = min(len(e), int(round(silence_s / 0.02)))
        e_sp, f_sp = e[: len(e) - n_sil], f[: len(f) - n_sil]
        out = {k: float("nan") for k in FEATURE_NAMES}
        if len(e_sp) < 5:
            return out
        voiced = np.isfinite(f_sp)
        st = 12.0 * np.log2(np.where(voiced, f_sp, 1.0) / 100.0)
        med = np.median(12.0 * np.log2(np.array(self.all_f0) / 100.0)) if len(self.all_f0) > 10 else np.nan
        vidx = np.where(voiced)[0]
        if len(vidx) >= 3:
            last = vidx[-min(15, len(vidx)) :]
            out["f0_final_rel"] = float(np.mean(st[last[-3:]]) - med) if np.isfinite(med) else float("nan")
            out["f0_slope"] = float(np.polyfit(last * 0.02, st[last], 1)[0]) if len(last) >= 4 else float("nan")
            # length of the final voiced run (final lengthening)
            run = 1
            while run < len(vidx) and vidx[-run - 1] == vidx[-run] - 1:
                run += 1
            out["final_voiced_ms"] = run * 20.0
            out["gap_since_voiced_ms"] = (len(f_sp) - 1 - vidx[-1]) * 20.0
        tail = e_sp[-15:]
        out["energy_slope"] = float(np.polyfit(np.arange(len(tail)) * 0.02, tail, 1)[0])
        out["energy_final_rel"] = float(np.mean(e_sp[-5:]) - np.percentile(e_sp, 90))
        out["voiced_ratio_2s"] = float(np.mean(voiced[-100:]))
        # syllable-rate proxy: energy peaks per second over the last 2 s of speech
        seg = e_sp[-100:]
        if len(seg) >= 10:
            sm = np.convolve(seg, np.ones(3) / 3, mode="same")
            peaks = (sm[1:-1] > sm[:-2]) & (sm[1:-1] > sm[2:]) & (sm[1:-1] > np.percentile(seg, 60))
            out["rate_peaks_per_s"] = float(peaks.sum() / (len(seg) * 0.02))
        return out


FEATURE_NAMES = (
    "f0_final_rel",
    "f0_slope",
    "final_voiced_ms",
    "gap_since_voiced_ms",
    "energy_slope",
    "energy_final_rel",
    "voiced_ratio_2s",
    "rate_peaks_per_s",
)

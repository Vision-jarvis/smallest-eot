"""Barge-in arbiter: `classify(audio_since_onset, partials) -> {label, conf, decided_at_ms}`.

Two small gradient-boosted heads over arbiter/features.py: one on the first 300 ms after onset
(the statement's decision window), one on the first 600 ms (the streaming-update horizon).
Policy: commit at 300 ms if the 300 ms head is confident (max class prob >= `commit_conf`,
tuned on a held-out part of train), otherwise decide at 600 ms. A barge-in therefore always
either stops the agent within 300 ms or is reported as a late (missed-within-300 ms) stop.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from arbiter.features import FEATURES, PRE_MS, window_features

ART = Path(__file__).resolve().parents[1] / "models/artifacts/arbiter.pkl"
LABELS = ("backchannel", "barge_in", "noise")


class Arbiter:
    def __init__(self, head300, head600, commit_conf: float) -> None:
        self.h300, self.h600, self.commit_conf = head300, head600, commit_conf

    @staticmethod
    def _x(audio: np.ndarray, sr: int, ms: int) -> np.ndarray:
        f = window_features(audio, sr, PRE_MS / 1000, ms)
        return np.array([[f[k] for k in FEATURES]])

    def classify(self, audio_since_onset: np.ndarray, partials=None, sr: int = 16000) -> dict:
        """`audio_since_onset` starts PRE_MS (100 ms) before the trigger; partials unused in v0."""
        p = self.h300.predict_proba(self._x(audio_since_onset, sr, 300))[0]
        if p.max() >= self.commit_conf:
            return {"label": LABELS[int(p.argmax())], "conf": float(p.max()), "decided_at_ms": 300}
        p = self.h600.predict_proba(self._x(audio_since_onset, sr, 600))[0]
        return {"label": LABELS[int(p.argmax())], "conf": float(p.max()), "decided_at_ms": 600}

    def save(self, path: Path = ART) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps({"h300": self.h300, "h600": self.h600, "commit_conf": self.commit_conf}))

    @classmethod
    def load(cls, path: Path = ART) -> "Arbiter":
        d = pickle.loads(path.read_bytes())
        return cls(d["h300"], d["h600"], d["commit_conf"])

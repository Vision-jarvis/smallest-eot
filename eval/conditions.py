"""Channel conditions applied to a test set before replay (evaluation-only stress slices).

* clean:     the audio as distributed.
* mulaw8k:   G.711 mu-law 8 kHz phone leg (core.audio.mulaw_roundtrip).
* snr10/snr5: additive MUSAN noise (OpenSLR 17, CC BY 4.0) at 10 / 5 dB SNR over speech,
             then the mu-law leg is NOT applied (one factor at a time).

The noise clip and offset are seeded by turn id, so every detector sees identical audio.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from core.audio import mix_at_snr, mulaw_roundtrip, resample
from eval.data import Turn

ROOT = Path(__file__).resolve().parents[1]
MUSAN_NOISE = ROOT / "data/musan/musan/noise"
CONDITIONS = ("clean", "mulaw8k", "snr10", "snr5")


def _seed(turn_id: str) -> int:
    return int(hashlib.md5(turn_id.encode()).hexdigest()[:8], 16)


@lru_cache(maxsize=1)
def noise_files() -> tuple[Path, ...]:
    files = tuple(sorted(MUSAN_NOISE.rglob("*.wav")))
    if not files:
        raise FileNotFoundError(f"MUSAN noise not found under {MUSAN_NOISE}; see data/README.md")
    return files


def _noise(rng: np.random.Generator, sr: int) -> np.ndarray:
    import soundfile as sf

    f = noise_files()[int(rng.integers(len(noise_files())))]
    x, s = sf.read(f, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return resample(x, s, sr)


def apply(turn: Turn, condition: str) -> Turn:
    if condition == "clean":
        return turn
    rng = np.random.default_rng(_seed(turn.id))
    if condition == "mulaw8k":
        x = mulaw_roundtrip(turn.audio, turn.sr)
    elif condition.startswith("snr"):
        x = mix_at_snr(turn.audio, _noise(rng, turn.sr), float(condition[3:]), rng)
    else:
        raise ValueError(condition)
    return replace(turn, audio=x[: len(turn.audio)])



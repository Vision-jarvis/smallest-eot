"""A tiny FAKE shard in IndicVoices' exact Hindi schema, for pipeline tests only.

    python -m bharat_turn.fake_indicvoices --out data/_fake_indicvoices/hindi --n 24

Each fake "extempore segment" is two or three Chirp3 TTS Hindi clips from data/smart_turn_hi
joined by 0.3 to 1.5 s of low-level noise, with 0.6 s of noise at the end, under 8 fake
speaker ids. It exists so that bharat_turn.indicvoices_episodes and models.gpu.finetune can
be dry-run end to end before the IndicVoices terms are accepted. Nothing trained or scored on
it is a result.
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from core.audio import decode, resample

ROOT = Path(__file__).resolve().parents[1]
COLS = ["audio_filepath", "text", "duration", "lang", "samples", "verbatim", "normalized", "speaker_id", "scenario", "task_name",
        "gender", "age_group", "job_type", "qualification", "area", "district", "state", "occupation", "verification_report",
        "unsanitized_verbatim", "unsanitized_normalized"]


def make(out: Path, n: int, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    src = pd.read_parquet(sorted((ROOT / "data/smart_turn_hi").glob("hi_*.parquet"))[0])
    clips = [resample(*decode(a), 16000) for a in src["audio"].head(3 * n)]
    noise = lambda s: (rng.standard_normal(int(s * 16000)) * 2e-3).astype(np.float32)
    rows = []
    for i in range(n):
        k = int(rng.integers(2, 4))
        parts = []
        for j in range(k):
            parts.append(clips[(3 * i + j) % len(clips)])
            parts.append(noise(rng.uniform(0.3, 1.5) if j < k - 1 else 0.6))
        x = np.concatenate(parts)
        buf = io.BytesIO()
        sf.write(buf, x, 16000, format="WAV", subtype="PCM_16")
        rec, idx = i // 2, i % 2
        row = {c: "" for c in COLS}
        row.update({"audio_filepath": {"bytes": buf.getvalue(), "path": f"fakerec{rec:03d}_{idx}.wav"}, "duration": len(x) / 16000,
                    "lang": "hindi", "samples": len(x), "speaker_id": f"fake_spk_{rec % 8}", "scenario": "Extempore", "task_name": "Extempore"})
        rows.append(row)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "valid-00000-of-00001.parquet"
    pd.DataFrame(rows, columns=COLS).to_parquet(path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data/_fake_indicvoices/hindi")
    ap.add_argument("--n", type=int, default=24)
    a = ap.parse_args()
    print(make(a.out, a.n))

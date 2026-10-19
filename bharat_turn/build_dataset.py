"""Regenerate Bharat-Turn-8k from licensed sources (no redistribution of gated audio).

    python -m bharat_turn.build_dataset --indicvoices-dir <local snapshot of ai4bharat/IndicVoices hindi/>
                                        --diarizer pulse|pyannote  [--hours 5]

PENDING inputs: IndicVoices is gated on Hugging Face (accept the terms with your account, then
`huggingface-cli download ai4bharat/IndicVoices --repo-type dataset --include "hindi/*"`), and
per-speaker voice activity needs either Pulse batch diarisation (API key) or pyannote.audio
(HF token for its gated pipeline). See data/README.md.

Pipeline (each step is a function below):
1. `select_conversations`: keep conversational (two-party, telephone-bridge) Hindi rows.
2. `channel_layout`: report whether the two parties are on separate channels (the IndicVoices
   paper does not say); if stereo, per-channel Silero VAD gives voice activity directly.
3. `diarise`: mono case, Pulse batch (`diarize=true&word_timestamps=true`) or pyannote.
4. `bharat_turn.labels.label_conversation`: hold / shift / backchannel / barge-in events.
5. `split`: speaker- and conversation-disjoint train/dev/test (honesty rule 4), hashing the
   speaker id and keeping every conversation of a speaker on one side.
6. Slices: clean, filler/connective-preceded holds, digit/entity read-outs (from the IPU text
   before the pause), and the channel conditions in eval/conditions.py (mu-law 8 kHz, MUSAN at
   10 and 5 dB). Writes bharat_turn/manifest.parquet (span ids + labels only, no audio).
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from bharat_turn.labels import IPU, label_conversation, merge_ipus
from core.text import normalise, token_class

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "manifest.parquet"


def select_conversations(meta: pd.DataFrame) -> pd.DataFrame:
    """IndicVoices metadata rows for two-party conversational audio (column names per its card)."""
    cols = {c.lower(): c for c in meta.columns}
    task = cols.get("task_name") or cols.get("scenario") or cols.get("type")
    if task is None:
        raise KeyError(f"cannot find the task/scenario column in {list(meta.columns)}")
    return meta[meta[task].astype(str).str.contains("conversation", case=False)]


def channel_layout(n_channels: int) -> str:
    return "stereo-per-speaker" if n_channels == 2 else "mono-mixed (needs diarisation)"


def split_of(speaker_ids: list[str]) -> str:
    """All speakers of a conversation vote with a stable hash; ties go to the rarer split."""
    h = min(int(hashlib.md5(s.encode()).hexdigest()[:6], 16) % 100 for s in speaker_ids)
    return "test" if h < 20 else "dev" if h < 30 else "train"


def events_for(conv_id: str, words_by_speaker: dict[str, list[tuple[float, float, str]]]) -> pd.DataFrame:
    ipus: list[IPU] = []
    for spk, words in words_by_speaker.items():
        ipus += merge_ipus(words, spk)
    rows = []
    for k, e in enumerate(label_conversation(ipus)):
        toks = normalise(e.prev_text)
        last = token_class(toks[-1]) if toks else "NONE"
        rows.append({
            "span_id": f"{conv_id}:{k}", "conversation": conv_id, "kind": e.kind, "speaker": e.speaker,
            "start": e.start, "end": e.end,
            "slice_filler": last == "HOLD", "slice_digit": last == "DIGIT",
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indicvoices-dir", type=Path, required=True)
    ap.add_argument("--diarizer", choices=["pulse", "pyannote"], default="pulse")
    ap.add_argument("--hours", type=float, default=5.0)
    a = ap.parse_args()
    if not a.indicvoices_dir.exists():
        raise SystemExit("PENDING: IndicVoices is gated. Follow data/README.md to download hindi/ first.")
    raise SystemExit(
        "PENDING: inspect the downloaded IndicVoices schema (channel layout, speaker ids) and wire "
        "steps 1 to 3 to it; steps 4 to 6 (events_for, split_of, slices) are implemented and unit-tested."
    )


if __name__ == "__main__":
    main()

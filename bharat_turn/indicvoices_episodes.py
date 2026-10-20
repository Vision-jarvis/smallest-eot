"""Real-Hindi pause episodes from IndicVoices extempore audio (PENDING: IndicVoices terms).

    # 1. after accepting the terms on huggingface.co/datasets/ai4bharat/IndicVoices:
    huggingface-cli download ai4bharat/IndicVoices --repo-type dataset \
        --include "hindi/valid-*" "hindi/train-0000*" --local-dir data/indicvoices
    # 2. look at the real schema and task names once (prints, writes nothing):
    python -m bharat_turn.indicvoices_episodes --src data/indicvoices/hindi --inspect
    # 3. build (CPU only, about 1 to 2 h for 6,000 pauses with 4 workers):
    python -m bharat_turn.indicvoices_episodes --src data/indicvoices/hindi --max-pauses 6000

What it makes. One row per pause *window*, in Smart Turn's training schema (audio bytes,
endpoint_bool, language, dataset) plus the span keys our eot-bench metric code needs
(rec_id, span_index, label, span_dur, offset_s), so the same file trains the Smart Turn
fine-tune and scores dev Lat@5% / FCR@300 with eval/metrics.py.

How a pause is found and labelled (no ASR, no API, no oracle words):
* Rows are kept if `task_name` matches --task-regex (default "extempore"): one speaker
  answering a prompt in spontaneous Hindi, the closest public stand-in for a caller's turn.
* Silero VAD (core/vad.py) on the 16 kHz waveform gives speech runs; runs closer than
  --merge-gap-s are merged (that removes stop-closure gaps, not pauses).
* hold = an internal silence of 0.2 to 5.0 s with speech after it (eot-bench's hold filter).
* eot  = the trailing silence after the last speech run of the *last segment of a
  recording*. IndicVoices ships annotator-cut segments; a segment end is not always the end
  of the answer. If --recording-regex groups segments of one recording (inspect the paths
  first), only the last segment's end is EoT and the other segment ends are dropped. If no
  grouping is possible every segment end is used and the rows carry eot_rule="segment_end"
  so they can be filtered or sliced. This is the main label caveat; build_report.json
  counts both.
* A window ends `offset_s` into the pause (0.2 s for dev/test, the eot-bench Smart Turn
  score point; train adds --train-offsets) and holds the last <= 8 s of audio before that.
  It never contains audio from after the window end (causal).
* Shortcut guard: an EoT window needs >= offset_s of *real* trailing audio (never padded),
  otherwise the pause is dropped and counted, so "file ends in digital zeros" cannot
  become the EoT cue. build_report.json logs the tail RMS per class for a sanity check.
* Split: speaker-disjoint train/dev/test (70/10/20) with bharat_turn.build_dataset.split_of
  on speaker_id (honesty rule 4). Gated audio stays under data/ (git-ignored), never
  redistributed.
"""

from __future__ import annotations

import argparse
import io
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from bharat_turn.build_dataset import split_of
from core.audio import decode, resample

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/indicvoices_pauses"
SR = 16000
WINDOW_S = 8.0
MIN_HOLD, MAX_HOLD = 0.2, 5.0
DEV_OFFSET = 0.2


@dataclass(frozen=True)
class Cfg:
    task_regex: str = "extempore"
    recording_regex: str = r"^(?P<rec>.+?)[_-](?P<idx>\d+)$"
    merge_gap_s: float = 0.1
    speech_th: float = 0.5
    min_speech_s: float = 0.1
    max_holds_per_segment: int = 4
    train_offsets: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5)


def speech_runs(p: np.ndarray, hop_s: float, cfg: Cfg) -> list[tuple[float, float]]:
    """(start, end) seconds of speech from per-window Silero probabilities."""
    on = p >= cfg.speech_th
    runs, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            runs.append([i * hop_s, j * hop_s])
            i = j
        else:
            i += 1
    merged: list[list[float]] = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < cfg.merge_gap_s:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(a, b) for a, b in merged if b - a >= cfg.min_speech_s]


def pauses(runs: list[tuple[float, float]], dur: float, is_last_segment: bool, cfg: Cfg) -> list[dict]:
    """Labelled pauses of one segment: internal holds, plus the trailing EoT if allowed."""
    out = []
    for k in range(len(runs) - 1):
        a, b = runs[k][1], runs[k + 1][0]
        if MIN_HOLD <= b - a <= MAX_HOLD:
            out.append({"label": "hold", "start": a, "end": b})
    holds = out[: cfg.max_holds_per_segment] if cfg.max_holds_per_segment else out
    if runs and is_last_segment:
        holds.append({"label": "eot", "start": runs[-1][1], "end": dur})
    return holds


def _wav_bytes(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def recording_key(path: str, cfg: Cfg) -> tuple[str, int] | None:
    stem = Path(str(path)).stem
    m = re.match(cfg.recording_regex, stem) if cfg.recording_regex else None
    return (m.group("rec"), int(m.group("idx"))) if m else None


def windows_for_segment(row: dict, is_last: bool, eot_rule: str, cfg: Cfg) -> tuple[list[dict], Counter]:
    """All pause windows for one IndicVoices row (a segment)."""
    from core.vad import speech_prob_trace

    stats: Counter = Counter()
    x, sr = decode(row["audio_filepath"])
    x = resample(x, sr, SR)
    dur = len(x) / SR
    tr = speech_prob_trace(x, SR)
    hop = float(tr[1, 0] - tr[0, 0]) if len(tr) > 1 else 0.032
    runs = speech_runs(tr[:, 1], hop, cfg)
    spk = str(row.get("speaker_id") or "unknown")
    split = split_of([spk])
    seg_id = str(row.get("_seg_id"))
    rows = []
    for k, pz in enumerate(pauses(runs, dur, is_last, cfg)):
        n_before = len(rows)
        span_dur = pz["end"] - pz["start"]
        offsets = (DEV_OFFSET,) if split != "train" else cfg.train_offsets
        for off in offsets:
            if pz["label"] == "hold" and off >= span_dur:
                continue  # the user would already be talking again
            if pz["label"] == "eot" and span_dur < off - 1e-9:
                stats["eot_dropped_short_tail"] += 1
                continue  # never pad an EoT with synthetic silence (shortcut guard)
            t_end = pz["start"] + off
            w = x[max(0, int(round((t_end - WINDOW_S) * SR))) : int(round(t_end * SR))]
            tail = w[-int(0.1 * SR):]
            rows.append({
                "audio": {"bytes": _wav_bytes(w), "path": None},
                "id": f"{seg_id}:{k}:{int(off * 1000)}", "language": "hin", "endpoint_bool": pz["label"] == "eot",
                "dataset": "indicvoices_extempore", "synthetic": False, "midfiller": None, "endfiller": None,
                "speaker_id": spk, "split": split, "rec_id": seg_id, "span_index": k, "label": pz["label"],
                "span_dur": float(span_dur), "offset_s": float(off), "eot_rule": eot_rule if pz["label"] == "eot" else None,
                "tail_rms": float(np.sqrt(np.mean(tail.astype(np.float64) ** 2) + 1e-12)),
            })
            stats[f"{split}_{pz['label']}_windows"] += 1
        if len(rows) > n_before:
            stats["pauses"] += 1
            stats[f"{split}_{pz['label']}"] += 1
    stats["segments"] += 1
    return rows, stats


def _work(args) -> tuple[list[dict], Counter]:
    row, is_last, rule, cfg = args
    try:
        return windows_for_segment(row, is_last, rule, cfg)
    except Exception as e:  # a corrupt clip must not kill a 2-hour build; it is counted
        return [], Counter({f"error_{type(e).__name__}": 1})


def plan_segments(meta: pd.DataFrame, cfg: Cfg) -> list[tuple[int, bool, str]]:
    """(row index, is_last_segment_of_recording, eot_rule) for every kept row."""
    keys = [recording_key(p, cfg) for p in meta["_path"]]
    grouped = sum(k is not None for k in keys) >= 0.9 * max(1, len(keys))
    if not grouped:
        return [(i, True, "segment_end") for i in range(len(meta))]
    last: dict[str, int] = {}
    for k in keys:
        if k is not None:
            last[k[0]] = max(last.get(k[0], -1), k[1])
    return [(i, k is not None and k[1] == last[k[0]], "recording_end") for i, k in enumerate(keys)]


def _path_of(a) -> str:
    return str(a.get("path") or "") if isinstance(a, dict) else ""


def load_rows(src: Path, cfg: Cfg, max_rows: int | None) -> pd.DataFrame:
    files = sorted(src.glob("*.parquet"))
    if not files:
        raise SystemExit(f"PENDING: no IndicVoices parquet under {src}. Accept the terms and download first (data/README.md).")
    parts, n = [], 0
    for f in files:
        df = pd.read_parquet(f)
        df = df[df["task_name"].astype(str).str.contains(cfg.task_regex, case=False, regex=True)]
        df = df.assign(_path=[_path_of(a) for a in df["audio_filepath"]], _seg_id=[f"{f.stem}:{i}" for i in df.index])
        parts.append(df)
        n += len(df)
        if max_rows and n >= max_rows:
            break
    out = pd.concat(parts, ignore_index=True)
    return out.iloc[:max_rows] if max_rows else out


def inspect(src: Path) -> None:
    f = sorted(src.glob("*.parquet"))[0]
    df = pd.read_parquet(f)
    print(f"{f.name}: {len(df)} rows, columns {list(df.columns)}")
    print("task_name:", df["task_name"].value_counts().head(20).to_dict())
    print("scenario:", df["scenario"].value_counts().head(10).to_dict() if "scenario" in df else "-")
    print("duration s: ", df["duration"].describe().round(1).to_dict())
    print("example audio paths:", [_path_of(a) for a in df["audio_filepath"].head(5)])
    print("speakers in shard:", df["speaker_id"].nunique())


def build(src: Path, cfg: Cfg, max_pauses: int, max_rows: int | None, workers: int, out: Path = OUT) -> dict:
    meta = load_rows(src, cfg, max_rows)
    plan = plan_segments(meta, cfg)
    recs = meta.drop(columns=[c for c in ("verbatim", "unsanitized_verbatim", "unsanitized_normalized") if c in meta]).to_dict("records")
    rows, stats = [], Counter()
    jobs = [(recs[i], last, rule, cfg) for i, last, rule in plan]
    with ProcessPoolExecutor(workers) as ex:
        for b in range(0, len(jobs), 64):  # batches, so --max-pauses stops the build early
            for r, s in ex.map(_work, jobs[b : b + 64], chunksize=4):
                rows.extend(r)
                stats.update(s)
            if stats["pauses"] >= max_pauses:
                break
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    report = {"source": str(src), "task_regex": cfg.task_regex, "rows_kept": len(meta), "eot_rule": plan[0][2] if plan else None,
              "windows": len(df), "counts": dict(stats), "cfg": cfg.__dict__}
    if len(df):
        for split, g in df.groupby("split"):
            g.drop(columns=["tail_rms"]).to_parquet(out / f"{split}.parquet")
        report["speakers_per_split"] = df.groupby("split")["speaker_id"].nunique().to_dict()
        report["tail_rms_median_by_label"] = df.groupby("label")["tail_rms"].median().to_dict()
        report["pauses_per_split"] = df[df.offset_s == DEV_OFFSET].groupby(["split", "label"]).size().unstack(fill_value=0).to_dict()
    (out / "build_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=ROOT / "data/indicvoices/hindi")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--task-regex", default=Cfg.task_regex)
    ap.add_argument("--recording-regex", default=Cfg.recording_regex, help="'' disables grouping (every segment end = EoT)")
    ap.add_argument("--max-pauses", type=int, default=6000)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if not a.src.exists():
        raise SystemExit("PENDING: IndicVoices is gated. Accept the terms on Hugging Face, then download hindi/ (data/README.md).")
    if a.inspect:
        inspect(a.src)
        return
    cfg = Cfg(task_regex=a.task_regex, recording_regex=a.recording_regex)
    print(json.dumps(build(a.src, cfg, a.max_pauses, a.max_rows, a.workers, a.out), indent=1, default=str))


if __name__ == "__main__":
    main()

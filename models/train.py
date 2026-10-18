"""Train Pulse-EoT-hi v0 on CPU (minutes, not hours).

    python -m models.train --variants full audio_only ...

Data (all disjoint from the eot-bench Hindi test set):
  * train: Smart Turn v3.2 *train* split, Hindi clips -> single-pause episodes
           (+ optional real-human non-Hindi clips with no lexical features, `--human`).
  * dev:   Smart Turn v3.2 *test* split, Hindi clips -> episodes. Used for model selection and
           for freezing the eot-bench knob triple. Never used to fit weights.

Steps: episodes -> streaming ASR replay (Vosk) -> per-hop features through the same
FeatureState the runtime uses (no train/serve skew) -> out-of-fold lexical logits ->
HistGradientBoosting hazard head -> ONNX export with a parity check.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from core.asr import visible_text
from eval import asr_cache
from eval.harness import grid_times, hop_index, replay
from models import lexical as lexmod
from models.episodes import EPISODES, build_and_save, load_episodes
from models.pulse_eot_hi import ART, FEATURES, VARIANTS, FeatureState, vectorise

ROOT = Path(__file__).resolve().parents[1]
PAD_S = 3.0
FEATS_DIR = ROOT / "data/cache/features"


class FeatureRecorder:
    """A detector that scores nothing but records the feature dict at every hop."""

    name = "feature-recorder"
    score_point = None
    uses_text = True

    def __init__(self) -> None:
        self.state = FeatureState(None)
        self.rows: list[dict] = []

    def start_turn(self, sr, agent_text):
        self.state.start_turn(sr)
        self.rows = []

    def push_audio(self, frame):
        self.state.push_audio(frame)

    def push_partial(self, p):
        self.state.push_partial(p)

    def silence_s(self, t):
        return self.state.vad.trailing_silence(t)

    def score(self, t, need):
        f = self.state.features(t)
        f["t"] = t
        f["in_speech"] = float(self.state.in_speech())
        f["text"] = visible_text(self.state.partials, t)[0]
        self.rows.append(f)
        return 0.0


def _featurise(args) -> list[dict]:
    turn, partials = args
    rec = FeatureRecorder()
    replay(rec, turn, partials, pad_s=PAD_S, lazy=False)
    by_hop = {hop_index(r["t"]): r for r in rec.rows}
    s = turn.spans[0]
    out = []
    for t in grid_times(s.start, s.end):
        r = by_hop.get(hop_index(t))
        if r is None:
            continue
        out.append({**r, "id": turn.id, "silence_dur": round(t - s.start, 6), "span_dur": s.dur, "label": s.label, "dataset": turn.meta.get("dataset", "")})
    return out


def featurise(name: str, use_asr: bool, workers: int) -> pd.DataFrame:
    FEATS_DIR.mkdir(parents=True, exist_ok=True)
    out = FEATS_DIR / f"{name}.parquet"
    if out.exists():
        return pd.read_parquet(out)
    turns = load_episodes(name)
    parts: dict = {}
    if use_asr:
        asr_cache.build(turns, name, PAD_S, workers)
        parts = asr_cache.load(name)
    rows = []
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_featurise, [(t, parts.get(t.id, [])) for t in turns], chunksize=4):
            rows.extend(r)
    df = pd.DataFrame(rows)
    df.to_parquet(out)
    return df


def add_lexical(train: pd.DataFrame, others: list[pd.DataFrame], drop: frozenset[str]) -> tuple[lexmod.LexicalHead, pd.DataFrame, list[pd.DataFrame]]:
    """Fit the lexical head on train episode texts; OOF logits for train hops, full head for others."""
    from sklearn.model_selection import GroupKFold

    ep = train[train.has_text > 0].sort_values("silence_dur").groupby("id").last()
    texts, y, ids = ep["text"].tolist(), (ep["label"] == "eot").to_numpy(int), ep.index.to_numpy()
    train = train.copy()
    train["lex_logit"] = np.nan
    fold_of = {}
    for k, (_, te) in enumerate(GroupKFold(5).split(texts, y, ids)):
        for i in te:
            fold_of[ids[i]] = k
    for k in range(5):
        tr = [i for i in range(len(ids)) if fold_of[ids[i]] != k]
        head = lexmod.fit([texts[i] for i in tr], y[tr], drop)
        m = train["id"].map(fold_of) == k
        train.loc[m, "lex_logit"] = [head.logit(t) for t in train.loc[m, "text"]]
    full = lexmod.fit(texts, y, drop)
    outs = []
    for o in others:
        o = o.copy()
        o["lex_logit"] = [full.logit(t) for t in o["text"]]
        outs.append(o)
    return full, train, outs


def matrix(df: pd.DataFrame, names: list[str], drop: frozenset[str]) -> np.ndarray:
    d = df.copy()
    for c in drop:  # class removed: its one-hots fold into OTHER
        for i in (1, 2):
            d[f"c{i}_OTHER"] = d[f"c{i}_OTHER"] + d[f"c{i}_{c}"]
            d[f"c{i}_{c}"] = 0.0
        if c == "TAG":
            d["tag_q"] = 0.0
    return np.stack([vectorise(r, names) for r in d[names].to_dict("records")])


def dev_predictions(df: pd.DataFrame, p: np.ndarray) -> pd.DataFrame:
    """Hop features + scores -> eot-bench-schema predictions (speech hops score 0, as at runtime)."""
    return pd.DataFrame(
        {"id": df["id"], "language": "hi", "span_index": 0, "timestamp": df["t"], "silence_dur": df["silence_dur"], "p_eot": np.where(df["in_speech"] > 0, 0.0, p), "label": df["label"]}
    )


def export_onnx(model, n_features: int, path: Path, X_check: np.ndarray) -> None:
    from skl2onnx import to_onnx
    import onnxruntime as ort

    onx = to_onnx(model, X_check[:1].astype(np.float32), options={id(model): {"zipmap": False}}, target_opset=17)
    path.write_bytes(onx.SerializeToString())
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    p_onnx = sess.run(None, {sess.get_inputs()[0].name: X_check.astype(np.float32)})[1][:, 1]
    p_skl = model.predict_proba(X_check)[:, 1]
    err = float(np.max(np.abs(p_onnx - p_skl)))
    if err > 1e-4:
        raise RuntimeError(f"ONNX parity failed: max |dp| = {err}")


def train_variant(variant: str, tr: pd.DataFrame, dev: pd.DataFrame, tag: str = "") -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier

    from eval import metrics

    cfg = VARIANTS[variant]
    drop = frozenset(cfg["drop_classes"])
    names = cfg["features"]
    head, tr_l, (dev_l,) = add_lexical(tr, [dev], drop)
    # risk set: hops at which the span is still silent and the VAD agrees (speech hops score 0)
    trm = tr_l[tr_l.in_speech == 0]
    X, y = matrix(trm, names, drop), (trm["label"] == "eot").to_numpy(int)
    Xd = matrix(dev_l, names, drop)
    best = None
    for lr, leaves, iters in [(0.05, 15, 200), (0.05, 31, 300), (0.1, 15, 150), (0.03, 31, 500)]:
        m = HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=leaves, max_iter=iters, l2_regularization=1.0, random_state=0)
        m.fit(X, y)
        _, s = metrics.summarise(dev_predictions(dev_l, m.predict_proba(Xd)[:, 1]))
        score = s["Lat@5%"] if s["Lat@5%"] is not None else 9.0
        print(f"  [{variant}] lr={lr} leaves={leaves} iters={iters}: dev Lat@5%={score:.3f} FCR@300={s['FCR@300']} auc={s['auc']:.3f}", flush=True)
        if best is None or score < best[0]:
            best = (score, m, (lr, leaves, iters), s)
    _, model, hp, dev_summary = best
    ART.mkdir(parents=True, exist_ok=True)
    stem = f"pulse_eot_hi_{variant}{tag}"
    export_onnx(model, len(names), ART / f"{stem}.onnx", Xd[:2000])
    lex_name = f"{stem}_lexical.npz"
    head.save(ART / lex_name)
    meta = {
        "variant": variant,
        "features": names,
        "drop_classes": sorted(drop),
        "lexical": lex_name if "lex_logit" in names or "has_text" in names else None,
        "hyperparams": {"learning_rate": hp[0], "max_leaf_nodes": hp[1], "max_iter": hp[2]},
        "n_train_hops": int(len(X)),
        "n_train_episodes": int(tr["id"].nunique()),
        "dev_summary": {k: dev_summary[k] for k in ("FCR@300", "FCR@600", "Lat@5%", "Lat@10%", "auc", "ap")},
        "dev_frozen_knobs_lat5": dev_summary["operating_points"]["lat_at_5pct"],
        "dev_frozen_knobs_fcr300": dev_summary["operating_points"]["fcr_at_300ms"],
    }
    (ART / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["full"])
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--human", action="store_true", help="add real-human non-Hindi clips (no lexical features)")
    ap.add_argument("--channel", default="clean", choices=["clean", "mulaw8k"], help="train-time channel (8 kHz ablation)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    sfx = "" if a.channel == "clean" else f"_{a.channel}"
    names = {"train": f"train_hi{sfx}", "dev": "dev_hi"}
    if not (EPISODES / f"{names['dev']}.parquet").exists():
        build_and_save(names["dev"], "test", None)
    if not (EPISODES / f"{names['train']}.parquet").exists():
        build_and_save(names["train"], "train", a.max_train)
        if a.channel == "mulaw8k":
            _mulaw_episodes(names["train"])
    tr = featurise(names["train"], True, a.workers)
    if a.human:
        if not (EPISODES / "train_human.parquet").exists():
            from models.episodes import build_human

            build_human("train_human", a.max_train)
        tr = pd.concat([tr, featurise("train_human", False, a.workers)], ignore_index=True)
    dev = featurise(names["dev"], True, a.workers)
    tag = sfx + ("_human" if a.human else "")
    for v in a.variants:
        meta = train_variant(v, tr, dev, tag)
        print(json.dumps({"variant": v, **meta["dev_summary"]}))


def _mulaw_episodes(name: str) -> None:
    """Rewrite an episode file with every clip passed through the mu-law 8 kHz phone leg."""
    import io

    import soundfile as sf

    from core.audio import mulaw_roundtrip

    p = EPISODES / f"{name}.parquet"
    df = pd.read_parquet(p)
    out = []
    for fl in df["flac"]:
        x, sr = sf.read(io.BytesIO(fl), dtype="float32")
        buf = io.BytesIO()
        sf.write(buf, mulaw_roundtrip(x, sr)[: len(x)], sr, format="FLAC")
        out.append(buf.getvalue())
    df["flac"] = out
    df.to_parquet(p)


if __name__ == "__main__":
    main()

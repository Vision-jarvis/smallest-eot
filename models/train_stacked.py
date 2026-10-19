"""Pulse-EoT-hi v0.1 ("stacked"): add the frozen Smart Turn v3.2 probability as a feature.

    python -m models.train_stacked --variants stacked stacked_audio_only smart_turn_only

Why stacking needs different data: Smart Turn v3.2 was trained on smart-turn-data-v3.2-train,
so its scores on those clips are in-sample (over-confident). The fusion head is therefore fit
only on the Smart Turn *test* split, which Smart Turn never trained on:

  * fusion-train: 60% of test-split clips (Hindi TTS clips with ASR text + real-human clips,
    mostly English, with no text), by a stable hash of the clip id;
  * dev: the other 40%, used for hyper-parameters and the frozen knob triple.

The lexical head is still fit on the *train*-split Hindi transcripts (disjoint from both).
eot-bench Hindi is never used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

from models import lexical as lexmod
from models.episodes import EPISODES, build_and_save
from models.export_onnx import export
from models.pulse_eot_hi import ART, VARIANTS
from models.train import GRID, dev_predictions, featurise, matrix


def _is_dev(i: str) -> bool:
    return int(hashlib.md5(i.encode()).hexdigest()[:4], 16) % 100 < 40


def lexical_head(train_feats: pd.DataFrame) -> lexmod.LexicalHead:
    ep = train_feats[train_feats.has_text > 0].sort_values("silence_dur").groupby("id").last()
    return lexmod.fit(ep["text"].tolist(), (ep["label"] == "eot").to_numpy(int))


def main() -> None:
    from sklearn.ensemble import HistGradientBoostingClassifier

    from eval import metrics

    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["stacked"])
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    if not (EPISODES / "dev_human.parquet").exists():
        build_and_save("dev_human", "test", None, pattern="human_*.parquet")
    hi = featurise("dev_hi", True, a.workers, smart_turn=True)
    hu = featurise("dev_human", False, a.workers, smart_turn=True)
    hu = hu.assign(text="", has_text=0.0)
    allf = pd.concat([hi.assign(src="hi_tts"), hu.assign(src="human")], ignore_index=True)
    head = lexical_head(featurise("train_hi", True, a.workers))
    allf["lex_logit"] = [head.logit(t) if t else np.nan for t in allf["text"]]
    dev_mask = allf["id"].map(_is_dev)
    tr, dev = allf[~dev_mask], allf[dev_mask]
    for v in a.variants:
        cfg = VARIANTS[v]
        names = cfg["features"]
        trm = tr[tr.in_speech == 0]
        X, y, Xd = matrix(trm, names, frozenset()), (trm["label"] == "eot").to_numpy(int), matrix(dev, names, frozenset())
        best = None
        for lr, leaves, iters in GRID:
            m = HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=leaves, max_iter=iters, l2_regularization=1.0, random_state=0).fit(X, y)
            _, s = metrics.summarise(dev_predictions(dev, m.predict_proba(Xd)[:, 1]))
            score = s["Lat@5%"] if s["Lat@5%"] is not None else 9.0
            print(f"  [{v}] lr={lr} leaves={leaves} iters={iters}: dev Lat@5%={score:.3f} FCR@300={s['FCR@300']} auc={s['auc']:.3f}", flush=True)
            if best is None or score < best[0]:
                best = (score, m, (lr, leaves, iters), s)
        _, model, hp, s = best
        stem = f"pulse_eot_hi_{v}"
        err = export(model, len(names), ART / f"{stem}.onnx", Xd[:5000])
        head.save(ART / f"{stem}_lexical.npz")
        per_src = {}
        for src in ("hi_tts", "human"):
            d = dev[dev.src == src]
            _, ss = metrics.summarise(dev_predictions(d, model.predict_proba(matrix(d, names, frozenset()))[:, 1]))
            per_src[src] = {k: ss[k] for k in ("FCR@300", "Lat@5%", "auc")}
        meta = {
            "variant": v, "features": names, "drop_classes": [], "smart_turn": True,
            "lexical": f"{stem}_lexical.npz" if "lex_logit" in names else None,
            "hyperparams": {"learning_rate": hp[0], "max_leaf_nodes": hp[1], "max_iter": hp[2]},
            "train_data": "60% of smart-turn-data-v3.2-test clips (Hindi TTS + real-human non-Hindi); lexical head on smart-turn train Hindi",
            "n_train_hops": int(len(X)), "n_train_episodes": int(tr["id"].nunique()), "onnx_max_abs_diff_vs_sklearn": err,
            "dev_summary": {k: s[k] for k in ("FCR@300", "FCR@600", "Lat@5%", "Lat@10%", "auc", "ap")},
            "dev_by_source": per_src,
            "dev_frozen_knobs_lat5": s["operating_points"]["lat_at_5pct"],
            "dev_frozen_knobs_fcr300": s["operating_points"]["fcr_at_300ms"],
        }
        (ART / f"{stem}.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps({"variant": v, **meta["dev_summary"], "by_source": per_src}))


if __name__ == "__main__":
    main()

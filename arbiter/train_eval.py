"""Train and evaluate the barge-in arbiter on the proxy overlap set.

    python -m arbiter.train_eval

Metrics (problem statement Section 7):
  FBR = #(backchannel or noise classified barge_in) / #(backchannel + noise)
  MBR = #(true barge-ins not flagged barge_in within 300 ms) / #(barge-ins)
  reaction latency p50/p90 = onset -> barge_in decision, over detected barge-ins
  arbiter score = 1 - (FBR + MBR) / 2
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from arbiter.data import SR, build
from arbiter.features import FEATURES, energy_onset, window_features
from arbiter.model import LABELS, Arbiter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/arbiter"
CACHE = ROOT / "data/cache/arbiter"


def _feat(s: dict) -> dict | None:
    on = energy_onset(s["audio"], SR)
    if on is None:
        return None
    row = {"id": s["id"], "source": s["source"], "label": s["label"], "language": s["language"], "onset": on}
    for ms in (300, 600):
        f = window_features(s["audio"], SR, on, ms)
        row.update({f"{k}@{ms}": v for k, v in f.items()})
    return row


def featurise(split: str, workers: int) -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{split}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    samples = build(split)
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for r in ex.map(_feat, samples, chunksize=16) if r is not None]
    df = pd.DataFrame(rows)
    df.to_parquet(p)
    return df


def _head():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0, class_weight="balanced", random_state=0)


def decide(h300, h600, df: pd.DataFrame, conf: float) -> pd.DataFrame:
    X3 = df[[f"{k}@300" for k in FEATURES]].to_numpy()
    X6 = df[[f"{k}@600" for k in FEATURES]].to_numpy()
    p3, p6 = h300.predict_proba(X3), h600.predict_proba(X6)
    early = p3.max(1) >= conf
    lab = np.where(early, p3.argmax(1), p6.argmax(1))
    return pd.DataFrame({"true": df["label"].to_numpy(), "pred": np.array(LABELS)[lab], "decided_at_ms": np.where(early, 300, 600), "conf": np.where(early, p3.max(1), p6.max(1))})


def scores(d: pd.DataFrame) -> dict:
    neg = d[d.true != "barge_in"]
    pos = d[d.true == "barge_in"]
    fbr = float((neg.pred == "barge_in").mean())
    hit300 = (pos.pred == "barge_in") & (pos.decided_at_ms <= 300)
    mbr = float(1 - hit300.mean())
    det = pos[pos.pred == "barge_in"].decided_at_ms
    return {
        "FBR": fbr,
        "MBR": mbr,
        "arbiter_score": 1 - (fbr + mbr) / 2,
        "reaction_p50_ms": float(np.percentile(det, 50)) if len(det) else None,
        "reaction_p90_ms": float(np.percentile(det, 90)) if len(det) else None,
        "accuracy": float((d.true == d.pred).mean()),
        "n": {k: int(v) for k, v in d.true.value_counts().items()},
        "confusion": pd.crosstab(d.true, d.pred).to_dict(),
    }


def main() -> None:
    workers = max(1, (os.cpu_count() or 2) - 2)
    tr, te = featurise("train", workers), featurise("test", workers)
    # hold out 20% of training *sources* to tune the commit confidence
    rng = np.random.default_rng(0)
    srcs = tr.source.unique()
    hold = set(rng.choice(srcs, size=len(srcs) // 5, replace=False))
    fit, val = tr[~tr.source.isin(hold)], tr[tr.source.isin(hold)]
    y = lambda d: d["label"].map({k: i for i, k in enumerate(LABELS)}).to_numpy()
    h300, h600 = _head(), _head()
    h300.fit(fit[[f"{k}@300" for k in FEATURES]].to_numpy(), y(fit))
    h600.fit(fit[[f"{k}@600" for k in FEATURES]].to_numpy(), y(fit))
    grid = [(c, scores(decide(h300, h600, val, c))) for c in np.round(np.arange(0.4, 1.0, 0.05), 2)]
    conf, _ = max(grid, key=lambda g: (g[1]["arbiter_score"], -g[0]))
    # refit on all train with the frozen commit confidence, evaluate once on test
    h300.fit(tr[[f"{k}@300" for k in FEATURES]].to_numpy(), y(tr))
    h600.fit(tr[[f"{k}@600" for k in FEATURES]].to_numpy(), y(tr))
    Arbiter(h300, h600, float(conf)).save()
    d = decide(h300, h600, te, conf)
    res = {"commit_conf_tuned_on_train_holdout": float(conf), "test": scores(d), "test_always_wait_600": scores(decide(h300, h600, te, 1.01)), "test_always_commit_300": scores(decide(h300, h600, te, 0.0))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: (v if k != "test" else {kk: vv for kk, vv in v.items() if kk != "confusion"}) for k, v in res.items() if k != "test_always_wait_600"}, indent=1))
    print(pd.crosstab(d.true, d.pred))


if __name__ == "__main__":
    main()



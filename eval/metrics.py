"""eot-bench policy sweep and the problem statement's metrics.

Definitions (identical to livekit/eot-bench `eot_harness/metrics.py`, checked number-for-number
in tests/test_metrics_parity.py):

* Spans: `hold` spans are kept only if 0.2 s <= duration <= 5.0 s; every `eot` span is kept.
* Policy knobs: threshold th, action_delay d, timeout T (grid: th 0..1 step 0.01,
  d 0.2..1.0 step 0.1, T 1.0..3.5 step 0.5, T >= d).
* Fire time in a span: max(d, first silence_dur at which p_eot > th) (trace mode), or
  max(d, score_point) if the single score at score_point exceeds th (score-point mode).
* Hold cut-off: the hold lasts longer than the fire time, or longer than T.
* EoT latency: fire time if it is <= T, else T (the timeout forces the turn end).
* FCR = mean hold cut-off; latency = mean EoT latency; detect = fraction of EoT spans fired
  by the model before T.
* Operating points: FCR@300/600 = min FCR s.t. latency <= 0.3/0.6 s; Lat@5%/10% = min
  latency s.t. FCR <= 5%/10%; Detect@5% = detect rate at the Lat@5% point.
* VAD baseline (eot-bench's "VAD baseline"): fire at a fixed silence delay on oracle spans.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

THRESHOLDS = np.round(np.arange(0.0, 1.01, 0.01), 2)
ACTION_DELAYS = np.round(np.arange(0.2, 1.01, 0.1), 2)
TIMEOUTS = np.round(np.arange(1.0, 3.51, 0.5), 2)
MIN_HOLD, MAX_HOLD = 0.2, 5.0
EPS = 1e-9


@dataclass
class SpanTable:
    """Per-span arrays needed by the sweep."""

    keys: pd.DataFrame  # id, span_index, label, span_dur
    is_hold: np.ndarray
    dur: np.ndarray
    # trace mode: per span, silence_dur grid and running max of p (monotone, for searchsorted)
    sil: list[np.ndarray]
    cummax: list[np.ndarray]
    score_point: float | None


def span_table(pred: pd.DataFrame, score_point: float | None = None) -> SpanTable:
    pred = pred.sort_values(["id", "span_index", "silence_dur", "timestamp"], kind="stable")
    g = pred.groupby(["id", "span_index"], sort=False)
    keys = g.agg(label=("label", "first"), span_dur=("silence_dur", "max")).reset_index()
    hold = keys["label"].to_numpy() == "hold"
    keep = ~hold | ((keys["span_dur"] >= MIN_HOLD - EPS) & (keys["span_dur"] <= MAX_HOLD + EPS))
    keys = keys[keep].reset_index(drop=True)
    keep_set = set(zip(keys["id"], keys["span_index"]))
    sil, cm = [], []
    for (rid, si), grp in g:
        if (rid, si) not in keep_set:
            continue
        s = grp["silence_dur"].to_numpy(dtype=float)
        p = grp["p_eot"].to_numpy(dtype=float)
        if score_point is not None:
            m = np.isclose(s, score_point, atol=1e-6)
            s, p = s[m][:1], p[m][:1]
        sil.append(s)
        cm.append(np.maximum.accumulate(p) if len(p) else p)
    # groupby iteration order equals keys order (both sorted by first appearance)
    return SpanTable(keys, keys["label"].to_numpy() == "hold", keys["span_dur"].to_numpy(float), sil, cm, score_point)


def first_crossing(st: SpanTable, th: float) -> np.ndarray:
    """Silence time of the first p_eot > th in each span (inf if never)."""
    out = np.full(len(st.sil), np.inf)
    for i, (s, c) in enumerate(zip(st.sil, st.cummax)):
        if len(c) == 0:
            continue
        j = int(np.searchsorted(c, th, side="right"))
        if j < len(c):
            out[i] = s[j]
    return out


def sweep(pred: pd.DataFrame, score_point: float | None = None, thresholds=THRESHOLDS, action_delays=ACTION_DELAYS, timeouts=TIMEOUTS) -> pd.DataFrame:
    st = span_table(pred, score_point)
    hd, hold = st.dur[st.is_hold], st.is_hold
    rows = []
    for th in thresholds:
        fire = first_crossing(st, float(th))
        hf, ef = fire[hold], fire[~hold]
        for d in action_delays:
            h_ft = np.maximum(d, hf)
            model_cut = np.isfinite(hf) & (hd > h_ft + EPS)
            e_ft = np.maximum(d, ef)
            for T in timeouts:
                if T + EPS < d:
                    continue
                cut = model_cut | (hd > T + EPS)
                det = np.isfinite(ef) & (e_ft <= T + EPS)
                lat = np.where(det, e_ft, T)
                rows.append(
                    {
                        "threshold": float(th),
                        "action_delay": float(d),
                        "timeout": float(T),
                        "cutoff_rate": float(cut.mean()),
                        "mean_latency": float(lat.mean()),
                        "detect_rate": float(det.mean()),
                        "n_hold": int(hold.sum()),
                        "n_eot": int((~hold).sum()),
                    }
                )
    return pd.DataFrame(rows)


def vad_baseline(pred: pd.DataFrame, delays=None) -> pd.DataFrame:
    """eot-bench's silence-only baseline on oracle spans: fire at a fixed delay."""
    st = span_table(pred)
    hd = st.dur[st.is_hold]
    if delays is None:
        delays = np.unique(np.concatenate([ACTION_DELAYS, TIMEOUTS]))
    return pd.DataFrame(
        [{"threshold": math.nan, "action_delay": float(d), "timeout": float(d), "cutoff_rate": float((hd > d + EPS).mean()), "mean_latency": float(d), "detect_rate": 1.0} for d in delays]
    )


def _best(sw: pd.DataFrame, budget_col: str, budget: float, opt_col: str) -> dict | None:
    f = sw[sw[budget_col] <= budget + EPS]
    if f.empty:
        return None
    # eot-bench tie-breaks: lower optimised value, then lower budget value, then higher threshold
    f = f.sort_values([opt_col, budget_col, "threshold", "action_delay", "timeout"], ascending=[True, True, False, True, True], kind="stable")
    return f.iloc[0].to_dict()


def operating_points(sw: pd.DataFrame) -> dict:
    out = {}
    for name, (bc, b, oc) in {
        "fcr_at_300ms": ("mean_latency", 0.3, "cutoff_rate"),
        "fcr_at_600ms": ("mean_latency", 0.6, "cutoff_rate"),
        "lat_at_5pct": ("cutoff_rate", 0.05, "mean_latency"),
        "lat_at_10pct": ("cutoff_rate", 0.10, "mean_latency"),
    }.items():
        best = _best(sw, bc, b, oc)
        out[name] = None if best is None else {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in best.items()}
    return out


def pareto(sw: pd.DataFrame) -> pd.DataFrame:
    o = sw.sort_values(["mean_latency", "cutoff_rate", "threshold", "action_delay", "timeout"], ascending=[True, True, False, True, True], kind="stable")
    best, keep = math.inf, []
    for idx, c in zip(o.index, o["cutoff_rate"]):
        if c < best - EPS:
            keep.append(idx)
            best = c
    return o.loc[keep]


def auc_ap(pred: pd.DataFrame, score_point: float | None = None) -> dict:
    """Span-level discrimination: score at score_point (or max over the span)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    st = span_table(pred, score_point)
    y = (~st.is_hold).astype(int)
    s = np.array([c[-1] if len(c) else np.nan for c in st.cummax])
    m = np.isfinite(s)
    if len(np.unique(y[m])) < 2:
        return {"auc": math.nan, "ap": math.nan}
    return {"auc": float(roc_auc_score(y[m], s[m])), "ap": float(average_precision_score(y[m], s[m]))}


def apply_knobs(pred: pd.DataFrame, knobs: dict, score_point: float | None = None) -> pd.DataFrame:
    """Per-span outcome at one frozen knob triple (for slices such as filler-preceded holds)."""
    st = span_table(pred, score_point)
    fire = first_crossing(st, knobs["threshold"])
    ft = np.maximum(knobs["action_delay"], fire)
    T = knobs["timeout"]
    cut = (np.isfinite(fire) & (st.dur > ft + EPS)) | (st.dur > T + EPS)
    det = np.isfinite(fire) & (ft <= T + EPS)
    out = st.keys.copy()
    out["cut"] = np.where(st.is_hold, cut, False)
    out["latency"] = np.where(st.is_hold, np.nan, np.where(det, ft, T))
    out["detected"] = np.where(st.is_hold, np.nan, det)
    return out


def summarise(pred: pd.DataFrame, score_point: float | None = None) -> tuple[pd.DataFrame, dict]:
    sw = sweep(pred, score_point)
    ops = operating_points(sw)
    summary = {
        "operating_points": ops,
        "FCR@300": None if ops["fcr_at_300ms"] is None else ops["fcr_at_300ms"]["cutoff_rate"],
        "FCR@600": None if ops["fcr_at_600ms"] is None else ops["fcr_at_600ms"]["cutoff_rate"],
        "Lat@5%": None if ops["lat_at_5pct"] is None else ops["lat_at_5pct"]["mean_latency"],
        "Lat@10%": None if ops["lat_at_10pct"] is None else ops["lat_at_10pct"]["mean_latency"],
        "Detect@5%": None if ops["lat_at_5pct"] is None else ops["lat_at_5pct"]["detect_rate"],
        **auc_ap(pred, score_point),
        "n_hold": int(sw["n_hold"].iloc[0]),
        "n_eot": int(sw["n_eot"].iloc[0]),
    }
    return sw, summary

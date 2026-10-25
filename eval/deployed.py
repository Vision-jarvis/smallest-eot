"""Deployed (VAD-timed) policy evaluation: what an agent would actually do on a live call.

eot-bench times every policy knob on the *oracle* silence spans. A deployed agent has no
oracle: its silence clock is its own VAD. Here the knobs act on the detector's causal trace

    fire at the first hop with  vad_silence >= action_delay and p_eot > threshold,
                          or    vad_silence >= timeout,

after which the policy re-arms only once the VAD hears speech again. Events are then scored
against the labelled spans:

* hold cut-in:   an event inside a hold span (0.2 to 5 s, as eot-bench filters) -> FCR.
* speech cut-in: an event before the true offset that is outside every labelled silence
                 span (VAD dropout mid-speech, or a pause shorter than 100 ms).
* latency:       first event after the true offset (EoT span start) minus that offset. The
                 recording is padded with 3 s of silence so slow timers can still fire; a turn
                 with no event by the end of the pad is charged the full pad length.

With threshold 0 and no timeout this is a pure trailing-silence timer, i.e. the Pulse
`eou_timeout_ms` / Atoms VAD-release behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from eval.data import Turn

EPS = 1e-9


def events(t: np.ndarray, p: np.ndarray, sil: np.ndarray, threshold: float, action_delay: float, timeout: float) -> list[float]:
    out, armed = [], True
    for ti, pi, si in zip(t, p, sil):
        if si >= ti - 1e-6:  # the user has not spoken yet in this turn: nothing to end
            continue
        if si <= EPS:
            armed = True
            continue
        if armed and ((si + EPS >= action_delay and pi > threshold) or si + EPS >= timeout):
            out.append(float(ti))
            armed = False
    return out


def score_turn(turn: Turn, ev: list[float], pad_s: float) -> dict:
    holds = [s for s in turn.spans if s.label == "hold" and 0.2 - EPS <= s.dur <= 5.0 + EPS]
    eot = [s for s in turn.spans if s.label == "eot"]
    cut = [any(s.start < e < s.end - EPS for e in ev) for s in holds]
    last_end = eot[0].start if eot else turn.duration
    in_span = [e >= last_end - EPS or any(s.start - EPS <= e <= s.end + EPS for s in turn.spans) for e in ev]
    lat = None
    if eot:
        s = eot[0]
        after = [e for e in ev if e >= s.start - EPS]
        lat = (after[0] - s.start) if after else (turn.duration + pad_s - s.start)
    premature = int(any(e < last_end - EPS for e in ev))  # any end-of-turn event before the true end
    return {"n_hold": len(holds), "n_cut": int(sum(cut)), "speech_cutins": int(sum(1 for x in in_span if not x)), "latency": lat, "premature": premature}


def evaluate(turns: list[Turn], traces: pd.DataFrame, knobs: dict, pad_s: float = 3.0) -> dict:
    by_id = {r.id: r for r in traces.itertuples(index=False)}
    n_hold = n_cut = cutins = prem = 0
    lats = []
    for turn in turns:
        tr = by_id[turn.id]
        p = np.nan_to_num(np.asarray(tr.p, dtype=float), nan=0.0)
        ev = events(np.asarray(tr.t), p, np.asarray(tr.sil), knobs["threshold"], knobs["action_delay"], knobs.get("timeout", np.inf))
        r = score_turn(turn, ev, pad_s)
        n_hold += r["n_hold"]
        n_cut += r["n_cut"]
        cutins += r["speech_cutins"]
        prem += r["premature"]
        if r["latency"] is not None:
            lats.append(r["latency"])
    lats = np.array(lats)
    return {
        **{k: float(v) for k, v in knobs.items()},
        "fcr": n_cut / max(1, n_hold),
        "mean_latency": float(lats.mean()),
        "p90_latency": float(np.percentile(lats, 90)),
        "speech_cutins_per_turn": cutins / max(1, len(turns)),
        "premature_turn_rate": prem / max(1, len(turns)),
        "n_hold": n_hold,
        "n_eot": int(len(lats)),
    }


def timer_sweep(turns: list[Turn], traces: pd.DataFrame, timers_ms=range(300, 1201, 100)) -> pd.DataFrame:
    """Pure silence-timer curve (threshold 0, no model): the eou_timeout_ms-style baseline."""
    return pd.DataFrame([evaluate(turns, traces, {"threshold": -1.0, "action_delay": ms / 1000, "timeout": np.inf}) for ms in timers_ms])

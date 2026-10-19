"""Deployed (VAD-timed) curves on eot-bench hi from saved traces: the silence-timer sweep
300..1200 ms (the eou_timeout_ms-style production default) and Pulse-EoT-hi at its dev-frozen
knobs.  python -m eval.run_deployed"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from eval.data import load_eot_bench_hi
from eval.deployed import evaluate, timer_sweep

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/eot_bench_hi/clean"


def main() -> None:
    turns = load_eot_bench_hi()
    rows = []
    tr = pd.read_parquet(BASE / "silero_timer/traces.parquet")
    for r in timer_sweep(turns, tr).to_dict("records"):
        rows.append({"system": f"silence timer {int(r['action_delay'] * 1000)} ms", **r})
    s = BASE / "pulse_eot_hi/summary.json"
    if s.exists():
        summ = json.loads(s.read_text())
        traces = pd.read_parquet(BASE / "pulse_eot_hi/traces.parquet")
        rows.append({"system": "Pulse-EoT-hi v0.1, dev-frozen knobs", **evaluate(turns, traces, summ["frozen_knobs"])})
        o = summ["operating_points"]["lat_at_5pct"]
        oracle = {k: o[k] for k in ("threshold", "action_delay", "timeout")}
        rows.append({"system": "Pulse-EoT-hi v0.1, test Lat@5% knobs (oracle, reference only)", **evaluate(turns, traces, oracle)})
    df = pd.DataFrame(rows)[["system", "threshold", "action_delay", "timeout", "fcr", "mean_latency", "p90_latency", "speech_cutins_per_turn"]]
    df.to_csv(BASE / "deployed.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

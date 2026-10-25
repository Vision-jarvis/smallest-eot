"""Real Pulse `eou_timeout_ms` sweep next to the real-VAD silence-timer stand-in, dev-test half.

    python -m eval.pulse_vs_timer

Both rows use the same scorer (eval/deployed.py: a hold is cut if any end-of-turn event lands
inside it; latency = first event at or after the true end minus the true end) on the same
200 dev-test turns with the same 3 s noise-floor pad. The Pulse event is the arrival time (on
the audio clock) of each `is_final: true` transcription; the timer event is Silero VAD silence
reaching the timer value. Writes results/eot_bench_hi_devtest/pulse_vs_timer.{md,csv}.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from eval.data import load_eot_bench_hi
from eval.deployed import timer_sweep
from eval.report import wilson

ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "results/eot_bench_hi_devtest"
PULSE_CSV = DEV / "clean/pulse_endpointing/eou_sweep.csv"


def best_within(df: pd.DataFrame, budget_s: float) -> str:
    ok = df[df.mean_latency <= budget_s + 1e-9]
    if ok.empty:
        return "-"
    r = ok.sort_values(["fcr", "mean_latency"]).iloc[0]
    return f"{100 * r.fcr:.1f}% ({int(r.setting)} ms)"


def main() -> None:
    turns = load_eot_bench_hi(half="devtest")
    tr = pd.read_parquet(DEV / "clean/silero_timer/traces.parquet")
    timer = timer_sweep(turns, tr, timers_ms=range(300, 1201, 100))
    timer["setting"] = (timer.action_delay * 1000).round().astype(int)
    pulse_all = pd.read_csv(PULSE_CSV)
    pulse_all["setting"] = pulse_all.eou_timeout_ms
    pulse = pulse_all[pulse_all.event == "silence"]
    naive = pulse_all[pulse_all.event == "all"]
    rows = []
    for ms in sorted(pulse.setting):
        p = pulse[pulse.setting == ms].iloc[0]
        q = naive[naive.setting == ms].iloc[0]
        t = timer[timer.setting == ms]
        t = t.iloc[0] if len(t) else None
        lo, hi = wilson(int(p.n_cut), int(p.n_hold))
        rows.append({
            "timeout_ms": ms,
            "pulse_fcr": p.fcr, "pulse_fcr_ci": f"[{100 * lo:.1f}%, {100 * hi:.1f}%]",
            "pulse_premature": p.premature_turn_rate,
            "pulse_mean_latency_ms": 1000 * p.mean_latency, "pulse_p90_latency_ms": 1000 * p.p90_latency,
            "pulse_overhead_ms": 1000 * p.mean_latency - ms,
            "allfinals_fcr": q.fcr, "allfinals_premature": q.premature_turn_rate, "allfinals_mean_latency_ms": 1000 * q.mean_latency,
            "timer_fcr": None if t is None else t.fcr,
            "timer_premature": None if t is None else t.premature_turn_rate,
            "timer_mean_latency_ms": None if t is None else 1000 * t.mean_latency,
            "n_hold": int(p.n_hold), "n_eot": int(p.n_eot), "n_turns": int(p.n_turns),
        })
    df = pd.DataFrame(rows)
    df.to_csv(DEV / "pulse_vs_timer.csv", index=False)
    f = lambda x: "-" if x is None or pd.isna(x) else f"{100 * x:.1f}%"
    g = lambda x: "-" if x is None or pd.isna(x) else f"{x:.0f} ms"
    md = ["## Pulse `eou_timeout_ms` sweep (REAL, live API 2026-10-24) vs the real-VAD timer, eot-bench `hi` dev-test half", "",
          "`python -m adapters.pulse_endpointing` streamed each of the 200 dev-test turns (plus a 3 s noise-floor pad) in real time "
          "(4096-byte chunks = 128 ms) to `wss://api.smallest.ai/waves/v1/stt/live` with `model=pulse, language=hi, endpointing=true, "
          "eou_timeout_ms=<x>` (other parameters at their defaults, so `finalize_on_words` is on) and logged every message's arrival time "
          "on the audio clock. Two readings of the Pulse stream: **silence final** = the first `is_final` after a `speech_ended` VAD event "
          "(the eou_timeout_ms window expired); **any final** = every `is_final`, which also includes word-count finals Pulse emits in the "
          "middle of speech. The timer row is our Silero VAD silence timer on the same turns. Same scorer for all (eval/deployed.py). "
          "FCR = share of holds (pauses of 0.2 to 5 s inside the turn) with an end-of-turn event inside the hold. Because a late event "
          "can land after the user has resumed speaking (a cut-in that FCR does not count), the table also gives **premature turns** = "
          "share of turns with any end-of-turn event before the true end. Latency = true end of speech to the first event at or after it. "
          "Brackets: 95% Wilson interval.", "",
          "| eou_timeout_ms | Pulse silence final: FCR | premature turns | mean latency | p90 latency | latency minus timeout | Pulse any final: FCR / premature / latency | Real-VAD timer: FCR / premature / latency |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['timeout_ms']} | {f(r['pulse_fcr'])} {r['pulse_fcr_ci']} | {f(r['pulse_premature'])} | {g(r['pulse_mean_latency_ms'])} | "
                  f"{g(r['pulse_p90_latency_ms'])} | {r['pulse_overhead_ms']:+.0f} ms | "
                  f"{f(r['allfinals_fcr'])} / {f(r['allfinals_premature'])} / {g(r['allfinals_mean_latency_ms'])} | "
                  f"{f(r['timer_fcr'])} / {f(r['timer_premature'])} / {g(r['timer_mean_latency_ms'])} |")
    md += ["", f"n = {rows[0]['n_turns']} turns, {rows[0]['n_hold']} holds, {rows[0]['n_eot']} turn ends.", "",
           "FCR at a mean-latency budget (best setting whose mean latency fits; the deployed-policy analogue of eot-bench FCR@x):", "",
           "| Budget | Pulse silence final | Pulse any final | Real-VAD timer |", "|---|---|---|---|"]
    for b in (0.3, 0.6, 0.8, 1.0):
        md.append(f"| {int(b * 1000)} ms | {best_within(pulse, b)} | {best_within(naive, b)} | {best_within(timer, b)} |")
    v = DEV / "clean/pulse_eot_hi__asr_pulse/summary.json"
    if v.exists():
        s = json.loads(v.read_text())
        base = json.loads((DEV / "clean/pulse_eot_hi/summary.json").read_text())
        md += ["", "Pulse-EoT-hi v0.1 with real Pulse partials (endpointing=false) instead of Vosk, same 200 turns, eot-bench knob sweep "
               "(`python -m eval.run_bench --detector pulse_eot_hi --asr pulse`; same weights, only the text source changes):", "",
               "| Text source | FCR@300 | FCR@600 | Lat@5% | Lat@10% | AUC |", "|---|---|---|---|---|---|"]
        for name, x in (("Vosk small-hi replay", base), ("Pulse partials (REAL)", s)):
            f3 = "-" if x.get("FCR@300") is None else f"{100 * x['FCR@300']:.1f}%"
            md.append(f"| {name} | {f3} | {100 * x['FCR@600']:.1f}% | {1000 * x['Lat@5%']:.0f} ms | {1000 * x['Lat@10%']:.0f} ms | {x['auc']:.3f} |")
        from eval import metrics
        from eval.rescore_devtest import paired_bootstrap

        pa = pd.read_parquet(DEV / "clean/pulse_eot_hi__asr_pulse/predictions.parquet")
        pb = pd.read_parquet(DEV / "clean/pulse_eot_hi/predictions.parquet")
        kn = lambda pr: {k: metrics.summarise(pr, None)[1]["operating_points"]["lat_at_5pct"][k] for k in ("threshold", "action_delay", "timeout")}
        bt = paired_bootstrap(pa, pb, kn(pa), kn(pb), None, None)
        md += ["", f"Paired bootstrap over {bt['n_turns']} turns ({bt['resamples']} resamples), Pulse partials minus Vosk, each at its own Lat@5% knobs: "
               f"latency {1000 * bt['lat_diff_s']:+.1f} ms (95% CI {1000 * bt['lat_ci_s'][0]:+.1f} to {1000 * bt['lat_ci_s'][1]:+.1f}), "
               f"FCR {100 * bt['fcr_diff']:+.2f} pp (95% CI {100 * bt['fcr_ci'][0]:+.2f} to {100 * bt['fcr_ci'][1]:+.2f})."]
        if "frozen_result" in s:
            fr, fb = s["frozen_result"], base.get("frozen_result", {})
            md += ["", f"At the dev-frozen knobs: Vosk FCR {100 * fb.get('fcr', float('nan')):.1f}% at {1000 * fb.get('mean_latency', float('nan')):.0f} ms, "
                   f"Pulse partials FCR {100 * fr['fcr']:.1f}% at {1000 * fr['mean_latency']:.0f} ms."]
    (DEV / "pulse_vs_timer.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

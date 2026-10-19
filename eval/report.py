"""Collect every results/*/summary.json into results/report.md, Pareto plots and points matrix.

    python -m eval.report

Only numbers present in results/ are written; nothing is typed in by hand except LiveKit's
published Hindi rows, which are labelled as published and read from the eot-bench repo.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.text import token_class, normalise  # noqa: E402
from eval import metrics  # noqa: E402
from eval.data import load_eot_bench_hi, word_before  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PUB = ROOT / "third_party/eot-bench/output/livekit__eot-bench-data__validation__min_silence_100ms/hi"
PUBLISHED = {  # run dir -> display name (numbers are recomputed from LiveKit's committed predictions)
    "live_kit_turn_detector_adapter__turn_detector_v1": "LiveKit Turn Detector v1 (published)",
    "live_kit_turn_detector_mini_adapter__7cd280604d": "LiveKit v1-mini (published)",
    "deepgram_flux_streaming_adapter__flux_general_multi": "Deepgram Flux (published)",
    "smart_turn_audio_adapter__cb4f57229a": "SmartTurn v3.2 (published, GPU fp32)",
    "vap_adapter__d8cef83e57": "VAP (published)",
}
NAMES = {
    "silero_timer": "Silero VAD timer (ours, real VAD)",
    "smart_turn_v32": "SmartTurn v3.2 int8 CPU (ours)",
    "smart_turn_v32_fp32": "SmartTurn v3.2 fp32 on CPU (ours)",
    "hinglish_turn": "deveshu/hinglish-turn-detector (ours)",
    "pulse_eot_hi": "Pulse-EoT-hi v0.1 stacked (ours)",
    "pulse_eot_hi__full": "Pulse-EoT-hi v0, Hindi TTS only (ours)",
}


def name_of(d: str) -> str:
    if d in NAMES:
        return NAMES[d]
    if d.startswith("pulse_eot_hi__"):
        v = d.split("__", 1)[1]
        return f"Pulse-EoT-hi {'v0.1' if v.startswith(('stacked', 'smart_turn')) else 'v0'} [{v}]"
    return d


def pct(x) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def ms(x) -> str:
    return "-" if x is None else f"{1000 * x:.0f} ms"


def published_rows() -> list[dict]:
    out = []
    for d, nm in PUBLISHED.items():
        p = PUB / d
        if not p.exists():
            continue
        sp = json.loads((p / "manifest.json").read_text()).get("score_point")
        pred = pd.read_parquet(p / "predictions.parquet")
        sw, s = metrics.summarise(pred, sp)
        out.append({"system": nm, "kind": "published", "sweep": sw, **s})
    vb = metrics.vad_baseline(pd.read_parquet(PUB / "smart_turn_audio_adapter__cb4f57229a/predictions.parquet"))
    ops = metrics.operating_points(vb)
    out.append({"system": "VAD baseline, oracle spans (eot-bench)", "kind": "published", "sweep": vb, "FCR@300": ops["fcr_at_300ms"]["cutoff_rate"], "FCR@600": ops["fcr_at_600ms"]["cutoff_rate"], "Lat@5%": ops["lat_at_5pct"]["mean_latency"], "Lat@10%": ops["lat_at_10pct"]["mean_latency"], "Detect@5%": 1.0, "auc": None})
    return out


def our_rows(set_name: str, condition: str) -> list[dict]:
    base = RES / set_name / condition
    out = []
    for d in sorted(base.glob("*/summary.json")) if base.exists() else []:
        s = json.loads(d.read_text())
        sw = pd.read_parquet(d.parent / "sweep.parquet")
        out.append({"system": name_of(d.parent.name), "dir": d.parent.name, "kind": "ours", "sweep": sw, **s})
    return out


def table(rows: list[dict]) -> str:
    lines = ["| System | FCR@300 | FCR@600 | Lat@5% | Lat@10% | Detect@5% | AUC |", "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r.get("Lat@5%") or 9)):
        auc = r.get("auc")
        lines.append(f"| {r['system']} | {pct(r.get('FCR@300'))} | {pct(r.get('FCR@600'))} | {ms(r.get('Lat@5%'))} | {ms(r.get('Lat@10%'))} | {pct(r.get('Detect@5%'))} | {'-' if auc is None else f'{auc:.3f}'} |")
    return "\n".join(lines)


def pareto_plot(rows: list[dict], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for r in rows:
        sw = r["sweep"]
        fr = metrics.pareto(sw) if "threshold" in sw and sw["threshold"].notna().any() else sw.sort_values("mean_latency")
        fr = fr[fr.cutoff_rate <= 0.5]
        ls = "--" if r["kind"] == "published" else "-"
        lw = 2.6 if r.get("dir") == "pulse_eot_hi" else 1.3
        ax.step(fr.mean_latency * 1000, fr.cutoff_rate * 100, where="post", ls=ls, lw=lw, label=r["system"])
    ax.axvline(300, color="grey", lw=0.6)
    ax.axvline(600, color="grey", lw=0.6)
    ax.axhline(5, color="grey", lw=0.6)
    ax.set_xlabel("mean latency after true end of turn (ms)")
    ax.set_ylabel("false cut-off rate on holds (%)")
    ax.set_title(title)
    ax.set_xlim(150, 1300)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def slice_table(dirs: list[str]) -> str:
    """FCR on filler/connective-preceded and digit-preceded holds at each system's Lat@5% knobs.

    Slices use the reference word before the pause (labels only, never model input).
    """
    turns = {t.id: t for t in load_eot_bench_hi()}
    keys = []
    for t in turns.values():
        for s in t.spans:
            if s.label != "hold":
                continue
            w = word_before(t, s.start)
            toks = normalise(w or "")
            c = token_class(toks[-1]) if toks else "NONE"
            keys.append({"id": t.id, "span_index": s.index, "slice": {"HOLD": "filler/connective", "DIGIT": "digit/number"}.get(c, "other")})
    sl = pd.DataFrame(keys)
    lines = ["| System | knobs (th, delay, timeout) | all holds | filler/connective holds (FFCR) | digit holds |", "|---|---|---|---|---|"]
    for d in dirs:
        p = RES / "eot_bench_hi/clean" / d
        if not (p / "summary.json").exists():
            continue
        s = json.loads((p / "summary.json").read_text())
        frozen = s.get("frozen_knobs") or s["operating_points"]["lat_at_5pct"]
        src = "dev-frozen" if s.get("frozen_knobs") else "test Lat@5%"
        pred = pd.read_parquet(p / "predictions.parquet")
        o = metrics.apply_knobs(pred, frozen, s.get("score_point")).merge(sl, on=["id", "span_index"])
        g = o.groupby("slice")["cut"].agg(["mean", "size"])
        cell = lambda k: f"{pct(g.loc[k, 'mean'])} (n={g.loc[k, 'size']})" if k in g.index else "-"
        lines.append(f"| {name_of(d)} | {frozen['threshold']:.2f}, {frozen['action_delay']:.1f}, {frozen['timeout']:.1f} ({src}) | {pct(o['cut'].mean())} | {cell('filler/connective')} | {cell('digit/number')} |")
    return "\n".join(lines)


def failure_gallery(d: str = "pulse_eot_hi", n: int = 20) -> str:
    """The n worst decisions of a system at its dev-frozen knobs: cut-off holds (longest first)
    and slowest EoTs, with turn id, span times, the reference words around the pause and the
    streaming ASR text the model had seen. Audio: data/hf/livekit__eot-bench-data (row id)."""
    from eval import asr_cache
    from core.asr import visible_text

    p = RES / "eot_bench_hi/clean" / d
    s = json.loads((p / "summary.json").read_text())
    knobs = s.get("frozen_knobs") or s["operating_points"]["lat_at_5pct"]
    o = metrics.apply_knobs(pd.read_parquet(p / "predictions.parquet"), knobs)
    turns = {t.id: t for t in load_eot_bench_hi()}
    parts = asr_cache.load("eot_bench_hi_clean")
    cut = o[(o.label == "hold") & o.cut].sort_values("span_dur", ascending=False).head(n // 2)
    late = o[o.label == "eot"].sort_values("latency", ascending=False).head(n - len(cut))
    lines = [f"Knobs: threshold {knobs['threshold']:.2f}, action_delay {knobs['action_delay']:.1f} s, timeout {knobs['timeout']:.1f} s (dev-frozen).", "",
             "| # | turn id | kind | pause (s) | words before pause (reference) | ASR text visible at pause end+0.3 s | latency |", "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(pd.concat([cut, late]).itertuples(index=False), 1):
        t = turns[r.id]
        sp = next(x for x in t.spans if x.index == r.span_index)
        before = " ".join(str(w["word"]).strip() for w in t.words if float(w["end"]) <= sp.start + 0.05)[-60:]
        seen = visible_text(parts.get(r.id, []), sp.start + 0.3)[0][-60:]
        kind = "false cut-off" if r.label == "hold" else "slow EoT"
        lat = "-" if r.label == "hold" else f"{1000 * r.latency:.0f} ms"
        lines.append(f"| {i} | {r.id} | {kind} | {sp.start:.1f} to {sp.end:.2f} | {before} | {seen} | {lat} |")
    return "\n".join(lines)


def points(row: dict | None) -> dict:
    if row is None:
        return {}
    lat, fcr = row.get("Lat@5%"), row.get("FCR@300")
    p1 = 0.0 if lat is None else float(np.clip((0.700 - lat) / (0.700 - 0.450), 0, 1) * 20)
    p2 = 0.0 if fcr is None else float(np.clip((0.495 - fcr) / (0.495 - 0.10), 0, 1) * 10)
    return {"row1_lat5_eotbench_hi (20)": round(p1, 2), "row2_fcr300_eotbench_hi (10)": round(p2, 2)}


def main() -> None:
    pub = published_rows()
    clean = our_rows("eot_bench_hi", "clean")
    RES.mkdir(exist_ok=True)
    main_rows = [r for r in clean if "__" not in r["dir"] or r["dir"] == "pulse_eot_hi__full"]
    pareto_plot(pub + main_rows, RES / "pareto_eot_bench_hi.png", "eot-bench hi (400 turns), knob-grid Pareto frontiers")
    md = ["# Results (generated by `python -m eval.report`)", "", "## eot-bench `hi` (400 turns, 16 kHz), eot-bench knob sweep", "",
          "Rows marked *published* are recomputed by our metrics code from LiveKit's committed prediction files (identical to their report). Rows marked *ours* were run in this repo through the streaming harness on CPU.", "", table(pub + clean), "", "![pareto](pareto_eot_bench_hi.png)", ""]
    for cond in ("mulaw8k", "snr10", "snr5"):
        rows = our_rows("eot_bench_hi", cond)
        if rows:
            md += [f"## eot-bench `hi`, channel condition `{cond}` (evaluation-only stress slice)", "", table(rows), ""]
    dev = our_rows("dev_hi", "clean")
    if dev:
        md += ["## Dev split (Smart Turn v3.2 *test* Hindi clips as single-pause episodes; used for model selection and knob freezing)", "", table(dev), ""]
    md += ["## Hold slices on eot-bench `hi`", "", slice_table([r["dir"] for r in clean]), ""]
    dep = RES / "eot_bench_hi/clean/deployed.csv"
    if dep.exists():
        md += ["## Deployed (VAD-timed) policy on eot-bench `hi`", "", pd.read_csv(dep).to_markdown(index=False, floatfmt=".3f"), ""]
    arb = RES / "arbiter/summary.json"
    if arb.exists():
        a = json.loads(arb.read_text())["test"]
        md += ["## Barge-in arbiter (proxy overlap set, see arbiter/data.py)", "", f"FBR {pct(a['FBR'])}, MBR {pct(a['MBR'])}, arbiter score {a['arbiter_score']:.3f}, reaction p50/p90 {a['reaction_p50_ms']}/{a['reaction_p90_ms']} ms, n={a['n']}", "", "Confusion (rows true, columns predicted):", "", pd.DataFrame(a["confusion"]).fillna(0).astype(int).to_markdown(), ""]
    for f in sorted((RES / "efficiency").glob("*.json")) if (RES / "efficiency").exists() else []:
        e = json.loads(f.read_text())
        md += [f"- Efficiency `{e['detector']}`: p50/p95/p99 {e['ms_per_hop_p50']:.2f}/{e['ms_per_hop_p95']:.2f}/{e['ms_per_hop_p99']:.2f} ms per hop (1 thread), {e['calls_per_core']:.0f} calls per core, model {e['model_mb']:.1f} MB, {e['rss_mb_for_calls']:.0f} MB RSS for {e['concurrent_calls_held']} call states"]
    if (RES / "eot_bench_hi/clean/pulse_eot_hi/summary.json").exists():
        header = "# 20 failure cases, Pulse-EoT-hi v0.1 on eot-bench hi\n\n"
        (RES / "failures.md").write_text(header + failure_gallery() + "\n", encoding="utf-8")
        md += ["", "Failure gallery: [failures.md](failures.md)"]
    ours = next((r for r in clean if r["dir"] == "pulse_eot_hi"), None)
    md += ["", "## Points matrix (only rows measurable without Bharat-Turn-8k)", "", "```", json.dumps(points(ours), indent=1), "```", ""]
    (RES / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()

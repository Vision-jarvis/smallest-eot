"""Pulse `model=pulse` endpointing baseline: Smallest's actual Hindi production default (run 2026-10-24).

The statement asks for Pulse's own `eou_timeout_ms` sweep {300, 400, 600, 800, 1000, 1200}
on eot-bench `hi`. This module:

1. `stream_turn()` streams one turn to the Pulse socket in real time (20 ms PCM16 frames
   grouped into the documented 4096-byte chunks, paced against the wall clock), then sends
   {"type":"close_stream"}, and logs every server message with its arrival time on the audio
   clock (seconds since the first frame was sent).
2. `run_sweep()` does that for every turn and timeout value, writing
   data/cache/pulse/eou_<ms>/<turn_id>.jsonl (the "logged arrival times" honesty rule 2 asks for).
3. `score_sweep()` turns each log into end-of-turn events, two ways (see `final_times`):
   "silence" = the final Pulse emits when the eou_timeout_ms window expires after
   `speech_ended`; "all" = every `is_final: true` (also the mid-speech word-count finals),
   and scores them with eval/deployed.py, the same scorer as the real-VAD timer.
4. `run_partials()` streams with endpointing=false and caches every partial/final
   (data/cache/pulse/partials/<turn_id>.jsonl): the Pulse text our own detector would consume.

Cost first (no key, no network): prints audio minutes, list-price dollars and wall time:
    python -m adapters.pulse_endpointing --estimate-cost
Run (once SMALLEST_API_KEY is set):
    python -m adapters.pulse_endpointing --sweep 300 400 600 800 1000 1200
Partials cache (endpointing=false, no scoring; model input for the v1 refit, all 400 turns):
    python -m adapters.pulse_endpointing --partials --half all
    python -m adapters.pulse_endpointing --build-asr-cache   # -> eval.run_bench --asr pulse

Scope: the 200-turn dev-test half by default (eval/split.py). Scoring the sealed half needs
--final and is logged in results/SEALED_LEDGER.md. The --partials mode only collects Pulse
transcripts (model inputs, no labels read, nothing scored), so it may cover all 400 turns.
Price: Pulse streaming $0.006/min on the Standard plan (docs.smallest.ai changelog,
28 May 2026; the 2026-10-20 model card said $0.004/min). Wire shapes seen on the real API:
docs/pulse_api_notes.md.
Smoke test against the local mock (no key, not Pulse numbers):
    python -m sidecar.mock_pulse --port 8765 &   python -m adapters.pulse_endpointing --base ws://127.0.0.1:8765 --limit 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from core.audio import pad_silence, to_pcm16
from eval.data import Turn, load_eot_bench_hi
from eval.deployed import score_turn
from eval.split import guard
from sidecar.pulse_protocol import CHUNK_BYTES, PULSE_WS, auth_headers, pulse_url

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "data/cache/pulse"
PARTIALS = ROOT / "data/cache/pulse/partials"
SWEEP_MS = (300, 400, 600, 800, 1000, 1200)
PAD_S = 3.0
USD_PER_MIN_REALTIME = 0.006  # Standard plan streaming, docs changelog 2026-05-28 (was 0.004)


def estimate_cost(turns: list[Turn], passes: int, concurrency: int = 4) -> dict:
    """Audio streamed = every turn plus its 3 s silence pad, once per pass (timeout value)."""
    minutes = sum(t.duration + PAD_S for t in turns) / 60 * passes
    return {"turns": len(turns), "passes": passes, "audio_minutes": round(minutes, 1),
            "usd_at_list_price": round(minutes * USD_PER_MIN_REALTIME, 2),
            "wall_hours_at_concurrency": round(minutes / 60 / concurrency, 2), "concurrency": concurrency,
            "price_source": "docs.smallest.ai changelog 2026-05-28: streaming $0.006/min (Standard plan)"}


async def stream_turn(x: np.ndarray, sr: int, url: str, headers: dict, realtime: bool = True) -> list[dict]:
    import websockets

    pcm = to_pcm16(x).tobytes()
    log: list[dict] = []
    async with websockets.connect(url, additional_headers=headers, max_size=None) as ws:
        t0 = time.perf_counter()

        async def reader():
            async for m in ws:
                log.append({"t_arrival": time.perf_counter() - t0, "msg": json.loads(m)})

        rd = asyncio.create_task(reader())
        for i in range(0, len(pcm), CHUNK_BYTES):
            chunk = pcm[i : i + CHUNK_BYTES]
            if realtime:  # never send audio before it would exist on a live call
                due = (i + len(chunk)) / 2 / sr
                await asyncio.sleep(max(0.0, due - (time.perf_counter() - t0)))
            await ws.send(chunk)
        await ws.send(json.dumps({"type": "close_stream"}))
        try:
            await asyncio.wait_for(rd, timeout=15)
        except asyncio.TimeoutError:
            rd.cancel()
    return log


def final_times(log: list[dict], mode: str = "all") -> list[float]:
    """End-of-turn events from one Pulse log.

    mode="all": every `is_final: true` transcription except the close-of-stream `is_last` one
    (what an agent that treats each final as end-of-turn would do).
    mode="silence": only the first final after a `speech_ended` VAD event with no
    `speech_started` in between, i.e. the finals Pulse emits because the eou_timeout_ms silence
    window expired. With the default `finalize_on_words=true`, Pulse also emits word-count
    finals in the middle of speech (docs: features/endpointing); those are dropped here.
    """
    out, armed = [], False
    for e in log:
        m = e["msg"]
        t = m.get("type")
        if t == "speech_ended":
            armed = True
        elif t == "speech_started":
            armed = False
        elif t == "transcription" and m.get("is_final") and not m.get("is_last"):
            if mode == "all" or armed:
                out.append(e["t_arrival"])
            armed = False
    return out


def partials_from_log(log: list[dict]) -> list[tuple[float, str, bool]]:
    """Pulse messages -> the (t_arrival, text, is_final) partial records our detectors consume."""
    out = []
    for e in log:
        m = e["msg"]
        if m.get("type") == "transcription" and m.get("transcript"):
            out.append((e["t_arrival"], m["transcript"], bool(m.get("is_final"))))
    return out


class PulseError(RuntimeError):
    pass


def _collect(jobs: list[tuple[Turn, Path, str]], key: str, concurrency: int) -> None:
    """Stream every (turn, out_path, url) job not already on disk. One retry per job; an
    error message from the server (auth, credits, rate limit) stops the whole run."""

    async def go():
        sem = asyncio.Semaphore(concurrency)
        done = [0]

        async def one(turn: Turn, out: Path, url: str):
            if out.exists():
                return
            async with sem:
                x = pad_silence(turn.audio, turn.sr, PAD_S, np.random.default_rng(0))
                log = None
                for attempt in range(2):
                    try:
                        log = await stream_turn(x, turn.sr, url, auth_headers(key))
                        break
                    except Exception as e:  # network hiccup: retry once
                        if attempt == 1:
                            print(f"FAILED {turn.id}: {type(e).__name__}", flush=True)
                            return
                errs = [e["msg"] for e in log if e["msg"].get("type") == "error"]
                if errs:
                    raise PulseError(f"{turn.id}: {json.dumps(errs[0])[:300]}")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in log), encoding="utf-8")
            done[0] += 1
            if done[0] % 25 == 0:
                print(f"  {done[0]} streams written", flush=True)

        await asyncio.gather(*(one(*j) for j in jobs))

    asyncio.run(go())


def run_sweep(turns: list[Turn], timeouts, base: str, key: str, concurrency: int = 4) -> None:
    jobs = [(t, LOGS / f"eou_{ms}" / f"{t.id}.jsonl", pulse_url(sample_rate=t.sr, endpointing=True, eou_timeout_ms=ms, base=base))
            for ms in timeouts for t in turns]
    _collect(jobs, key, concurrency)


def run_partials(turns: list[Turn], base: str, key: str, concurrency: int = 4) -> None:
    """endpointing=false: Pulse transcribes, our detector decides. Logs every partial/final."""
    jobs = [(t, PARTIALS / f"{t.id}.jsonl", pulse_url(sample_rate=t.sr, endpointing=False, base=base)) for t in turns]
    _collect(jobs, key, concurrency)


def streamed_minutes(turns: list[Turn], dirs: list[Path]) -> float:
    """Audio minutes actually sent = (turn + pad) for every log file present (spend accounting)."""
    return sum((t.duration + PAD_S) for d in dirs for t in turns if (d / f"{t.id}.jsonl").exists()) / 60


def build_asr_cache(turns: list[Turn]) -> Path:
    """Partials logs -> data/cache/asr_eot_bench_hi_pulse.parquet, the same (turn_id, t_arrival,
    text, is_final) schema as the Vosk cache, so `run_bench --asr pulse` can replay it."""
    rows = []
    for t in turns:
        f = PARTIALS / f"{t.id}.jsonl"
        if not f.exists():
            raise SystemExit(f"no Pulse partials for {t.id}; run --partials --half all first")
        log = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows += [{"turn_id": t.id, "t_arrival": ta, "text": tx.strip(), "is_final": fi} for ta, tx, fi in partials_from_log(log)]
    out = ROOT / "data/cache/asr_eot_bench_hi_pulse.parquet"
    pd.DataFrame(rows, columns=["turn_id", "t_arrival", "text", "is_final"]).to_parquet(out)
    return out


def score_sweep(turns: list[Turn], timeouts, modes=("silence", "all")) -> pd.DataFrame:
    rows = []
    for mode in modes:
        for ms in timeouts:
            n_hold = n_cut = cutins = n_ev = prem = 0
            lats = []
            for turn in turns:
                f = LOGS / f"eou_{ms}" / f"{turn.id}.jsonl"
                log = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
                ev = final_times(log, mode)
                r = score_turn(turn, ev, PAD_S)
                n_hold, n_cut, cutins, n_ev = n_hold + r["n_hold"], n_cut + r["n_cut"], cutins + r["speech_cutins"], n_ev + len(ev)
                prem += r["premature"]
                if r["latency"] is not None:
                    lats.append(r["latency"])
            rows.append({"event": mode, "eou_timeout_ms": ms, "fcr": n_cut / max(1, n_hold), "mean_latency": float(np.mean(lats)),
                         "p90_latency": float(np.percentile(lats, 90)), "speech_cutins_per_turn": cutins / len(turns),
                         "premature_turn_rate": prem / len(turns), "events_per_turn": n_ev / len(turns), "n_hold": n_hold, "n_cut": n_cut, "n_eot": len(lats), "n_turns": len(turns)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=int, nargs="+", default=list(SWEEP_MS))
    ap.add_argument("--base", default=PULSE_WS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--half", default="devtest", choices=("devtest", "sealed", "all"))
    ap.add_argument("--final", action="store_true", help="allow scoring the sealed half (logged)")
    ap.add_argument("--partials", action="store_true", help="collect Pulse partials with endpointing=false (no scoring)")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--estimate-cost", action="store_true", help="print the cost and exit (no key needed)")
    ap.add_argument("--score-only", action="store_true", help="re-score logs already on disk (no network, no key)")
    ap.add_argument("--build-asr-cache", action="store_true", help="partials logs -> data/cache/asr_eot_bench_hi_pulse.parquet (no network)")
    a = ap.parse_args()
    if a.build_asr_cache:
        print(build_asr_cache(load_eot_bench_hi(a.limit, half="all")))
        return
    passes = 1 if a.partials else len(a.sweep)
    if a.estimate_cost:
        print(json.dumps(estimate_cost(load_eot_bench_hi(a.limit, half=a.half), passes, a.concurrency), indent=1))
        return
    key = os.environ.get("SMALLEST_API_KEY", "")
    if a.base == PULSE_WS and not key and not a.score_only:
        raise SystemExit("set SMALLEST_API_KEY to run against Pulse (see STATUS.md).")
    if not a.partials:
        guard(a.half, a.final, "Pulse eou_timeout_ms sweep")
    turns = load_eot_bench_hi(a.limit, half=a.half)
    global LOGS, PARTIALS
    if a.base != PULSE_WS:
        LOGS = ROOT / "data/cache/pulse_mock"
        PARTIALS = LOGS / "partials"
    if a.base == PULSE_WS and not a.score_only:
        print("cost:", json.dumps(estimate_cost(turns, passes, a.concurrency)), flush=True)
    if a.partials:
        run_partials(turns, a.base, key or "mock", a.concurrency)
        n = sum((PARTIALS / f"{t.id}.jsonl").exists() for t in turns)
        print(f"partials cached for {n}/{len(turns)} turns in {PARTIALS}; {streamed_minutes(turns, [PARTIALS]):.1f} audio-minutes on disk")
        return
    if not a.score_only:
        run_sweep(turns, a.sweep, a.base, key or "mock", a.concurrency)
    df = score_sweep(turns, a.sweep)
    mins = streamed_minutes(turns, [LOGS / f"eou_{ms}" for ms in a.sweep])
    # mock (protocol test) output never goes under results/
    sub = "eot_bench_hi_devtest" if a.half == "devtest" else "eot_bench_hi"
    out = ROOT / f"results/{sub}/clean/pulse_endpointing" if a.base == PULSE_WS else ROOT / "data/cache/pulse_mock/result"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "eou_sweep.csv", index=False)
    print(df.to_string(index=False))
    print(f"audio-minutes on disk for these timeouts: {mins:.1f} (USD {mins * USD_PER_MIN_REALTIME:.2f} at list price)")


if __name__ == "__main__":
    main()

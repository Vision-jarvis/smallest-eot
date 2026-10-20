"""Pulse `model=pulse` endpointing baseline: Smallest's actual Hindi production default.  PENDING (API key).

The statement asks for Pulse's own `eou_timeout_ms` sweep {300, 400, 600, 800, 1000, 1200}
on eot-bench `hi`. This module does all of it except holding a key:

1. `stream_turn()` streams one turn to the Pulse socket in real time (20 ms PCM16 frames
   grouped into the documented 4096-byte chunks, paced against the wall clock), then sends
   {"type":"close_stream"}, and logs every server message with its arrival time on the audio
   clock (seconds since the first frame was sent).
2. `run_sweep()` does that for every turn and timeout value, writing
   data/cache/pulse/eou_<ms>/<turn_id>.jsonl (the "logged arrival times" honesty rule 2 asks for).
3. `score_sweep()` turns each log into events (arrival of every `is_final: true`
   transcription = the agent's end-of-turn trigger) and scores them with eval/deployed.py,
   plus an eot-bench-mode trace (p = 1 once a final has arrived in the current pause, the
   binary-score convention eot-bench uses for Soniox/Cartesia endpoint events).

Cost first (no key, no network): prints audio minutes, list-price dollars and wall time:
    python -m adapters.pulse_endpointing --estimate-cost
Run (once SMALLEST_API_KEY is set):
    python -m adapters.pulse_endpointing --sweep 300 400 600 800 1000 1200

Scope: all 400 eot-bench hi turns, like the real-VAD timer sweep it is compared with. It is a
baseline with no knob of ours in it, so it does not go through the sealed-half guard; nothing
in our model is tuned on its output. Price: Pulse realtime $0.004/min, batch $0.003/min
(docs.smallest.ai/model-cards/speech-to-text/pulse.md, Standard plan, fetched 2026-10-20).
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
from sidecar.pulse_protocol import CHUNK_BYTES, PULSE_WS, auth_headers, pulse_url

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "data/cache/pulse"
SWEEP_MS = (300, 400, 600, 800, 1000, 1200)
PAD_S = 3.0
USD_PER_MIN_REALTIME = 0.004


def estimate_cost(turns: list[Turn], timeouts, concurrency: int = 4) -> dict:
    """Audio streamed = every turn plus its 3 s silence pad, once per timeout value."""
    minutes = sum(t.duration + PAD_S for t in turns) / 60 * len(timeouts)
    return {"turns": len(turns), "timeouts": list(timeouts), "audio_minutes": round(minutes, 1),
            "usd_at_list_price": round(minutes * USD_PER_MIN_REALTIME, 2),
            "wall_hours_at_concurrency": round(minutes / 60 / concurrency, 2), "concurrency": concurrency,
            "price_source": "docs.smallest.ai/model-cards/speech-to-text/pulse.md: realtime $0.004/min (2026-10-20)"}


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


def final_times(log: list[dict]) -> list[float]:
    return [e["t_arrival"] for e in log if e["msg"].get("type") == "transcription" and e["msg"].get("is_final") and not e["msg"].get("is_last")]


def partials_from_log(log: list[dict]) -> list[tuple[float, str, bool]]:
    """Pulse messages -> the (t_arrival, text, is_final) partial records our detectors consume."""
    out = []
    for e in log:
        m = e["msg"]
        if m.get("type") == "transcription" and m.get("transcript"):
            out.append((e["t_arrival"], m["transcript"], bool(m.get("is_final"))))
    return out


def run_sweep(turns: list[Turn], timeouts, base: str, key: str, concurrency: int = 4) -> None:
    async def go():
        sem = asyncio.Semaphore(concurrency)

        async def one(turn: Turn, ms: int):
            out = LOGS / f"eou_{ms}" / f"{turn.id}.jsonl"
            if out.exists():
                return
            async with sem:
                x = pad_silence(turn.audio, turn.sr, PAD_S, np.random.default_rng(0))
                url = pulse_url(sample_rate=turn.sr, endpointing=True, eou_timeout_ms=ms, base=base)
                log = await stream_turn(x, turn.sr, url, auth_headers(key))
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in log), encoding="utf-8")

        await asyncio.gather(*(one(t, ms) for ms in timeouts for t in turns))

    asyncio.run(go())


def score_sweep(turns: list[Turn], timeouts) -> pd.DataFrame:
    rows = []
    for ms in timeouts:
        n_hold = n_cut = cutins = 0
        lats = []
        for turn in turns:
            f = LOGS / f"eou_{ms}" / f"{turn.id}.jsonl"
            log = [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
            r = score_turn(turn, final_times(log), PAD_S)
            n_hold, n_cut, cutins = n_hold + r["n_hold"], n_cut + r["n_cut"], cutins + r["speech_cutins"]
            if r["latency"] is not None:
                lats.append(r["latency"])
        rows.append({"eou_timeout_ms": ms, "fcr": n_cut / max(1, n_hold), "mean_latency": float(np.mean(lats)), "speech_cutins_per_turn": cutins / len(turns)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", type=int, nargs="+", default=list(SWEEP_MS))
    ap.add_argument("--base", default=PULSE_WS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--estimate-cost", action="store_true", help="print the cost of the sweep and exit (no key needed)")
    a = ap.parse_args()
    if a.estimate_cost:
        print(json.dumps(estimate_cost(load_eot_bench_hi(a.limit), a.sweep, a.concurrency), indent=1))
        return
    key = os.environ.get("SMALLEST_API_KEY", "")
    if a.base == PULSE_WS and not key:
        raise SystemExit("PENDING: set SMALLEST_API_KEY to run the Pulse eou_timeout_ms sweep (see STATUS.md).")
    turns = load_eot_bench_hi(a.limit)
    if a.base != PULSE_WS:
        global LOGS
        LOGS = ROOT / "data/cache/pulse_mock"
    if a.base == PULSE_WS:
        print("cost:", json.dumps(estimate_cost(turns, a.sweep, a.concurrency)))
    run_sweep(turns, a.sweep, a.base, key or "mock", a.concurrency)
    df = score_sweep(turns, a.sweep)
    # mock (protocol test) output never goes under results/
    out = ROOT / "results/eot_bench_hi/clean/pulse_endpointing" if a.base == PULSE_WS else ROOT / "data/cache/pulse_mock/result"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "eou_sweep.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

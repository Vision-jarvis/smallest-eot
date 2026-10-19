"""Drop-in sidecar: Pulse (`model=pulse`, `language=hi`, `vad_events=true`, `endpointing=false`)
partials + caller audio in, Pulse-2.0-shaped `{"type":"eot","eot_finalized":true,...}` out.

`SidecarCore` is the synchronous decision loop (unit-testable, no network). `run()` wires it to
a Pulse-compatible socket and a WAV file played in real time, printing one JSON event per line.

    python -m sidecar.eot_sidecar --wav call.wav                 # live Pulse (needs SMALLEST_API_KEY)
    python -m sidecar.eot_sidecar --wav call.wav --base ws://127.0.0.1:8765   # local mock

The frozen knob triple (threshold, action_delay, timeout) is read from the model's metadata
(chosen on the dev split, never on a test set) and applied with deployed semantics: the silence
clock is the sidecar's own causal VAD.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from core.asr import Partial
from core.audio import FRAME_MS, HOP_MS
from sidecar.pulse_protocol import PULSE_WS, auth_headers, eot_event, overlap_event, pulse_url

ONSET_DBFS = -45.0


@dataclass
class SidecarCore:
    detector: object
    knobs: dict
    sr: int = 16000
    session_id: str = "local"
    arbiter: object | None = None
    agent_speaking: bool = False
    t: float = 0.0
    _frames: int = 0
    _armed: bool = True
    _overlap_buf: list = field(default_factory=list)
    _overlap_pending: bool = False
    _overlap_done: bool = False
    compute_ms: list = field(default_factory=list)

    def __post_init__(self) -> None:
        self.detector.start_turn(self.sr, None)

    def on_pulse(self, msg: dict) -> None:
        """Feed one Pulse server message (stamped with the current audio clock)."""
        if msg.get("type") == "transcription" and msg.get("transcript"):
            self.detector.push_partial(Partial(self.t, msg["transcript"], bool(msg.get("is_final"))))

    def on_audio(self, frame: np.ndarray) -> list[dict]:
        t0 = time.perf_counter()
        out: list[dict] = []
        self.detector.push_audio(frame)
        self._frames += 1
        self.t = self._frames * FRAME_MS / 1000
        if self.agent_speaking and self.arbiter is not None:
            out += self._overlap(frame)
        if self._frames % (HOP_MS // FRAME_MS) == 0 and not self.agent_speaking:
            sil = self.detector.silence_s(self.t)
            if sil <= 0:
                self._armed = True
            else:
                p = self.detector.score(self.t, True)
                k = self.knobs
                if self._armed and ((sil >= k["action_delay"] and p > k["threshold"]) or sil >= k["timeout"]):
                    out.append(eot_event(self.session_id, p, sil * 1000, self.t))
                    self._armed = False
        self.compute_ms.append((time.perf_counter() - t0) * 1000)
        return out

    def _overlap(self, frame: np.ndarray) -> list[dict]:
        loud = 20 * np.log10(np.sqrt(np.mean(frame**2)) + 1e-9) > ONSET_DBFS
        self._overlap_buf = (self._overlap_buf + [frame])[-60:]  # keep 1.2 s
        if not self._overlap_pending and not self._overlap_done and loud:
            self._overlap_pending = True
            self._overlap_buf = self._overlap_buf[-6:]  # 100 ms pre-roll + onset frame
            self._onset_t = self.t - FRAME_MS / 1000
            return []
        if self._overlap_pending:
            elapsed_ms = (len(self._overlap_buf) - 5) * FRAME_MS
            if elapsed_ms >= 300:
                audio = np.concatenate(self._overlap_buf)
                r = self.arbiter.classify(audio, None, self.sr)
                if r["decided_at_ms"] <= elapsed_ms:
                    self._overlap_pending, self._overlap_done = False, True
                    return [overlap_event(r["label"], r["conf"], r["decided_at_ms"], self._onset_t + r["decided_at_ms"] / 1000)]
        return []

    def set_agent_speaking(self, speaking: bool) -> None:
        self.agent_speaking = speaking
        self._overlap_pending = self._overlap_done = False
        self._overlap_buf = []


async def run(wav: str, base: str, key: str, model_variant: str = "stacked") -> None:
    import soundfile as sf
    import websockets

    from core.audio import frames, resample
    from models.pulse_eot_hi import ART, PulseEoTHi

    det = PulseEoTHi.load(model_variant)
    knobs = json.loads((ART / f"pulse_eot_hi_{model_variant}.json").read_text())["dev_frozen_knobs_lat5"]
    x, sr = sf.read(wav, dtype="float32")
    x = resample(x.mean(axis=1) if x.ndim > 1 else x, sr, 16000)
    core = SidecarCore(det, knobs)
    url = pulse_url(sample_rate=16000, endpointing=False, vad_events=True, base=base)
    async with websockets.connect(url, additional_headers=auth_headers(key), max_size=None) as ws:
        t0 = time.perf_counter()

        async def reader():
            async for m in ws:
                core.on_pulse(json.loads(m))

        rd = asyncio.create_task(reader())
        for i, f in enumerate(frames(x, 16000)):
            await asyncio.sleep(max(0.0, (i + 1) * FRAME_MS / 1000 - (time.perf_counter() - t0)))
            await ws.send((np.clip(f, -1, 1) * 32767).astype(np.int16).tobytes())
            for ev in core.on_audio(f):
                print(json.dumps(ev), flush=True)
        await ws.send(json.dumps({"type": "close_stream"}))
        try:
            await asyncio.wait_for(rd, 10)
        except asyncio.TimeoutError:
            rd.cancel()
    ms = np.array(core.compute_ms)
    print(json.dumps({"type": "sidecar_stats", "frames": len(ms), "compute_ms_p50": float(np.percentile(ms, 50)), "compute_ms_p95": float(np.percentile(ms, 95)), "compute_ms_max": float(ms.max())}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--base", default=PULSE_WS)
    ap.add_argument("--variant", default="stacked")
    a = ap.parse_args()
    key = os.environ.get("SMALLEST_API_KEY", "")
    if a.base == PULSE_WS and not key:
        raise SystemExit("PENDING: set SMALLEST_API_KEY to connect to live Pulse, or pass --base ws://127.0.0.1:8765 for the mock.")
    asyncio.run(run(a.wav, a.base, key or "mock", a.variant))


if __name__ == "__main__":
    main()

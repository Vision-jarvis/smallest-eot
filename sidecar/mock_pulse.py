"""A local MOCK of the Pulse streaming STT socket, for offline tests of the client and sidecar.

It speaks the documented wire format (sidecar/pulse_protocol.py) but is NOT Pulse: text comes
from Vosk small-hi, VAD events from Silero, and `endpointing=true` finals fire after
`eou_timeout_ms` of Silero-measured trailing silence (how the Pulse docs describe it). Numbers
measured against this mock are never reported as Pulse numbers.

    python -m sidecar.mock_pulse --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from urllib.parse import parse_qs, urlparse

import numpy as np

from core.audio import from_pcm16
from core.vad import SileroStream
from sidecar.pulse_protocol import transcription, vad_event


class MockSession:
    def __init__(self, sr: int, endpointing: bool, eou_ms: int) -> None:
        from vosk import KaldiRecognizer

        from core.asr import _model

        self.sid = str(uuid.uuid4())
        self.sr = sr
        self.endpointing = endpointing
        self.eou_s = eou_ms / 1000
        self.rec = KaldiRecognizer(_model(), sr)
        self.vad = SileroStream(sr=sr)
        self.was_speech = False
        self.pending = ""
        self.last_partial = ""
        self.final_sent_for_pause = False

    def feed(self, pcm: bytes) -> list[dict]:
        out: list[dict] = []
        x = from_pcm16(np.frombuffer(pcm, dtype=np.int16))
        self.vad.push(x)
        t = self.vad.t
        if self.vad.in_speech and not self.was_speech:
            out.append(vad_event(self.sid, "speech_started", self.vad.speech_start or t))
            self.final_sent_for_pause = False
        if not self.vad.in_speech and self.was_speech:
            out.append(vad_event(self.sid, "speech_ended", self.vad.last_speech_end or t))
        self.was_speech = self.vad.in_speech
        vosk_final = self.rec.AcceptWaveform(pcm)
        if vosk_final:
            text = json.loads(self.rec.Result())["text"]
            self.pending = (self.pending + " " + text).strip()
            self.last_partial = ""
        else:
            part = json.loads(self.rec.PartialResult())["partial"]
            if part and part != self.last_partial:
                self.last_partial = part
                out.append(transcription(self.sid, (self.pending + " " + part).strip(), False))
        silence = self.vad.trailing_silence(t)
        if self.endpointing:
            if (self.pending or self.last_partial) and not self.final_sent_for_pause and silence >= self.eou_s:
                out.append(self._final())
        elif vosk_final and self.pending:
            out.append(self._final())
        return out

    def _final(self, is_last: bool = False) -> dict:
        text = (self.pending + " " + self.last_partial).strip()
        self.pending, self.last_partial = "", ""
        self.final_sent_for_pause = True
        self.rec.Reset()
        return transcription(self.sid, text, True, is_last)


async def handler(ws) -> None:
    q = parse_qs(urlparse(ws.request.path).query)
    if q.get("model", [""])[0] != "pulse":
        await ws.send(json.dumps({"type": "error", "error_code": "MOCK_ONLY_PULSE", "message": "mock supports model=pulse"}))
        return
    sess = MockSession(int(q.get("sample_rate", ["16000"])[0]), q.get("endpointing", ["true"])[0] == "true", int(q.get("eou_timeout_ms", ["600"])[0]))
    async for msg in ws:
        if isinstance(msg, bytes):
            for m in sess.feed(msg):
                await ws.send(json.dumps(m))
            continue
        ctl = json.loads(msg)
        if ctl.get("type") == "finalize":
            await ws.send(json.dumps(sess._final()))
        elif ctl.get("type") == "close_stream":
            await ws.send(json.dumps(sess._final(is_last=True)))
            await ws.close()
            return
        elif ctl.get("type") in ("ping", "keepalive", "keep_alive", "keep-alive"):
            await ws.send(json.dumps({"type": "pong"}))


async def serve(port: int) -> None:
    import websockets

    async with websockets.serve(handler, "127.0.0.1", port, max_size=None):
        await asyncio.Future()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    asyncio.run(serve(ap.parse_args().port))

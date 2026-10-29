"""Smallest Pulse streaming STT wire protocol, as documented on docs.smallest.ai (read 2026-10-18).

Endpoint  wss://api.smallest.ai/waves/v1/stt/live   auth header `Authorization: Bearer <key>`
Query     model=pulse, language=hi, encoding=linear16, sample_rate=8000|16000,
          endpointing=true|false, eou_timeout_ms=100..10000 (defaults 600 with endpointing on,
          800 with it off), vad_events=true, word_timestamps=true,
          no_format_delay=true (changelog 2026-10-13: with itn_normalize=true Pulse holds back a
          number still being dictated at the end of a final and emits it with the next final;
          no_format_delay=true opts out. The sidecar always sends it, for the same reason it sends
          endpointing=false: Pulse must not delay text while our model decides the turn.)
Client -> binary PCM frames (docs recommend 4096-byte chunks); JSON controls
          {"type":"finalize"}, {"type":"close_stream"}, {"type":"ping"}
Server -> {"type":"transcription","status":"success","session_id":..,"transcript":..,
           "is_final":bool,"is_last":bool, ["words":[{"word","start","end","confidence"}]],
           ["eot_finalized": true  (pulse-2 only)]}
          {"type":"speech_started"|"speech_ended","session_id":..,"timestamp":s}
          {"type":"error","error_code":..,"message":..}

The sidecar emits the Pulse-2.0-shaped event the problem statement specifies:
          {"type":"eot","eot_finalized":true,"p_eot":..,"silence_ms":..,"timestamp":..,"session_id":..}
          {"type":"user_overlap","label":"backchannel|barge_in|noise","conf":..,"decided_at_ms":..,"timestamp":..}
"""

from __future__ import annotations

from urllib.parse import urlencode

PULSE_WS = "wss://api.smallest.ai/waves/v1/stt/live"
CHUNK_BYTES = 4096


def pulse_url(
    *,
    language: str = "hi",
    sample_rate: int = 16000,
    endpointing: bool = False,
    eou_timeout_ms: int | None = None,
    vad_events: bool = True,
    word_timestamps: bool = True,
    no_format_delay: bool = True,
    base: str = PULSE_WS,
) -> str:
    q = {
        "model": "pulse",
        "language": language,
        "encoding": "linear16",
        "sample_rate": sample_rate,
        "endpointing": str(endpointing).lower(),
        "vad_events": str(vad_events).lower(),
        "word_timestamps": str(word_timestamps).lower(),
        "no_format_delay": str(no_format_delay).lower(),
    }
    if eou_timeout_ms is not None:
        if not 100 <= eou_timeout_ms <= 10000:
            raise ValueError("eou_timeout_ms must be in [100, 10000] per Pulse docs")
        q["eou_timeout_ms"] = eou_timeout_ms
    return f"{base}?{urlencode(q)}"


def auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def transcription(session_id: str, text: str, is_final: bool, is_last: bool = False, words: list | None = None) -> dict:
    msg = {"type": "transcription", "status": "success", "session_id": session_id, "transcript": text, "is_final": is_final, "is_last": is_last}
    if is_final:
        msg.update({"language": "hi", "languages": ["hi"]})
    if words is not None:
        msg["words"] = words
    return msg


def vad_event(session_id: str, kind: str, timestamp: float) -> dict:
    assert kind in ("speech_started", "speech_ended")
    return {"type": kind, "session_id": session_id, "timestamp": round(timestamp, 3)}


def eot_event(session_id: str, p_eot: float, silence_ms: float, timestamp: float) -> dict:
    return {"type": "eot", "eot_finalized": True, "p_eot": round(p_eot, 4), "silence_ms": int(round(silence_ms)), "timestamp": round(timestamp, 3), "session_id": session_id}


def overlap_event(label: str, conf: float, decided_at_ms: int, timestamp: float) -> dict:
    assert label in ("backchannel", "barge_in", "noise")
    return {"type": "user_overlap", "label": label, "conf": round(conf, 4), "decided_at_ms": int(decided_at_ms), "timestamp": round(timestamp, 3)}

# Pulse streaming STT: what the live API actually sends

Observed 2026-10-24 against `wss://api.smallest.ai/waves/v1/stt/live` with `model=pulse`, `language=hi`, `encoding=linear16`, `sample_rate=16000`, `vad_events=true`, `word_timestamps=true`, audio sent in real time in 4096-byte chunks (128 ms). Sources: a 2-call probe (`data/cache/pulse_probe/`, one turn with `endpointing=true, eou_timeout_ms=600`, one with `endpointing=false`), then the 1,200 sweep streams and 400 partials streams (`data/cache/pulse/`, gitignored). Docs read the same day: `docs.smallest.ai/models/speech-to-text/features/endpointing.md` and `.../end-of-utterance-timeout.md`.

## Message shapes

Three message types appeared, nothing else (no `error`, no `eot`, no `utterance_end`):

```json
{"type": "speech_started", "session_id": "1b7a...", "timestamp": 0.544}
{"type": "speech_ended",   "session_id": "1b7a...", "timestamp": 1.04}
{"type": "transcription", "status": "success", "session_id": "1131...",
 "transcript": "बस एक", "transcription": "बस एक",
 "is_final": false, "is_last": false, "language": null, "languages": null}
{"type": "transcription", "status": "success", "session_id": "1131...",
 "transcript": " बस एक।", "transcription": " बस एक।",
 "words": [{"start": 1.62, "end": 1.7, "word": "बस", "confidence": 0.9687},
           {"start": 2.34, "end": 2.42, "word": "एक।", "confidence": 0.9336}],
 "is_final": true, "is_last": false, "language": "hi", "languages": ["hi"]}
{"type": "transcription", "status": "success", "session_id": "1131...",
 "transcript": "", "transcription": "", "words": [],
 "is_final": true, "is_last": true, "language": null, "languages": null}
```

What differs from what `sidecar/pulse_protocol.py` assumed (docs read 2026-10-18):

1. **No end-of-turn field on `model=pulse`.** No `eot_finalized`, no `eot` message, in any of the logs. The docs say Pulse 2.0 has built-in end-of-turn detection; `model=pulse` signals a finished segment only through `is_final: true`. Our sidecar's `{"type":"eot",...}` event stays our own addition.
2. **The text is sent twice**, as `transcript` and as `transcription` (same string). Finals often start with a space and carry Devanagari punctuation (`।`); `core/text.normalise` strips both.
3. **`words` only on finals.** Partials carry no word timestamps. Word `start`/`end` are seconds on the audio clock.
4. **Partials are cumulative within a segment** ("बस", then "बस एक"), and a final replaces them; the next segment starts from empty. English loanwords come back in Latin script inside Devanagari text ("network का issue था", "manager", "number"), so lexical features must handle mixed script. This matches the `Partial` record the Vosk replay already used, so the partials drop into `eval/asr_cache` unchanged.
5. **`language` is set only on finals** (`"hi"`, `["hi"]`); `null` on partials and on the close-of-stream message.
6. **`close_stream` gets one last message**, `is_final: true, is_last: true`, empty text, then the server closes the socket. It must not be read as an end-of-turn (it arrives about 0.2 to 0.7 s after the last audio).
7. **VAD events carry an audio-clock `timestamp`**, but they arrive late. In the probe, `speech_ended` with timestamp 1.04 s arrived at 1.92 s on the audio clock with `eou_timeout_ms=600` (0.88 s later), and the final followed 38 ms after it.

## Endpointing semantics that matter for scoring

- **`is_final` is not always a pause.** With defaults, `finalize_on_words` is on and Pulse also emits finals in the middle of continuous speech once the pending transcript reaches a word count (docs: "Pulse may emit multiple finals before you send a finalize message; join these together"). In turn `hi__4147` (300 ms run) a 10-word segment was finalized 1.3 s after its last word while the speaker kept going, the next segment was cut the same way at 10 words, and only the third final (after `speech_ended`) was the real end of the turn. Across the 200 dev-test turns, finals per turn were 3.18 at `eou_timeout_ms=300` and 2.7 to 2.8 from 600 up, while silence finals per turn fell from 1.86 to 1.00. So from 600 ms up, roughly 60 to 65% of all finals are word-count finals in the middle of speech, and 48 to 55% of turns get at least one final before the speaker is done (sweep table, "any final"). An agent that treats every final as end-of-turn will cut in mid-sentence.
- **The silence final** (the one the `eou_timeout_ms` window produces) comes right after a `speech_ended` event (median 72 ms later, p90 106 ms, in the 300 ms run). `speech_ended` is tied to the window, not to a fixed VAD: its count per turn falls as the timeout rises (1.88 at 300 ms, 1.15 at 600 ms, 1.01 at 1200 ms). Its `timestamp` is close to the end of speech, and the final arrives about timeout + 0.33 s after that timestamp (median, 800 to 1200 ms runs: 1.13 / 1.33 / 1.53 s). Measured from the labelled true end of the turn, Pulse's mean latency is the timeout plus 317 to 397 ms. That overhead includes the 128 ms chunking of our client (64 ms on average), the network round trip from this laptop, and server-side processing; it is what a client sending 4096-byte chunks sees. `adapters/pulse_endpointing.final_times(log, "silence")` keeps only the first final after a `speech_ended` with no `speech_started` in between; `"all"` keeps every final. The sweep table reports both.
- **`endpointing=false`** (the sidecar's mode) still finalizes, on the model's own utterance timer (default 800 ms per docs). In the probe the final came 2.8 s after the true end of a short turn; over the 400-turn partials run, finals arrived a median 2.48 s (p10 1.92, p90 3.02 s) after their last word ended. So in that mode our detector, not Pulse's final, has to make the call. Partials keep flowing either way, roughly one per second of speech (4,027 transcription messages over 400 turns, 3.06 finals per turn), which is coarser than the 100 ms Vosk replay.
- Since 2026-07-12 endpointing defaults to on, so `adapters/pulse_endpointing.py` passes `endpointing=true` explicitly for the sweep and `endpointing=false` for the partials cache, and `sidecar/pulse_protocol.pulse_url` passes the flag on every connection.

## Billing

No usage or balance field appears in any message, and the socket gives no per-session duration. Spend in this repo is computed as audio minutes streamed (turn plus the 3 s pad, per log file on disk) times the Standard-plan streaming list price, $0.006/min (`adapters/pulse_endpointing.streamed_minutes`).

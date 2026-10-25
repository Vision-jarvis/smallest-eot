## Pulse `eou_timeout_ms` sweep (REAL, live API 2026-10-24) vs the real-VAD timer, eot-bench `hi` dev-test half

`python -m adapters.pulse_endpointing` streamed each of the 200 dev-test turns (plus a 3 s noise-floor pad) in real time (4096-byte chunks = 128 ms) to `wss://api.smallest.ai/waves/v1/stt/live` with `model=pulse, language=hi, endpointing=true, eou_timeout_ms=<x>` (other parameters at their defaults, so `finalize_on_words` is on) and logged every message's arrival time on the audio clock. Two readings of the Pulse stream: **silence final** = the first `is_final` after a `speech_ended` VAD event (the eou_timeout_ms window expired); **any final** = every `is_final`, which also includes word-count finals Pulse emits in the middle of speech. The timer row is our Silero VAD silence timer on the same turns. Same scorer for all (eval/deployed.py). FCR = share of holds (pauses of 0.2 to 5 s inside the turn) with an end-of-turn event inside the hold. Because a late event can land after the user has resumed speaking (a cut-in that FCR does not count), the table also gives **premature turns** = share of turns with any end-of-turn event before the true end. Latency = true end of speech to the first event at or after it. Brackets: 95% Wilson interval.

| eou_timeout_ms | Pulse silence final: FCR | premature turns | mean latency | p90 latency | latency minus timeout | Pulse any final: FCR / premature / latency | Real-VAD timer: FCR / premature / latency |
|---|---|---|---|---|---|---|---|
| 300 | 9.3% [6.3%, 13.5%] | 36.5% | 627 ms | 758 ms | +327 ms | 13.3% / 54.5% / 582 ms | 50.4% / 40.5% / 281 ms |
| 400 | 5.2% [3.1%, 8.8%] | 31.5% | 717 ms | 831 ms | +317 ms | 8.5% / 52.5% / 682 ms | 27.0% / 36.0% / 381 ms |
| 600 | 2.0% [0.9%, 4.6%] | 11.5% | 931 ms | 1047 ms | +331 ms | 7.7% / 49.0% / 847 ms | 6.0% / 18.0% / 581 ms |
| 800 | 0.8% [0.2%, 2.9%] | 4.0% | 1197 ms | 1259 ms | +397 ms | 8.5% / 47.5% / 1007 ms | 3.2% / 6.5% / 781 ms |
| 1000 | 0.8% [0.2%, 2.9%] | 1.5% | 1368 ms | 1459 ms | +368 ms | 8.5% / 48.0% / 1183 ms | 1.2% / 4.0% / 981 ms |
| 1200 | 0.4% [0.1%, 2.2%] | 1.0% | 1577 ms | 1676 ms | +377 ms | 8.5% / 49.0% / 1334 ms | 1.2% / 1.5% / 1181 ms |

n = 200 turns, 248 holds, 200 turn ends.

FCR at a mean-latency budget (best setting whose mean latency fits; the deployed-policy analogue of eot-bench FCR@x):

| Budget | Pulse silence final | Pulse any final | Real-VAD timer |
|---|---|---|---|
| 300 ms | - | - | 50.4% (300 ms) |
| 600 ms | - | 13.3% (300 ms) | 6.0% (600 ms) |
| 800 ms | 5.2% (400 ms) | 8.5% (400 ms) | 3.2% (800 ms) |
| 1000 ms | 2.0% (600 ms) | 7.7% (600 ms) | 1.2% (1000 ms) |

Pulse-EoT-hi v0.1 with real Pulse partials (endpointing=false) instead of Vosk, same 200 turns, eot-bench knob sweep (`python -m eval.run_bench --detector pulse_eot_hi --asr pulse`; same weights, only the text source changes):

| Text source | FCR@300 | FCR@600 | Lat@5% | Lat@10% | AUC |
|---|---|---|---|---|---|
| Vosk small-hi replay | 43.5% | 5.2% | 611 ms | 500 ms | 0.822 |
| Pulse partials (REAL) | 44.0% | 4.8% | 595 ms | 508 ms | 0.834 |

Paired bootstrap over 200 turns (1000 resamples), Pulse partials minus Vosk, each at its own Lat@5% knobs: latency -16.0 ms (95% CI -51.5 to +20.5), FCR +0.00 pp (95% CI -4.18 to +4.23).

At the dev-frozen knobs: Vosk FCR 5.6% at 1156 ms, Pulse partials FCR 4.0% at 1221 ms.

# Smallest.ai: "Haan, matlab..." Streaming end-of-turn and barge-in arbitration for Hindi/Hinglish phone calls

*The spec this repo was built to, published as written (the outreach section is omitted). Its targets, such as "prove it beats silence timers", are goals, not results: what was actually measured, including the negative results, is in the [README](../README.md) and [results/report.md](../results/report.md).*

*Freshness check 2026-10-29: Pulse ITN trailing-number hold (changelog 2026-10-13: with `itn_normalize=true` a number still being dictated at the end of a final is held back to the next final; `no_format_delay=true` opts out) added to Part D and the digit read-out slice; `hi` vs `hi-dev` number normalization (changelog 2026-10-21) noted for lexical features; Pipecat v1.15.0 (2026-10-29) `ClassifierUserTurnStopStrategy` added as a future baseline family and integration target; Smallest's own silent-hold and Hindi telephony posts (2026-10-22, 2026-10-26) added as the demo scenario. Pulse 2.0 EoT is still English-only (premise holds).*

*Freshness check 2026-10-23: Pulse streaming price corrected to $0.006/min (Standard plan), endpointing default-ON since 12 Jul 2026 noted (sidecar passes `endpointing=false`), LiveKit TurnDetector v1/v1-mini described as audio-based, TurnBench and MultiPartyTurnBench cited.*

> Build a causal, CPU-cheap turn-taking layer that tells a Pulse-based voice agent, every 100 ms, whether a Hindi or Hinglish caller on an 8 kHz phone line has finished speaking, is only thinking, or is just saying "haan" while the agent talks, and prove it beats silence timers and the open turn detectors on a public Hindi benchmark plus a new telephony stress set.

**Company:** Smallest.ai (Smallest AI), full-stack voice AI (Pulse STT, Lightning TTS, Electron LLM, Hydra speech-to-speech, Atoms agent platform), Series A, ~60 people, Bengaluru / San Francisco

## 1. About the company

Smallest.ai was founded in 2023 by Sudarshan Kamath (CEO) and Akshat Mandloi (CTO), both IIT Guwahati graduates who worked on autonomous-vehicle systems at Bosch. It raised an $8M seed led by Sierra Ventures in October 2025 (with 3one4 Capital and Better Capital) and a $13M Series A led by Seligman Ventures on 1 August 2026, taking total funding past $21M. Inc42 reports ~60 employees, 38-language support, and customers including Truecaller; the stated use of funds is scaling the voice platform across financial services, healthcare, contact centres and BPO.

The stack, all documented on docs.smallest.ai:

- **Pulse** (multilingual streaming STT, `wss://api.smallest.ai/waves/v1/stt/live?model=pulse`): Hindi is a first-class language code (`hi`, "bilingual, mixed-script", i.e. Hinglish), plus `hi-dev` and a `north_indic` auto-detect aggregator. Pulse publishes Hindi WER on telephony and contact-centre audio. Streaming list price is $0.006/min on the Standard plan (docs.smallest.ai changelog, 28 May 2026).
- **Pulse 2.0** (`model=pulse-2`): adds a built-in end-of-turn (EoT) model. Finals that close on a turn boundary carry `eot_finalized: true`. It is **English-only**: any `language` other than `en` returns `LANGUAGE_NOT_SUPPORTED_BY_MODEL` and closes the socket.
- **Hydra** (`wss://api.smallest.ai/waves/v1/s2s`): full-duplex speech-to-speech with server-side turn detection and native barge-in. The model card says it supports **English only** right now, with more languages "on the roadmap".
- **Atoms**: the hosted no-code/SDK agent platform that wraps Pulse + Electron + Lightning for phone calls.

Turn-taking is at the centre of this business. Every Atoms call runs on one decision repeated for each caller turn: speak now, or keep listening. Smallest clearly treats it as a headline metric. It benchmarked "Pulse EoT v1" against LiveKit, Deepgram Flux, SmartTurn and others on the public `livekit/eot-bench` and published the table on its Pulse 2.0 model card. But that table, and the model, cover English only, while Smallest's home market and its Series A customers talk in Hindi and Hinglish.

## 2. The problem

**The gap, from Smallest's own docs.**

1. Pulse 2.0's learned EoT exists only for English. A Hindi agent on `model=pulse` closes a turn with a trailing-silence timer, `eou_timeout_ms`. Per the endpointing page this defaults to 600 ms with endpointing on and 800 ms with it off. Endpointing itself has been ON by default since 12 Jul 2026 (docs.smallest.ai changelog), so an integration that does not set it explicitly gets Pulse's own turn closing. The docs themselves recommend 800 to 1200 ms for voice agents ("start at 1000 ms for conversational voice agents").
2. Hydra, the full-duplex product that handles barge-in natively, is English-only.
3. In Atoms, "Smart Turn Detection" is **off by default**. It is available only for English or Hindi default-language agents, and when the model is unsure it waits up to a "Wait Time" that defaults to **3 s** (range 1 to 10 s). The default listening path is VAD with a 0.4 s "Release Time".

So a Hindi/Hinglish agent built on Smallest's stack today has to pick a fixed silence number:

- **Short (400 to 600 ms):** the bot talks over callers in the middle of a thought. Hindi speakers routinely pause after a filler ("matlab...", "woh...") or before the clause-final verb.
- **Long (1000 ms or more):** every finished turn costs a second of dead air. The cascade (Pulse final, then Electron, then Lightning) sits on top of that second.

**A concrete call (NBFC collections, Hinglish, 8 kHz):**

> Agent: "Sir, aapka EMI 4,250 rupaye ka 5 tareekh ko due tha. Aap kab tak pay kar payenge?"
> Caller: "Haan dekhiye, woh... [650 ms] matlab salary thodi late aati hai na, toh main... [720 ms] pandrah tareekh tak [580 ms] kar dunga."

With `eou_timeout_ms=600` the agent cuts in at the 650 ms pause after "woh...". It replies to an empty promise-to-pay ("Kya aap aaj pay kar sakte hain?"), and the caller has to repeat himself, or hangs up. With 1000 ms the agent survives this turn. But on the next turn, a plain "Haan ji, theek hai.", it waits a full second before even starting the LLM. The commitment the lender actually needs ("pandrah tareekh tak kar dunga") arrives in the last 1.2 s of a 5 s turn. Hindi syntax puts the verb, which carries the commitment, at the very end.

**Why Smallest would pay for this.**

- Its Series A thesis names financial services, contact centres and BPO, which in India are dominated by Hindi/Hinglish phone calls.
- A turn-taking error is visible to the end customer, so it causes churn. A WER error is mostly invisible.
- The company has already built the English version (Pulse EoT v1), already benchmarks on the right public harness, and already exposes the right API surface (`eot_finalized`, `vad_events`). The missing piece is the Hindi/Hinglish model, a telephony-grade evaluation to show it works, and the barge-in half of the problem that Hydra solves only for English.

**Why this is not the Bolna problem.** Bolna's statement (another writer) is about hiding *downstream* latency: speculative LLM/TTS generation and vendor tail latency once a turn is assumed to have ended. This statement is about the *upstream decision signal itself*: when the turn has ended, and whether the user's speech during agent playback is a real interruption. Here, latency means dead air between the user's true acoustic offset and the EoT event. All LLM/TTS time is excluded by construction. The two pieces compose: a good EoT probability is the input a speculative generator would consume.

## 3. Why it is hard

**3.1 Verb-final syntax pushes the semantic end to the last word.** In Hindi (SOV), the clause usually ends with the verb group, e.g. "kar dunga", "ho gaya tha", "bhej dijiye". Every pause before that point is syntactically incomplete. An English-trained text or audio cue ("the clause is complete, pitch fell") fires too early. Example: "Mera account number hai [900 ms, reading from a card] 4 5 7 2..." has falling pitch on "hai", a complete-looking copular clause, and still continues. Compare "Main kal [600 ms] branch mein jaake [500 ms] jama kar dunga": three pauses, only the last is a turn end.

**3.2 Hindi-specific hold markers, often code-mixed.** Fillers and connectives that *project more speech*: "matlab", "woh", "ki" (complementiser: "mujhe lagta hai ki..."), "toh", "aur", "lekin", "kyunki", "jaise", English "so", "actually", "basically". Turn-final markers that *yield the floor*: tag questions "na?", "hai na?", "theek hai?", polite closers "ji", "bas", "itna hi". The same word can do both: "haan" can be an answer ("Haan." = shift), a turn-initial planner ("Haan... dekhiye..." = hold), or a backchannel during agent speech (neither). Pulse returns these in Devanagari, romanised Hinglish or mixed script, so a lexical feature has to survive script variation.

**3.3 Telephony narrowband and noise.** Indian outbound calls run over 8 kHz codecs. IndicVoices' own conversational data was recorded "over a 8kHz channel" through a cloud telephony bridge. At 8 kHz the >4 kHz band disappears, and with it much of the breathiness and fricative cues audio turn models learn from. Street noise, a TV, or a family member ("Mummy, kaun hai phone pe?") create false `speech_started` events and fake "continuations" that keep a silence timer from ever firing. Smallest's Hydra docs say plainly that its VAD "is conservative but not perfect" and that discarded turns must be ignored client-side.

**3.4 Backchannel vs barge-in while the agent is speaking.** In a full-duplex agent, user speech during playback must be classified in ~200 to 300 ms:
- "haan... haan", "achha", "hmm", "ji" while the agent reads loan terms = **backchannel**: keep talking.
- "nahi nahi, ruko", "ek minute", "wrong number hai" = **barge-in**: stop within 300 ms or the agent talks over a refusal.
- A cough, a horn, or "haan" said to someone else in the room = **noise**: ignore.

Hydra cancels the in-flight response on user speech ("cancelled", reason "interrupted") and is English-only. In a VAD pipeline, any of the above stops the agent. Atoms' only mitigation is a blunt "Interruption Backoff Timer" during which the caller cannot interrupt at all.

**3.5 The public Hindi benchmark is easy compared with real calls, and no open model targets the hard part.** `livekit/eot-bench-data` has a 400-turn Hindi split, but it is 16 kHz human-to-agent restaurant ordering with mostly Devanagari transcripts. On it, LiveKit Turn Detector v1 already reaches a 10.5% false-cutoff rate at a 300 ms latency budget. LiveKit's current TurnDetector v1 and v1-mini are audio-based models (14 languages including Hindi, open weights, v1-mini runs on CPU); the earlier text-only MultilingualModel is deprecated. So the open bar this statement must clear is an audio model, not a transcript classifier. That number tells Smallest little about an 8 kHz collections call in Hinglish. Community Hinglish fine-tunes of Smart Turn exist on Hugging Face (e.g. `deveshu/hinglish-turn-detector`, MIT). They report clip-level accuracy on held-out clips, not streaming false-cutoff vs latency trade-offs. So the problem has two halves: building the model, and building an honest, telephony-grade yardstick. Turn-taking benchmarks are moving fast: TurnBench ([arXiv:2608.25218](https://arxiv.org/abs/2608.25218), Aug 2026) and MultiPartyTurnBench ([arXiv:2610.20172](https://arxiv.org/abs/2610.20172), Oct 2026) are recent public benchmarks; neither is an 8 kHz Hindi/Hinglish telephony set, which is the gap Bharat-Turn-8k fills.

**3.6 A tight latency budget on a shared CPU.** The decision runs on every 100 ms hop, for every concurrent call, next to Pulse. A 0.5B-parameter LM per call (the deprecated LiveKit text detector was Qwen2.5-0.5B, ~50 to 160 ms per turn) is too heavy at contact-centre concurrency. The model has to be small enough to sit beside the STT server.

## 4. What you must build

**Part A: Streaming Hindi/Hinglish EoT model ("Pulse-EoT-hi").**
- Input at each 100 ms hop *t*: causal audio up to *t* (8 kHz or 16 kHz mono), plus, optionally, the Pulse streaming partial/final transcripts *as they had actually arrived by t* (with their real arrival timestamps), plus the last agent utterance text (dialogue context).
- Output: `p_eot(t)` in [0, 1], the probability that the user's turn has ended, emitted for every hop during user silence.
- A decision policy with exactly three eot-bench knobs (`threshold`, `action_delay`, `timeout`) converts `p_eot` into an `eot` event. Recommended formulation (not mandatory): a discrete-time hazard model over silence duration, with features from an audio encoder (e.g. initialised from Smart Turn v3.2, BSD-2, Whisper-tiny based, ~8M params) fused with a small lexical head on the trailing Pulse tokens (verb-group, connective, filler, tag-question classes, script-normalised).

**Part B: Barge-in arbiter (agent-speaking state).**
- Input: causal user-channel audio while the agent is playing (assume echo-cancelled user channel, i.e. agent audio is never mixed into the user channel), plus optional Pulse partials.
- Output within a declared window (≤ 300 ms after user speech onset): one of `{backchannel, barge_in, noise}` with a confidence, and a streaming update allowed until 600 ms.

**Part C: Bharat-Turn-8k stress benchmark.**
- Built from IndicVoices Hindi conversational audio (two-party, 8 kHz telephone role-play of customer care for ride-hailing and food delivery, per the IndicVoices paper), with per-speaker voice activity and transcripts aligned. Hold/shift/backchannel labels are derived as in Section 5.
- Slices: (i) clean conversational; (ii) filler/connective-preceded holds; (iii) digit and entity read-outs (note, changelog 2026-10-21: `language=hi` returns digits, e.g. Rs 25,000, for Hindi or English numbers, while `hi-dev` leaves Hindi-spoken numbers as Devanagari words, so any lexical feature that looks for digits must also match number words); (iv) noise-augmented at 10 and 5 dB SNR; (v) codec-simulated (μ-law 8 kHz, and AMR-NB if available via ffmpeg).
- Released as a script that regenerates the set from the licensed sources (no redistribution of gated audio), plus a frozen manifest of span IDs and labels.

**Part D: Drop-in integration.** A sidecar that connects to Pulse (`model=pulse`, `language=hi`, `vad_events=true`, `endpointing=false`, `no_format_delay=true`; the last because, per the Pulse changelog of 2026-10-13 (https://docs.smallest.ai/models/changelog/pulse-stt), with `itn_normalize=true` Pulse holds back a number still being dictated at the end of a final and emits it with the next final, which would delay exactly the digit read-out text the hold decision needs and inflate latency on account-number turns; the flag must be passed explicitly because Pulse endpointing defaults to ON since 12 Jul 2026, and otherwise Pulse's silence timer would close turns before our model decides), consumes its partials and VAD events, and emits a Pulse-2.0-shaped message (`{"type":"eot","eot_finalized":true,"p_eot":0.93,"timestamp":...}`). That lets an Atoms or LiveKit/Pipecat agent switch from the silence timer with no other code change.

**Out of scope:** LLM response generation, speculative decoding, TTS, and vendor latency (the Bolna problem); ASR accuracy; languages other than Hindi/Hinglish (stretch only).

**Honesty rules (violations void the score):**
1. **Causality.** `p_eot(t)` may depend only on audio with timestamp ≤ *t* and on transcript messages whose *arrival* time ≤ *t*. Any model lookahead (e.g. a centred conv) is added to reported latency.
2. **No oracle text at inference.** Test-time transcripts must come from a streaming ASR run live or replayed with logged arrival times (Pulse streaming, or an open streaming ASR). Ground-truth transcripts are allowed for training labels only.
3. **Test sets are untouchable.** `livekit/eot-bench-data` Hindi validation and the Bharat-Turn-8k test split are never used for training, threshold tuning or early stopping. Knobs are tuned on a dev split and frozen. The same knob triple is reported on every test set: no per-test-set or per-slice retuning.
4. **Disjointness.** Bharat-Turn splits are speaker-disjoint *and* conversation-disjoint. No Smart Turn data clip may share a speaker with Bharat-Turn test (if speaker IDs are unavailable, exclude the overlapping source entirely).
5. **Budget.** p95 compute ≤ 10 ms per 100 ms hop on one x86 CPU core (int8 allowed), model ≤ 50 MB. The lexical head may not call a hosted LLM.
6. **Full curves, not cherry-picked points.** Every result is a full sweep over the eot-bench knob grid with the Pareto frontier published, not a single favourable operating point.

## 5. Data, services and resources

| Resource | What it gives | Licence (as shown on the page) | Use |
|---|---|---|---|
| `livekit/eot-bench` (GitHub) + `livekit/eot-bench-data` (HF), `hi` config, 400 turns | Real human-to-agent user turns, 16 kHz, word timings, silence spans, agent context; harness that sweeps threshold/action_delay/timeout | Repo: Apache-2.0; HF card: CC BY 4.0 | **Primary public test set** and the scoring harness (adapter API `predict_batch`) |
| IndicVoices (`ai4bharat/IndicVoices`, `hindi/` folder) | 22 languages, 23.7K h total (11.2K h transcribed); 8% read, 76% extempore, 15% conversational; conversational = two participants on an 8 kHz telephone bridge, role-play scenarios | CC BY 4.0, gated (share contact info) | Training data + Bharat-Turn-8k source. Extempore answers give long natural holds |
| `pipecat-ai/smart-turn-data-v3.2-train` | ~271K labelled clips across 23 languages incl. Hindi; fields `endpoint_bool`, `midfiller`, `endfiller`, `synthetic` | CC BY 4.0 | Pretraining / augmentation for the complete-vs-incomplete head |
| MUCS 2021 sub-task 2 (OpenSLR 104) | Hindi-English code-switched speech, 89.86 h train / 5.18 h test, 16 kHz, from spoken technical tutorials | CC BY-SA 4.0 | Code-mixed lexical patterns for the text head (lecture speech, *not* turn-taking labels) |
| Gram Vaani 1111 h Hindi (OpenSLR 118) | Spontaneous telephone Hindi, regional variety, 105 h labelled + 1,000 h unlabelled, ~61% of labelled audio at 8 kHz | Academic use free; commercial use needs permission from Gram Vaani | Telephony-channel augmentation and SSL only. **Flag to Smallest**: not for a shipped model without permission |
| MUSAN (OpenSLR 17) | Noise, music, babble | CC BY 4.0 | Noise augmentation at 10/5 dB SNR |
| CALLFRIEND Hindi (LDC96S52) | 60 two-channel 8 kHz telephone conversations, 5 to 30 min | LDC user agreement (paid) | Optional gold-standard two-channel check if Smallest has LDC access |
| Smallest Pulse API (`model=pulse`, `language=hi`) | Streaming partials, word timestamps, diarization, `vad_events`; batch at $0.003/min | Commercial API | Live streaming transcripts at test time; batch diarization + word timing for labelling |

**Label construction for Bharat-Turn-8k.** IndicVoices conversational calls are recorded on one bridge. The paper does not state whether the two parties are on separate channels, so check on download.
1. If the two parties are mixed in one channel: run Pulse batch with diarization and word timestamps (or pyannote.audio, MIT) to get per-speaker voice activity, then hand-correct a 30-minute dev subset to measure diarization error.
2. Inter-pausal units: speech runs separated by ≥ 200 ms silence (matching eot-bench's 0.2 s minimum hold span; holds longer than 5 s are dropped, as in its `metrics.py`).
3. A silence span is **hold** if the same speaker resumes before the other speaks. It is **EoT/shift** if the other speaker takes the floor (allowing ≤ 1 s overlap). It is **backchannel** if the other speaker produces < 1 s of speech made only of {haan, ha, hmm, achha, ji, haanji, ok, theek} and the first speaker continues.
4. Barge-in events: other-speaker speech > 1 s that starts during a turn and leads the first speaker to stop within 1.5 s.

**Synthetic augmentation (calibrated, never in test).**
- Insert Hindi fillers recorded by Lightning TTS (Hindi voices) into real holds.
- Splice backchannels from IndicVoices speakers onto agent-speech segments.
- Simulate codecs with ffmpeg μ-law 8 kHz.
- Calibrate synthetic hold-duration distributions to the measured IndicVoices hold/shift silence histograms and report both histograms side by side.

**Compute.** Laptop CPU for inference and evaluation. ≤ $50 of rented GPU (one 24 GB card, ~15 to 20 GPU-hours) for fine-tuning. Pulse API spend under $15 (≈ 10 h batch labelling at $0.003/min ≈ $1.80, plus ≈ 5 h streaming replay of test audio at $0.006/min ≈ $1.80; re-costed 2026-10-23 from the earlier $0.004/min figure, 1.5x).

## 6. Interfaces and constraints

**Audio in:** PCM16 mono, 8000 or 16000 Hz, 20 ms frames. The model is called on a 100 ms hop.

**Model API (matches the eot-bench adapter contract):**
```python
class PulseEoTHi:
    adapter_id = "pulse-eot-hi"
    def predict_batch(self, batch):  # item: {"audio": causal_prefix_np, "messages": [...], "partials": [(t_arrival, text, is_final), ...]}
        return [p_eot for item in batch]
```

**Barge-in API:** `classify(audio_since_onset, partials) -> {"label": "backchannel|barge_in|noise", "conf": float, "decided_at_ms": int}`.

**Sidecar wire format (Pulse-2.0 compatible):**
```json
{"type":"eot","eot_finalized":true,"p_eot":0.91,"silence_ms":340,"timestamp":12.48,"session_id":"..."}
{"type":"user_overlap","label":"backchannel","conf":0.88,"decided_at_ms":240,"timestamp":31.02}
```

**Budgets:**
- p95 ≤ 10 ms CPU per hop; model ≤ 50 MB.
- Barge-in decision ≤ 300 ms after onset.
- Sidecar adds ≤ 20 ms beyond Pulse partial arrival.
- Zero GPU at inference.
- Memory ≤ 300 MB per process serving ≥ 50 concurrent calls (measured).

## 7. Evaluation

**Metric definitions** (identical to `livekit/eot-bench` so numbers are comparable with Smallest's published English table):
- **Hold span:** a within-turn silence of 0.2 to 5.0 s. **EoT span:** the silence after a true turn end.
- **False-cutoff rate (FCR):** fraction of hold spans in which the policy fires `eot` before the user resumes.
  FCR = #(hold spans with an eot event) / #(hold spans).
- **Latency:** mean time from true acoustic offset to the `eot` event over EoT spans (forced at `timeout` if never fired).
- **FCR@300 / FCR@600:** the lowest FCR over the knob grid subject to mean latency ≤ 300 / 600 ms.
- **Lat@5% / Lat@10%:** the lowest mean latency subject to FCR ≤ 5% / 10%.
- **Detect@5%:** fraction of EoT spans where `p_eot` crosses the threshold before `timeout`, at the Lat@5% operating point.
- **AUC/AP:** span-level discrimination, as on Smallest's Pulse 2.0 card.
- **Filler-hold FCR (FFCR):** FCR restricted to holds whose last pre-pause token is a filler/connective (matlab, woh, ki, toh, aur, lekin, so, actually...), at the frozen Lat@5% knobs.
- **Barge-in:**
  - False barge-in rate FBR = #(backchannels or noise classified barge_in) / #(backchannels + noise).
  - Missed barge-in rate MBR = #(true barge-ins not flagged within 300 ms) / #(barge-ins).
  - Reaction latency p50/p90 (onset to `barge_in` decision).
  - Arbiter score = 1 - (FBR + MBR) / 2.
- **Noise robustness:** ΔLat@5% = Lat@5%(5 dB SNR) - Lat@5%(clean) on Bharat-Turn-8k.

**Baselines to beat** (all run through the same harness):

| Baseline | Source | Hindi eot-bench (published by LiveKit's harness): FCR@300 / FCR@600 / Lat@5% |
|---|---|---|
| VAD-only silence timer (Silero VAD, MIT) | eot-bench | 49.5% / 7.4% / 700 ms |
| VAP (Ekstedt & Skantze, MIT) | eot-bench | 49.5% / 7.4% / 655 ms |
| SmartTurn v3.2 (BSD-2) | eot-bench | 42.6% / 6.5% / 639 ms |
| LiveKit Turn Detector v1-mini | eot-bench | 27.4% / 5.4% / 624 ms |
| Deepgram Flux (hosted) | eot-bench | 19.2% / 4.5% / 583 ms |
| LiveKit Turn Detector v1 (best published) | eot-bench | 10.5% / 2.0% / 454 ms |
| **Pulse `model=pulse` endpointing**, `eou_timeout_ms` ∈ {300, 400, 600, 800, 1000, 1200} | run by you | to be measured: this is Smallest's actual Hindi production default |
| Community Hinglish Smart Turn fine-tune (`deveshu/hinglish-turn-detector`, MIT) | run by you | to be measured |

Reference point: on English, Smallest's own Pulse EoT v1 scores 28.4% / 8.8% / 784 ms (Pulse 2.0 model card). That is the level of ambition Smallest has already shipped for English.

**Targets.**
- 3-day bar: beat SmartTurn v3.2 and LiveKit v1-mini on Hindi eot-bench (FCR@300 < 27.4% and Lat@5% < 624 ms).
- On Bharat-Turn-8k: dominate the Pulse `eou_timeout_ms` sweep at every point of the Pareto curve.
- Stretch: within 100 ms of LiveKit v1's 454 ms Lat@5% on Hindi eot-bench, while staying ≤ 50 MB on CPU.

**Points matrix (sums to 100):**

| # | Criterion | Measured on | Points |
|---|---|---|---|
| 1 | Lat@5% (lower is better, scaled between VAD 700 ms = 0 and 450 ms = full) | eot-bench `hi` | 20 |
| 2 | FCR@300 (scaled between 49.5% = 0 and 10% = full) | eot-bench `hi` | 10 |
| 3 | Lat@5% and FCR@300 vs the Pulse `eou_timeout_ms` Pareto | Bharat-Turn-8k test | 20 |
| 4 | Filler-hold FCR and digit/entity-readout FCR | Bharat-Turn-8k slices | 10 |
| 5 | Arbiter score and p90 reaction latency ≤ 300 ms | Bharat-Turn-8k overlap set | 15 |
| 6 | Noise and codec robustness (ΔLat@5% at 5 dB, μ-law) | Bharat-Turn-8k slices | 5 |
| 7 | Efficiency: p95 ms/hop, MB, calls per core | Laptop CPU benchmark | 10 |
| 8 | Reproducibility, honesty-rule compliance, failure analysis | Repo + report | 10 |
| | **Total** | | **100** |

**Reporting.**
- Pareto curves (FCR vs latency) for every system on both test sets, all on one plot per set.
- A per-slice table.
- Ablations: audio-only vs audio+Pulse-text; 16 kHz vs 8 kHz training; with vs without Smart Turn pretraining; lexical feature classes removed one at a time.
- A confusion table for the arbiter.
- 20 annotated failure cases with audio and timestamps.

## 8. Deliverables

```
pulse-eot-hi/
  bharat_turn/        # build_dataset.py (from licensed sources), manifest.parquet, label_rules.md
  models/             # train.py, export_onnx.py, pulse_eot_hi_int8.onnx
  arbiter/            # backchannel vs barge-in classifier + eval
  adapters/           # eot-bench adapter, Pulse-endpointing baseline adapter
  sidecar/            # Pulse WS client -> {"type":"eot"} events; LiveKit/Pipecat plugin shim
  eval/               # run_all.sh -> results/*.json, pareto_*.png, report.md
  demo/               # browser mic or phone-dial demo
```

- **5-minute check for Smallest engineers:** `pip install -e . && bash eval/run_all.sh --quick`. This runs the model plus three baselines on the 400 Hindi eot-bench turns on CPU and prints the comparison table next to LiveKit's published Hindi numbers.
- A 4-page technical report and a 60-second demo video.

## 9. The 3-day cut (~25 solo hours)

**Day 1 (8 h): the yardstick first.**
- (2 h) Clone eot-bench; reproduce the Hindi VAD-only and SmartTurn v3.2 rows locally to within ±1 pp, which validates the harness.
- (1.5 h) Write the Pulse-endpointing baseline adapter: replay eot-bench `hi` audio through Pulse streaming with `eou_timeout_ms` swept over 6 values, log final arrival times, score.
- (3 h) Download the IndicVoices Hindi conversational subset; check the channel layout; run Pulse batch diarization + word timestamps on ~5 h; produce hold/shift/backchannel spans with the label rules; hand-check 100 spans.
- (1.5 h) Freeze Bharat-Turn-8k v0: speaker- and conversation-disjoint dev/test, plus μ-law and 5 dB MUSAN slices.

**Day 2 (9 h): the model.**
- (3 h) Fine-tune Smart Turn v3.2 (Whisper-tiny encoder) on IndicVoices Hindi spans + the Hindi portion of smart-turn-data v3.2, with 8 kHz and noise augmentation (rented GPU).
- (2 h) Lexical head: script-normalised trailing-token classes from Pulse partials (verb-aux, connective, filler, tag-question) into a small GBDT or MLP; fuse with the audio logit and silence duration in a hazard layer.
- (1 h) Export int8 ONNX; measure ms/hop.
- (3 h) Tune knobs on dev only, freeze, run the full sweep on both test sets, draw Pareto plots.

**Day 3 (8 h): barge-in, demo, write-up.**
- (3 h) Arbiter: a 300 ms-window classifier on the backchannel/barge-in spans (same encoder, separate head); evaluate FBR/MBR.
- (2 h) Sidecar on live Pulse WebSocket emitting `eot` events; record the demo.
- (3 h) Report, ablations (at least audio-only vs fused, 16k vs 8k), failure gallery, README.

**Deferred:** the full Bharat-Turn label set beyond ~5 h, AMR-NB codec, multi-turn context from agent text, other Indic languages, Hydra-style in-model integration.

**60-second demo shot list:**
1. (0-8 s) Split screen: the same Hinglish collections clip ("woh... matlab... pandrah tareekh tak kar dunga") played to a 600 ms silence-timer agent, which cuts in at "woh".
2. (8-20 s) The same clip through Pulse + Pulse-EoT-hi: a live `p_eot` trace stays low through the "matlab" and "tak" pauses and spikes 250 ms after "kar dunga".
3. (20-30 s) "Haan ji, theek hai." Fires in ~300 ms against the 1000 ms timer. A dead-air counter shows the difference.
4. (30-40 s) Agent reading terms; caller says "haan... achha" (agent keeps talking, labelled backchannel), then "nahi nahi ruko" (agent stops at ~240 ms).
5. (40-52 s) Pareto plot: our curve under every baseline on Hindi eot-bench and Bharat-Turn-8k; table row against the Pulse `eou_timeout_ms` sweep.
6. (52-60 s) `htop`: 8 MB int8 model, X ms/hop on one laptop core. End card: "Pulse 2.0 EoT, but for Bharat."

## 10. Stretch goals

- Extend to `north_indic` languages (Marathi, Bengali and Gujarati are already in Smart Turn data and Pulse).
- Distil the hazard model into Pulse's shared encoder as a LoRA head (the same pattern the Pulse 2.0 card describes for its emotion/gender heads), so EoT costs nothing extra server-side.
- A Hindi backchannel *generator* policy for Hydra ("ji", "haan ji" at the right moments), evaluated with Full-Duplex-Bench (Smallest already maintains a fork).
- Turn-aware early finalisation: hand `p_eot` to the LLM layer as a soft signal (the interface where this work meets Bolna-style speculation).
- Add a Hindi row to Smallest's public EoT benchmark page.

## Sources

- Smallest Pulse endpointing (defaults 600/800 ms, Pulse 2.0 ignores knobs): https://docs.smallest.ai/models/documentation/speech-to-text-pulse/features/endpointing.md
- Smallest `eou_timeout_ms` page (range 100 to 10,000 ms, recommendations): https://docs.smallest.ai/models/documentation/speech-to-text-pulse/features/end-of-utterance-timeout.md
- Pulse 2.0 model card (English-only, `eot_finalized`, Pulse EoT v1 eot-bench table, LoRA heads): https://docs.smallest.ai/model-cards/speech-to-text/pulse-2-0.md
- Smallest EoT benchmark page: https://docs.smallest.ai/models/speech-to-text/benchmarks/end-of-turn-detection.md
- Pulse STT changelog (Pulse 2.0 launch, `LANGUAGE_NOT_SUPPORTED_BY_MODEL`): https://docs.smallest.ai/models/changelog/pulse-stt
- Pulse model card (`hi` bilingual mixed-script, `north_indic`, Hindi telephony WER): https://docs.smallest.ai/model-cards/speech-to-text/pulse.md
- Pulse VAD events: https://docs.smallest.ai/models/speech-to-text/features/vad-events.md
- Hydra model card (English only, endpoint, barge-in): https://docs.smallest.ai/model-cards/speech-to-speech/hydra.md
- Hydra turn detection & barge-in ("VAD is conservative but not perfect"): https://docs.smallest.ai/waves/documentation/speech-to-speech-hydra/turn-detection-barge-in
- Atoms speech settings (Smart Turn off by default, English/Hindi only, 3 s wait, VAD release 0.4 s, backoff timer): https://docs.smallest.ai/voice-agents/build/speech-settings.md
- Smallest GitHub org (Full-Duplex-Bench fork): https://github.com/smallest-inc
- Series A, Inc42 (1 Aug 2026, $13M, ~60 staff, customers): https://inc42.com/buzz/smallest-ai-bags-13-mn-to-build-next-gen-voice-ai-models/
- Seed, SiliconANGLE (Oct 2025, $8M): https://siliconangle.com/2025/10/09/exclusive-voice-ai-developer-smallest-ai-nabs-8m-investment
- Seed, Entrepreneur India (founders, IIT Guwahati, Bosch): https://india.entrepreneur.com/news-and-trends/sierra-ventures-leads-usd-8-mn-round-in-voice-ai-firm/498804
- LiveKit eot-bench (metrics, adapter API, Apache-2.0): https://github.com/livekit/eot-bench
- eot-bench Hindi comparison report (all Hindi baseline numbers above): https://github.com/livekit/eot-bench/blob/main/output/livekit__eot-bench-data__validation__min_silence_100ms/hi/comparison/report.md
- eot-bench data (CC BY 4.0, `hi` 400 rows): https://huggingface.co/datasets/livekit/eot-bench-data
- LiveKit turn detector docs (TurnDetector v1 and v1-mini are audio-based, 14 languages incl. Hindi, open weights, v1-mini on CPU; the text MultilingualModel, Qwen2.5-0.5B, is deprecated; LiveKit Model License): https://docs.livekit.io/agents/logic/turns/turn-detector
- Smart Turn v3.2 README (Whisper Tiny, 8M params, 23 languages incl. Hindi/Marathi/Bengali, BSD-2): https://github.com/pipecat-ai/smart-turn
- Smart Turn v3.2 training data (CC BY 4.0): https://huggingface.co/datasets/pipecat-ai/smart-turn-data-v3.2-train
- Smallest docs changelog (Pulse streaming $0.006/min Standard plan, 28 May 2026; endpointing default ON, 12 Jul 2026): docs.smallest.ai changelog
- TurnBench, arXiv:2608.25218: https://arxiv.org/abs/2608.25218 ; MultiPartyTurnBench, arXiv:2610.20172: https://arxiv.org/abs/2610.20172
- Community Hinglish turn detector (MIT): https://huggingface.co/deveshu/hinglish-turn-detector
- Ekstedt & Skantze, "Voice Activity Projection: Self-supervised Learning of Turn-taking Events", Interspeech 2022: https://arxiv.org/abs/2205.09812 ; code (MIT): https://github.com/ErikEkstedt/VoiceActivityProjection
- Inoue et al., "Multilingual Turn-taking Prediction Using Voice Activity Projection", LREC-COLING 2024: https://arxiv.org/abs/2403.06487
- Lin et al., "Full-Duplex-Bench", 2025: https://arxiv.org/abs/2503.04721
- Stivers et al., "Universals and cultural variation in turn-taking in conversation", PNAS 106(26), 2009 (avoidance of overlap and minimisation of gap across 10 languages; language means within 250 ms of the cross-language mean): https://hraf.yale.edu/documents/1088
- IndicVoices paper (8 kHz telephone-bridge conversations, role-play scenarios), Findings of ACL 2024: https://arxiv.org/abs/2403.01926 ; dataset (CC BY 4.0, gated): https://huggingface.co/datasets/ai4bharat/IndicVoices
- MUCS 2021 (OpenSLR 104, CC BY-SA 4.0): https://www.openslr.org/104/
- Gram Vaani 1111 h Hindi (OpenSLR 118, academic-only): https://www.openslr.org/118/
- MUSAN (OpenSLR 17, CC BY 4.0): https://www.openslr.org/17/
- CALLFRIEND Hindi (LDC96S52): https://catalog.ldc.upenn.edu/LDC96S52
- Silero VAD (MIT): https://github.com/snakers4/silero-vad ; pyannote.audio (MIT): https://github.com/pyannote/pyannote-audio

# pulse-eot-hi: what end-of-turn detection costs on Hindi phone speech, measured

I wanted to know how much a plain silence timer (Pulse's `eou_timeout_ms`, Atoms' VAD release) costs a Hindi voice agent, and whether the open learned turn detectors do better. This repo is the yardstick I built for that: a causal, CPU-only streaming replay of LiveKit's eot-bench `hi` set that reproduces the published Hindi rows, a real-VAD timer sweep, and an honest attempt at a Hindi model that so far does **not** beat Smart Turn v3.2 or the timer.

**How this was built.** I wrote the problem statement (problems/06-smallest.md), the metrics and the honesty rules. AI coding agents (Claude Code) wrote the code to that spec; I reviewed it, re-ran the numbers below on my machine, and can walk through any module.

## What holds today

1. **The yardstick reproduces LiveKit's Hindi rows.** Our metric code reproduces every published Hindi row from LiveKit's committed predictions (`tests/test_metrics_parity.py`), and our own CPU run of Smart Turn v3.2 fp32 through the streaming harness gives the published row exactly (42.6% / 6.5% / 639 ms on all 400 turns; `results/eot_bench_hi/clean/smart_turn_v32_fp32/summary.json`).
2. **What a silence timer costs on Hindi** (real Silero VAD timing, all 400 eot-bench `hi` turns, `results/eot_bench_hi/clean/deployed.csv`). A timer has no tuned knob, so these numbers are not affected by the test-set issue below.

   | Timer | False cut-offs on holds | Mean latency after true end |
   |---|---|---|
   | 400 ms | 27.5% | 379 ms |
   | 600 ms | 6.9% | 579 ms |
   | 800 ms | 2.9% | 779 ms |
   | 1000 ms | 0.9% | 979 ms |

   Going from 600 ms to 1000 ms cuts false cut-offs from 6.9% to 0.9% and adds 400 ms of dead air on every turn. The real Pulse `eou_timeout_ms` sweep is the same experiment against Pulse itself; it is written and costed (534 audio-minutes, about $2.13 at the $0.004/min list price) and PENDING an API key.
3. **Negative results on the open Hindi detectors.** The community Hinglish Smart Turn fine-tune (deveshu, 49.5% FCR@300, 667 ms Lat@5%) is no better than a VAD timer on real Hindi. Our own v0, trained only on Chirp3 TTS Hindi, collapses on human speech (696 ms Lat@5%). Our v0.1 (Smart Turn v3.2 plus a GBDT calibration layer) **matches Smart Turn v3.2, it does not beat it** (paired bootstrap below), and at knobs frozen on dev it is **worse than a 600 ms timer**: 10.3% false cut-offs at 1.11 s mean latency versus 6.9% at 0.58 s.

## Results tables

### eot-bench `hi`, eot-bench knob sweep (all 400 turns, historical)

These rows use eot-bench's definition (best knob triple on the test sweep), which is how every published row is computed, so they are comparable but **not deployable** numbers. The only deployable row for our model is the dev-frozen one in the last line. **eot-bench `hi` informed one design iteration** (v0 to v0.1, see "Test-set use").

| System | FCR@300 | FCR@600 | Lat@5% | Lat@10% |
|---|---|---|---|---|
| LiveKit Turn Detector v1 (published, cloud) | 10.5% | 2.0% | 454 ms | 306 ms |
| Soniox stt-rt-preview (published, cloud) | - | 3.4% | 553 ms | 553 ms |
| ultraVAD (published, open weights, 0.7B) | 39.1% | 6.5% | 616 ms | 509 ms |
| Gradium (published, cloud) | 49.5% | 5.8% | 618 ms | 530 ms |
| LiveKit v1-mini (published) | 27.4% | 5.4% | 624 ms | 505 ms |
| Pulse-EoT-hi v0.1 (ours, CPU) | 46.4% | 6.7% | 634 ms | 514 ms |
| SmartTurn v3.2 fp32, run by us on CPU | 42.6% | 6.5% | 639 ms | 532 ms |
| SmartTurn v3.2 int8 CPU build, run by us | 45.1% | 6.9% | 646 ms | 507 ms |
| deveshu/hinglish-turn-detector, run by us | 49.5% | 7.4% | 667 ms | 584 ms |
| Pulse-EoT-hi v0 (Hindi TTS-only training) | 48.9% | 7.2% | 696 ms | 568 ms |
| Silero VAD silence timer, run by us (real VAD) | 46.6% | 7.4% | 700 ms | 554 ms |
| VAD baseline, oracle silence (eot-bench) | 49.5% | 7.4% | 700 ms | 600 ms |
| *Pulse-EoT-hi v0.1 at dev-frozen knobs (th 0.85, delay 0.4 s, timeout 3.5 s): deployable* | *FCR 8.7% at mean latency 1234 ms* | | | |

Soniox has no FCR@300 point because none of its operating points reaches 300 ms mean latency. ultraVAD is a stronger open baseline than Smart Turn, but at 0.7B parameters it is far outside this project's 50 MB / 10 ms-per-hop CPU budget, so it is listed and not run. Full table with Lat@10%, Detect@5% and AUC: [results/report.md](results/report.md).

### eot-bench `hi` dev-test half (200 turns): where all iteration happens from now on

`python -m eval.rescore_devtest` rescored the saved predictions and LiveKit's committed ones on the dev-test half only ([results/eot_bench_hi_devtest/rescored.md](results/eot_bench_hi_devtest/rescored.md)). On these 200 turns: Smart Turn v3.2 fp32 (ours) 37.1% / 624 ms, v0.1 43.5% / 611 ms, LiveKit v1-mini 24.6% / 550 ms, Silero timer 48.4% / 741 ms (FCR@300 / Lat@5%). Paired bootstrap over the 200 turns (1000 resamples), v0.1 minus Smart Turn fp32, each at its own Lat@5% knobs: latency -12.5 ms (95% CI -30.0 to +6.5), FCR +0.40 pp (95% CI -1.38 to +2.12). The interval contains zero: **no detectable difference**.

### Other measurements (stated with their caveats)

- **Channel stress (evaluation only, all 400 turns, Lat@5% change vs clean):** mu-law 8 kHz: Smart Turn int8 -4 ms, v0.1 -5 ms, timer 0 ms. MUSAN noise at 5 dB SNR: Smart Turn int8 -6 ms, v0.1 +27 ms, timer +2 ms. Telephone band-limiting does not hurt these detectors on this set; noise hurts our fusion head. See `results/report.md`.
- **Efficiency, measured with x86 Python under emulation on an ARM laptop (Qualcomm ARMv8), one ORT thread** (`results/efficiency/pulse_eot_hi.json`): v0.1 costs 13.4 ms per 100 ms hop on average, p50 2.2 ms, p95 8.0 ms, **p99 401 ms** (the once-per-pause Smart Turn call, made synchronously), so about 7 calls per core. The process uses 622 MB RSS before call state and 113 MB more for 50 call states, against a budget of 300 MB per process. These numbers are emulated and not yet re-measured on native x86.
- **Barge-in arbiter: not a result.** `arbiter/` is a speech-onset vs noise classifier trained and tested on single-speaker Chirp3 TTS clips plus MUSAN. Its test set has 1464 "barge-ins", 134 noise clips and only 15 backchannels, of which it recognises 2 (backchannel recall 2/15). It has no lexical input, so it cannot tell "haan" from "nahi ruko". A real evaluation needs two-party Hindi audio and Pulse partials. No arbiter score is quoted anywhere as a result.

### Honesty notes

1. **Test-set use.** v0 was scored on all 400 eot-bench `hi` turns, and three design changes (stack the Smart Turn probability, remove the VAD gate, add raw VAD features) were decided after seeing that score. Then v0.1, nine ablations, three channel conditions, the deployed sweep and the slice table were all scored on the same 400. Weights and knobs were never fit on eot-bench, but the 634 ms is a once-contaminated number. From 2026-10-20 the 400 turns are split by a salted hash of the turn id (`eval/split.py`, ids in `eval/eot_bench_hi_split.csv`) into a **dev-test** half for iteration and a **sealed 200-turn half** that nothing is computed on until the one final frozen-knob v1 run. `run_bench` refuses the sealed half without `--final` and logs every such run in `results/SEALED_LEDGER.md` (empty today). The sealed half is "sealed from now on", not "never seen": it was part of the 400 that informed v0 to v0.1.
2. **No real Hindi speech in training.** Every Hindi training and dev clip is Google Chirp3 TTS (`data/smart_turn_hi` 2196 clips, `data/smart_turn_hi_test` 1165 clips, all `synthetic=True`). The real-human clips are English and Spanish. Hold/EoT pause lengths for dev episodes come from eot-bench **English** statistics. That is why v0 failed and why the GPU step below starts with real Hindi data.
3. **ASR.** Test-time text comes from Vosk small-hi replayed with logged arrival times, not oracle words. Vosk's Hindi transcripts on this audio are poor (`results/failures.md`), and the cache was built with parallel processes on an emulated CPU, which delays arrival times by a median of tens of ms. The lexical branch is effectively unproven.
4. **The demo trace `demo/hi__4344.png` is a failure case** (the model false-fires at the same digit read-out hold as the 600 ms timer). Do not use it as a showcase.

## The GPU path (real Hindi data, one command each)

All three are written and dry-run on CPU; none has been run on a GPU or against a live API.

| Step | Command | Needs | Verified cost |
|---|---|---|---|
| 1. Real-Hindi pause episodes from IndicVoices extempore audio | `python -m bharat_turn.indicvoices_episodes --src data/indicvoices/hindi --inspect`, then without `--inspect` | IndicVoices terms accepted on HF (PENDING) | $0 (CPU, Silero VAD) |
| 2. Smart Turn v3.2 Hindi fine-tune | `modal run models/gpu/modal_app.py` (CPU check: `python -m models.gpu.finetune --dry-run --fake-indicvoices`) | Modal account, step 1 output | estimate $0.26 (0.18 h on one L4); hard ceiling $2.86 at the 2 h timeout (`results/gpu_cost_estimate.json`) |
| 3. Pulse `eou_timeout_ms` sweep {300..1200} | `python -m adapters.pulse_endpointing --estimate-cost`, then `SMALLEST_API_KEY=... python -m adapters.pulse_endpointing` | Smallest API key (PENDING) | $2.13 at $0.004/min (533.7 audio-minutes) |

Step 2 is a real training loop: Smart Turn v3.2 rebuilt in PyTorch with the published fp32 ONNX weights (max |dp| 1.2e-7 against the ONNX, `models/gpu/smart_turn_torch.py`), BCE fine-tune with mu-law and MUSAN augmentation, early stopping on IndicVoices dev Lat@5% computed with the same eot-bench metric code, and fp32 + int8 ONNX export with a parity check. The CPU dry run trains, drives the loss of 16 real-Hindi-style windows from 0.114 to 0.005 in 30 steps (an overfit check, on a fake IndicVoices shard made from TTS clips: a pipeline test, not a result), and exports. The cost estimate uses measured FLOPs (7.2 GFLOP per forward) and measured log-mel cost, with an assumed 15% of the L4's dense bf16 peak; the run itself writes its real wall time and dollar cost to `run_summary.json`.

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"        # Linux/mac: .venv/bin/pip; GPU path: ".[dev,gpu]"
# one-time downloads (public, about 1 GB of models + the 133 MB Hindi test parquet; see data/README.md)
bash eval/run_all.sh --quick        # Windows: .\run.ps1 quick
python -m pytest -q tests           # metric parity with eot-bench, causality, split guard, labels, ONNX parity, GPU-path parity, sidecar
```

`--quick` runs the Silero timer, Smart Turn v3.2 int8 and Pulse-EoT-hi v0.1 on the 200 dev-test turns with 4 worker processes, then rebuilds the report. Measured on this laptop on 2026-10-20 (ASR cache already built): 112 s, and the streaming re-run reproduced the rescored dev-test rows exactly (timer 48.4% / 741 ms, Smart Turn int8 37.1% / 602 ms, v0.1 43.5% / 611 ms; `results/eot_bench_hi_devtest/clean/`). The full run is `bash eval/run_all.sh` (also dev-test only).

## Repo map

```
core/          audio (20 ms PCM16 frames, mu-law, MUSAN mixing), Silero VAD stream, Vosk streaming ASR replay,
               script-normalised Hindi lexical classes, causal prosody, Whisper log-mel + Smart Turn / deveshu ONNX
eval/          harness.py (streaming replay), metrics.py (eot-bench-identical sweep), split.py (dev-test / sealed),
               deployed.py (VAD-timed policy), run_bench.py, rescore_devtest.py, conditions.py, bench_cpu.py, report.py
adapters/      detector protocol, baselines (Silero timer, Smart Turn, deveshu), pulse_endpointing.py (Pulse sweep: PENDING key)
models/        pulse_eot_hi.py (runtime), train.py (v0), train_stacked.py (v0.1), export_onnx.py, artifacts/*.onnx,
               gpu/ (smart_turn_torch.py, finetune.py, modal_app.py, cost_estimate.py: PENDING GPU run)
arbiter/       speech-onset vs noise classifier on TTS proxy data (not a barge-in result, see above)
bharat_turn/   labels.py (tested label rules), indicvoices_episodes.py (PENDING data), fake_indicvoices.py (tests only)
sidecar/       Pulse wire protocol, eot sidecar (Pulse-2.0-shaped events), local mock Pulse server
results/       report.md, failures.md, eot_bench_hi_devtest/rescored.md, gpu_cost_estimate.json, per-system summary.json
```

Also see [STATUS.md](STATUS.md) (every critique finding and its state), [UNDERSTAND.md](UNDERSTAND.md) (what to be able to defend), [DATA.md](DATA.md) (datasets, models, licences) and [data/README.md](data/README.md) (download steps). Licence: MIT ([LICENSE](LICENSE)); third-party models and data keep their own licences (DATA.md).

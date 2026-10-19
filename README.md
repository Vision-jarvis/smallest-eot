# pulse-eot-hi: streaming Hindi/Hinglish end-of-turn and barge-in arbitration (CPU)

This repo is a causal, CPU-only turn-taking layer for a Pulse-based Hindi/Hinglish voice agent. Every 100 ms it says whether the caller has finished speaking (`p_eot`). While the agent talks, it says whether user speech is a backchannel, a barge-in, or noise. It also contains the yardstick to prove it: a streaming replay harness whose metrics reproduce LiveKit's eot-bench numbers exactly.

Status in one line: the yardstick, the CPU baselines and a first CPU-trained model all run with real numbers. The model beats Smart Turn v3.2 by a few ms on Lat@5%, but it does **not** yet reach the 3-day bar (beat LiveKit v1-mini). Anything that needs the Smallest API key, IndicVoices access or a GPU is built and marked PENDING (see [STATUS.md](STATUS.md)).

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/pip install -e ".[dev]"        # Linux/mac: .venv/bin/pip
# one-time downloads (public, about 1 GB of models + the 133 MB Hindi test parquet; see data/README.md)
bash eval/run_all.sh --quick        # Windows: .\run.ps1 quick
python -m pytest -q tests           # 23 tests: metric parity with eot-bench, causality, labels, ONNX parity, sidecar
```

`--quick` runs Pulse-EoT-hi plus the Silero timer and Smart Turn v3.2 on the 400 Hindi eot-bench turns on CPU. It then prints the comparison table next to LiveKit's published Hindi rows. Measured wall time on this laptop (18 cores, Vosk ASR cache already built) was **4 min 26 s**. The first run adds about 3 minutes to build the streaming ASR log. The full run is `bash eval/run_all.sh`.

## Results so far (all produced in this repo; full tables in [results/report.md](results/report.md))

### eot-bench `hi` (400 turns), eot-bench knob sweep

This is the same definition LiveKit and Smallest use.

| System | FCR@300 | FCR@600 | Lat@5% | Lat@10% |
|---|---|---|---|---|
| LiveKit Turn Detector v1 (published, cloud) | 10.5% | 2.0% | 454 ms | 306 ms |
| LiveKit v1-mini (published) | 27.4% | 5.4% | 624 ms | 505 ms |
| **Pulse-EoT-hi v0.1 (ours, CPU)** | **46.4%** | **6.7%** | **634 ms** | **514 ms** |
| SmartTurn v3.2 fp32, run by us on CPU | 42.6% | 6.5% | 639 ms | 532 ms |
| SmartTurn v3.2 int8 CPU build, run by us | 45.1% | 6.9% | 646 ms | 507 ms |
| deveshu/hinglish-turn-detector, run by us | 49.5% | 7.4% | 667 ms | 584 ms |
| Pulse-EoT-hi v0 (Hindi TTS-only training) | 48.9% | 7.2% | 696 ms | 568 ms |
| Silero VAD silence timer, run by us (real VAD) | 46.6% | 7.4% | 700 ms | 554 ms |
| VAD baseline, oracle silence (eot-bench) | 49.5% | 7.4% | 700 ms | 600 ms |

- **Harness validation:** our metric code reproduces every published Hindi row exactly from LiveKit's committed predictions (`tests/test_metrics_parity.py`). Our own end-to-end CPU run of SmartTurn v3.2 fp32 through our streaming harness gives 42.6% / 6.5% / 639 ms, identical to the published row. That is the Day-1 "reproduce within ±1 pp" check.
- **Points matrix (rows measurable today):**
  - Row 1 (Lat@5%): 5.3 / 20.
  - Row 2 (FCR@300): 0.8 / 10.
  - Rows 3, 4, 6 are defined on Bharat-Turn-8k and are PENDING (IndicVoices access).
- **Dev-frozen knobs (honesty rule 3):** the triple frozen on dev (th 0.85, delay 0.4 s, timeout 3.5 s), applied unchanged to the test set, gives FCR 8.7% at 1.23 s mean latency. Calibration does **not** transfer from the Smart Turn dev clips to real calls. The best test operating point (634 ms) is the eot-bench-comparable number, not a frozen-knob number.
- **Deployed (VAD-timed) curve:**
  - A real-VAD 600 ms silence timer gives 6.9% FCR at 579 ms.
  - v0.1 at dev-frozen knobs gives 10.3% at 1.11 s, worse than the timer.
  - v0.1 at the test Lat@5% knobs gives 4.7% at 625 ms. That row is oracle knobs, shown for reference only.
- **Channel stress (evaluation only):**
  - μ-law 8 kHz: v0.1 629 ms vs SmartTurn 642 ms.
  - MUSAN 5 dB: v0.1 661 ms vs SmartTurn 640 ms.
  - ΔLat@5% at 5 dB on eot-bench: v0.1 +27 ms, SmartTurn -6 ms.
- **Efficiency (one ORT thread):**
  - v0.1: 2.2 / 8.0 / 401 ms per hop at p50 / p95 / p99. The p99 is the once-per-pause Smart Turn call. Model files total 10.6 MB; 50 call states take 113 MB above a 622 MB process baseline; 7 calls per core.
  - v0 (no Smart Turn): p95 3.5 ms, 2.3 MB, 44 calls per core.
- **Barge-in arbiter, proxy set (not Hindi two-party audio):** FBR 18.1%, MBR 0.8%, arbiter score 0.905, decision at 300 ms. Only 15 backchannels are in the test set, so this number is weak.

### Honesty notes
1. **Test set use:** eot-bench `hi` was never used to fit weights or knobs. But v0.1's design was changed after seeing v0's eot-bench result. v0 was trained only on TTS clips and gated scores on the VAD. Both versions are reported.
2. **ASR:** test-time text comes from Vosk small-hi replayed with logged arrival times, not from oracle words.
3. **Dev data is synthetic:** dev episodes are Smart Turn test-split clips (Hindi = Chirp3 TTS) given English-calibrated pause lengths. That is why dev numbers (Lat@5% about 0.57 s, AUC 0.98) are far rosier than test.

## What is in the box (repo map, follows the statement's Deliverables)

```
core/          audio (20 ms PCM16 frames, mu-law, MUSAN mixing), Silero VAD stream, Vosk streaming ASR replay,
               script-normalised Hindi lexical classes, causal prosody, Whisper log-mel + Smart Turn / deveshu ONNX
eval/          harness.py (streaming replay), metrics.py (eot-bench-identical sweep), deployed.py (VAD-timed policy),
               run_bench.py, run_deployed.py, conditions.py (mu-law / SNR slices), bench_cpu.py, report.py, run_all.sh
adapters/      base.py (detector protocol + eot-bench predict_batch adapter), baselines.py (Silero timer, Smart Turn,
               deveshu), pulse_endpointing.py (Pulse eou_timeout_ms sweep: PENDING key)
models/        pulse_eot_hi.py (runtime), episodes.py, lexical.py, train.py (v0), train_stacked.py (v0.1),
               export_onnx.py, artifacts/*.onnx, gpu/ (Smart Turn Hindi fine-tune config + runner: PENDING GPU)
arbiter/       backchannel / barge-in / noise classifier, proxy data, train + eval
bharat_turn/   labels.py (hold/shift/backchannel/barge-in rules, tested), label_rules.md, build_dataset.py (PENDING data)
sidecar/       Pulse wire protocol, eot sidecar (Pulse-2.0-shaped events), local mock Pulse server
demo/          trace_demo.py: p_eot trace vs a 600 ms timer for any turn or wav
results/       report.md, failures.md (20 annotated failures), pareto_eot_bench_hi.png, per-system summary.json
```

Also see [UNDERSTAND.md](UNDERSTAND.md) (the 13 things to be able to defend) and [data/README.md](data/README.md) (sources, licences, manual steps).

# Status against the 3-day cut (Section 9)

Last updated 2026-10-19. Every number below is in `results/`.

## Day 1: the yardstick

| Item | State | Notes / command |
|---|---|---|
| Clone eot-bench and reproduce the Hindi VAD-only and SmartTurn v3.2 rows within ±1 pp | **DONE** | VAD baseline 49.5/7.4/700 and SmartTurn 42.6/6.5/639 are reproduced exactly. That covers both metric parity on LiveKit's predictions and our own end-to-end CPU run of the fp32 model. Commands: `pytest tests/test_metrics_parity.py` and `python -m eval.run_bench --detector smart_turn_v32_fp32` |
| Pulse-endpointing baseline (`eou_timeout_ms` ∈ {300..1200}) | **PENDING (API key)** | `adapters/pulse_endpointing.py` is complete: real-time streamer, logs with arrival times, scorer. It was smoke-tested against `sidecar/mock_pulse.py` (protocol only, not Pulse numbers). Run: `SMALLEST_API_KEY=... python -m adapters.pulse_endpointing --sweep 300 400 600 800 1000 1200`. Stand-in today: real-Silero timer sweep, in `results/eot_bench_hi/clean/deployed.csv` |
| IndicVoices Hindi conversational subset, channel layout, diarisation, label spans, hand-check 100 | **PENDING (gated dataset + key or HF token)** | Label rules are implemented and tested (`bharat_turn/labels.py`, `tests/test_labels.py`). Download steps are in `data/README.md`, then `python -m bharat_turn.build_dataset --indicvoices-dir data/indicvoices/hindi`. Steps 1 to 3 (schema, channel layout, diariser) must be wired to the real files once seen |
| Freeze Bharat-Turn-8k v0 (disjoint dev/test, μ-law, 5 dB MUSAN slices) | **PARTIAL** | Split function (speaker+conversation hash) and channel conditions are done. The μ-law and MUSAN 10/5 dB slices run today on eot-bench `hi` as stress tests. `manifest.parquet` is PENDING the data |

## Day 2: the model

| Item | State | Notes / command |
|---|---|---|
| Fine-tune Smart Turn v3.2 on IndicVoices + Hindi Smart Turn data, 8 kHz + noise aug | **PENDING (GPU + IndicVoices)** | Config `models/gpu/finetune_smart_turn_hi.yaml`, runner `bash models/gpu/run_gpu.sh`. Data prep runs on CPU today: `python -m models.gpu.finetune --prepare-only` |
| Lexical head + fusion with audio logit and silence in a hazard layer | **DONE (CPU v0 / v0.1)** | v0: `python -m models.train`. v0.1 (frozen Smart Turn feature): `python -m models.train_stacked`. Real results are in README. The bar FCR@300 < 27.4% and Lat@5% < 624 ms is **not met** (46.4%, 634 ms) |
| Export ONNX, measure ms/hop | **DONE** | `models/export_onnx.py` (parity-checked). `python -m eval.bench_cpu`: p95 8.0 ms/hop (v0.1), 3.5 ms (v0). The fusion head is float (about 100 KB of trees); int8 applies to the GPU encoder |
| Tune knobs on dev only, freeze, full sweep on test, Pareto plots | **DONE** | `summary.json["frozen_knobs"]` and `["frozen_result"]`, plus `results/pareto_eot_bench_hi.png`. The frozen knobs transfer poorly (see README). The Bharat-Turn test sweep is PENDING the data |

## Day 3: barge-in, demo, write-up

| Item | State | Notes / command |
|---|---|---|
| Arbiter (300 ms window), FBR/MBR | **PARTIAL** | `python -m arbiter.train_eval` on a proxy overlap set: FBR 18.1%, MBR 0.8%, n_backchannel = 15. A real evaluation needs the Bharat-Turn overlap set |
| Sidecar on live Pulse emitting `eot` events | **PARTIAL (API key)** | `sidecar/eot_sidecar.py` is unit-tested and runs against the mock: `python -m sidecar.mock_pulse` then `python -m sidecar.eot_sidecar --wav x.wav --base ws://127.0.0.1:8765`. For live Pulse, set `SMALLEST_API_KEY`. LiveKit/Pipecat plugin shim not written yet |
| Demo | **PARTIAL** | `python -m demo.trace_demo --turn hi__4344` writes a p_eot vs 600 ms-timer plot. A browser-mic or phone demo and the video need live Pulse |
| Report, ablations (audio-only vs fused, 16k vs 8k), failure gallery, README | **PARTIAL** | Done: `results/report.md` (ablations incl. lexical classes removed one at a time, audio-only, no lexical head, Smart Turn only), `results/failures.md` (20 cases), README. The 8 kHz-*training* ablation path exists (`python -m models.train --channel mulaw8k`) but was not run. Only 8 kHz *evaluation* was run. The 4-page technical report is not written |

## What the student must do
1. Accept the IndicVoices terms on Hugging Face and download `hindi/` (data/README.md, step 1).
2. Get a Smallest API key, then run the Pulse sweep and the live sidecar (commands above). Budget per the statement: under $15.
3. Rent one 24 GB GPU (about $50) for the Smart Turn Hindi fine-tune.
4. Optional: let `python data/_get_st_hi.py smart-turn-data-v3.2-train 83` finish the remaining 68 shards (about 34 GB streamed). Only 15 of 83 were used here.

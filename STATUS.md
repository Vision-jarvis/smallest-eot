# Status

Last updated 2026-10-20, after the independent critique (`critiques/06-smallest.md`). Legend: DONE = built and verified on this machine; PARTIAL = built, part of it blocked; PENDING = blocked on a key, data access or a GPU, with the exact command to finish.

## Critique findings

| ID | State | What was done / why deferred |
|---|---|---|
| B1 | FIXED | README, UNDERSTAND and the problem statement's section 11 now tell the yardstick + timer-cost + negative-result story. No "beats / dominates / 8 MB" anywhere. v0.1 is described as matching Smart Turn v3.2 (dev-test paired bootstrap: -12.5 ms, 95% CI -30.0 to +6.5) and as worse than a 600 ms timer at dev-frozen knobs |
| B2 | FIXED | The arbiter score is removed from README, report and pitch. `results/report.md` labels it "speech-onset vs noise classifier on TTS clips, backchannel recall 2/15". A real arbiter needs two-party Hindi audio and Pulse partials (not a Nov 4 deliverable) |
| M1 | FIXED | `eval/split.py`: 200 dev-test / 200 sealed turns by salted SHA-256 rank, ids committed in `eval/eot_bench_hi_split.csv`. `run_bench` defaults to dev-test and refuses the sealed half without `--final` (logged in `results/SEALED_LEDGER.md`, no entries yet). `eval/rescore_devtest.py` rescores saved and published predictions on dev-test only. What was looked at is written in README "Test-set use" |
| M2 | PARTIAL | README/report now give mean 13.4 / p95 8.0 / p99 401 ms per hop, 622 MB RSS, and say "x86 Python emulated on ARM". DEFERRED: the native x86 number (needs an x86 box; run `python -m eval.bench_cpu --detector pulse_eot_hi` on the GPU box's CPU) and moving the Smart Turn call to a worker thread (planned with the v1 encoder swap, since the encoder changes) |
| M3 | PARTIAL | Stated plainly ("zero real Hindi in training"). The real-Hindi pause builder exists (`bharat_turn/indicvoices_episodes.py`). DEFERRED: running it (needs the IndicVoices terms) |
| M4 | FIXED | The stub is replaced by a training loop that trains (`models/gpu/finetune.py`, CPU dry run + overfit check pass) and a one-command Modal runner. README no longer says the GPU part is "built"; it says "written and dry-run on CPU, not run on a GPU" |
| M5 | DEFERRED | The `has_text` routing fix and a v0.1-without-lexhead ablation change the model, so they belong to the v1 refit on IndicVoices dev with Pulse partials (needs the Smallest key; Pulse partials replay of the 400 turns is about 89 audio-minutes, $0.36). Re-scoring a patched v0.1 now would spend more looks at eot-bench for a model that is about to be replaced |
| M6 | FIXED | The slice table now shows every system at two common operating points (own Lat@5% knobs, own best knobs within 600 ms) on the dev-test half, with n and 95% Wilson intervals. The mixed-knob table is gone |
| M7 | PARTIAL | MOCK csv removed from `results/` (mock output now goes to `data/cache/`). `python -m adapters.pulse_endpointing --estimate-cost` gives 533.7 audio-minutes, $2.13 at $0.004/min. PENDING the Smallest key for the run itself |
| M8 | DEFERRED | Superseded by Pulse partials (key). The cache builder now caps workers at 4 and the README states the arrival-time inflation. If the key is late: delete `data/cache/asr_eot_bench_hi_clean.parquet` and rebuild with `workers=1` (about 45 CPU-minutes), then re-run on dev-test only |
| M9 | FIXED | README marks `demo/hi__4344.png` as a failure case. Day-1 picture: `python -m demo.timer_cost` writes `results/timer_cost_hi.png`. The Devanagari font in the old trace is not fixed (the trace is not to be used) |
| M10 | FIXED | Framing in UNDERSTAND section 13 and the statement section 11: Atoms already ships Smart Turn for Hindi (off by default); this repo measures what that class of model and the timer cost on Hindi |
| m1 | FIXED | Soniox (553 ms), ultraVAD (39.1% / 6.5% / 616 ms) and Gradium (49.5% / 5.8% / 618 ms) published rows added to README and report, with one sentence on why ultraVAD (0.7B) is out of budget |
| m2 | DEFERRED | The stacked head passed parity on 5000 dev rows (1.7e-7). It will be refit on IndicVoices dev and exported with the fixed exporter in the v1 step; re-exporting now would need a re-score on eot-bench for no expected change |
| m3 | PARTIAL | Traces and MOCK csv untracked; LICENSE (MIT) and DATA.md added; mel filterbank now computed in `core/encoders.py` (identical to transformers', tested); logs behind this round's numbers in `results/logs/`. DEFERRED: replacing `arbiter.pkl` (arbiter is off the Nov 4 path) |
| m4 | FIXED (doc) | `agent_text` is `None` on eot-bench hi and unused by the model; README no longer lists it as an input. Use it on two-party data once it exists |
| m5 | FIXED | UNDERSTAND section 3 cites `eot_harness/io.py` line 11 (`DEFAULT_TRANSCRIPT_LAG = 0.5`) and line 41 (eot-bench commit 9ee21b5) |
| m6 | No change needed | Statement facts verified by the critic |
| m7 | FIXED | Hook uses the measured number (6.9% at 600 ms), and says "cuts in", not "hangs up" |
| m8 | FIXED | The dev-frozen (deployable) row sits in the main README table with a note |
| Campaign | DONE | "How this was built" paragraph in README; MIT LICENSE; DATA.md; section 11 rewritten (4-line DM, "Day X/30 \| Smallest.ai \| build N of 11"); no em dashes; no tooling chatter in docs |

## 3-day cut (statement section 9)

| Item | State | Notes / command |
|---|---|---|
| Reproduce the Hindi VAD-only and SmartTurn v3.2 rows within 1 pp | DONE | Exact match. `pytest tests/test_metrics_parity.py`; `python -m eval.run_bench --detector smart_turn_v32_fp32 --half all --final` would re-run it on all 400 (logged; not needed) |
| Pulse `eou_timeout_ms` sweep {300..1200} | PENDING (key) | `python -m adapters.pulse_endpointing --estimate-cost` ($2.13), then `SMALLEST_API_KEY=... python -m adapters.pulse_endpointing --sweep 300 400 600 800 1000 1200`. Stand-in today: the real-VAD timer sweep in `results/eot_bench_hi/clean/deployed.csv` |
| Real-Hindi pause episodes (IndicVoices extempore) | PENDING (IndicVoices terms) | `huggingface-cli download ai4bharat/IndicVoices --repo-type dataset --include "hindi/valid-*" "hindi/train-0000*" --local-dir data/indicvoices`, then `python -m bharat_turn.indicvoices_episodes --src data/indicvoices/hindi --inspect` and the same without `--inspect`. Dry-run on a fake shard: `python -m bharat_turn.fake_indicvoices && python -m bharat_turn.indicvoices_episodes --src data/_fake_indicvoices/hindi --out data/_fake_indicvoices/pauses` |
| Bharat-Turn-8k (two-party, diarised) | PENDING (data + diariser) | Label rules done and tested (`bharat_turn/labels.py`). Not on the Nov 4 path |
| Fine-tune Smart Turn v3.2 on real Hindi, mu-law + MUSAN aug | PENDING (GPU run) | CPU dry run passes: `python -m models.gpu.finetune --dry-run --fake-indicvoices`. GPU: `modal run models/gpu/modal_app.py` (or `bash models/gpu/run_gpu.sh`). Estimate $0.26, ceiling $2.86 (`results/gpu_cost_estimate.json`) |
| Lexical head + fusion | DONE (CPU v0 / v0.1) | v0.1 does not beat Smart Turn v3.2; the bar (FCR@300 < 27.4%, Lat@5% < 624 ms) is not met |
| Export ONNX, measure ms/hop | DONE (emulated) | Native x86 measurement PENDING an x86 box |
| Tune knobs on dev only, freeze, run test once | PARTIAL | Frozen knobs exist and transfer poorly. The one sealed-half run waits for v1 |
| Arbiter | NOT A RESULT | TTS proxy only (see B2) |
| Sidecar on live Pulse | PARTIAL (key) | Runs against the local mock: `python -m sidecar.mock_pulse` then `python -m sidecar.eot_sidecar --wav x.wav --base ws://127.0.0.1:8765` |

## What the student must do

1. Accept the IndicVoices terms on Hugging Face, download a few Hindi shards (command above), run the pause builder with `--inspect` once and check the task names and audio paths it prints (the `--task-regex` and `--recording-regex` defaults may need adjusting).
2. Create a Modal account (`pip install modal && modal setup`), then `modal run models/gpu/modal_app.py`. Delete the data volume afterwards (`modal volume delete pulse-eot-hi-data`; IndicVoices audio is gated).
3. With the Smallest key: run the Pulse sweep ($2.13), then the Pulse partials replay for the v1 refit (about $0.36).
4. Run the sealed eot-bench half exactly once with the final v1 and frozen knobs: `python -m eval.run_bench --detector pulse_eot_hi --half sealed --final`.

## Smoke test

`bash eval/run_all.sh --quick` (Windows: `.\run.ps1 quick`) passed on 2026-10-20 in 112 s on the 200 dev-test turns; `python -m pytest -q tests` passes (29 tests).

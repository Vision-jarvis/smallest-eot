# Status

Last updated 2026-10-24 (live Pulse sweep and partials), first written after the independent critique (`critiques/06-smallest.md`). Legend: DONE = built and verified on this machine; PARTIAL = built, part of it blocked; PENDING = blocked on a key, data access or a GPU, with the exact command to finish.

## Critique findings

| ID | State | What was done / why deferred |
|---|---|---|
| B1 | FIXED | README, UNDERSTAND and the problem statement's section 11 now tell the yardstick + timer-cost + negative-result story. No "beats / dominates / 8 MB" anywhere. v0.1 is described as matching Smart Turn v3.2 (dev-test paired bootstrap: -12.5 ms, 95% CI -30.0 to +6.5) and as worse than a 600 ms timer at dev-frozen knobs |
| B2 | FIXED | The arbiter score is removed from README, report and pitch. `results/report.md` labels it "speech-onset vs noise classifier on TTS clips, backchannel recall 2/15". A real arbiter needs two-party Hindi audio and Pulse partials (not a Nov 4 deliverable) |
| M1 | FIXED | `eval/split.py`: 200 dev-test / 200 sealed turns by salted SHA-256 rank, ids committed in `eval/eot_bench_hi_split.csv`. `run_bench` defaults to dev-test and refuses the sealed half without `--final` (logged in `results/SEALED_LEDGER.md`, no entries yet). `eval/rescore_devtest.py` rescores saved and published predictions on dev-test only. What was looked at is written in README "Test-set use" |
| M2 | PARTIAL | README/report now give mean 13.4 / p95 8.0 / p99 401 ms per hop, 622 MB RSS, and say "x86 Python emulated on ARM". DEFERRED: the native x86 number (needs an x86 box; run `python -m eval.bench_cpu --detector pulse_eot_hi` on the GPU box's CPU) and moving the Smart Turn call to a worker thread (planned with the v1 encoder swap, since the encoder changes) |
| M3 | PARTIAL | Stated plainly ("zero real Hindi in training"). The real-Hindi pause builder exists (`bharat_turn/indicvoices_episodes.py`). DEFERRED: running it (needs the IndicVoices terms) |
| M4 | FIXED | The stub is replaced by a training loop that trains (`models/gpu/finetune.py`, CPU dry run + overfit check pass) and a one-command Modal runner. README no longer says the GPU part is "built"; it says "written and dry-run on CPU, not run on a GPU" |
| M5 | DEFERRED (inputs half ready) | The `has_text` routing fix and a v0.1-without-lexhead ablation belong to the v1 refit. Pulse partials for all 400 eot-bench turns are now cached (2026-10-24, see M8), but the refit itself was not run: it needs Pulse partials for the *training/dev* clips, not eot-bench. What it needs: (1) stream IndicVoices dev (816 windows, about 4.3 s each, about 100 audio-minutes with the 3 s pad, about $0.60) and the fusion-train clips through `adapters/pulse_endpointing.run_partials`; the IndicVoices train pauses (13,542 windows, about 930 minutes, about $5.60) are over this round's budget; (2) point `models/train.py` featurisation at a Pulse cache instead of `asr_cache.build` (Vosk); (3) a dev set with more than 13 end-of-turn windows, or Lat@5% knob selection stays noise (STATUS fine-tune row) |
| M6 | FIXED | The slice table now shows every system at two common operating points (own Lat@5% knobs, own best knobs within 600 ms) on the dev-test half, with n and 95% Wilson intervals. The mixed-knob table is gone |
| M7 | FIXED (REAL) | Ran 2026-10-24 on the live API, dev-test half (200 turns x 6 timeouts, 254.8 audio-minutes, $1.53 at $0.006/min): `results/eot_bench_hi_devtest/clean/pulse_endpointing/eou_sweep.csv`, table in `results/eot_bench_hi_devtest/pulse_vs_timer.md` and README. Pulse at 600 ms: 2.0% FCR, 11.5% premature turns, 931 ms mean latency. Price constant updated to $0.006/min; `--half` (default devtest, sealed needs `--final`) added. API shapes in `docs/pulse_api_notes.md` |
| M8 | DONE (REAL) | Pulse partials (endpointing=false) for all 400 turns, 88.9 audio-minutes, $0.53, cached in `data/cache/pulse/partials/` and `data/cache/asr_eot_bench_hi_pulse.parquet` (inputs only, nothing scored on the sealed half). `python -m eval.run_bench --detector pulse_eot_hi --asr pulse` on dev-test: FCR@300 44.0% (Vosk 43.5%), Lat@5% 595 ms (Vosk 611), paired bootstrap -16.0 ms (95% CI -51.5 to +20.5). The Vosk arrival-time inflation is moot for v1, which will read Pulse partials |
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
| Pulse `eou_timeout_ms` sweep {300..1200} | DONE (REAL, dev-test) | `python -m adapters.pulse_endpointing --sweep 300 400 600 800 1000 1200` (SMALLEST_API_KEY from `.env`), re-score with `--score-only`, table with `python -m eval.pulse_vs_timer`. Silence-final FCR / premature turns / mean latency: 300 ms 9.3% / 36.5% / 627 ms; 400 5.2% / 31.5% / 717; 600 2.0% / 11.5% / 931; 800 0.8% / 4.0% / 1197; 1000 0.8% / 1.5% / 1368; 1200 0.4% / 1.0% / 1577. Real-VAD timer on the same turns: 600 ms 6.0% / 18.0% / 581 ms, 800 3.2% / 6.5% / 781, 1000 1.2% / 4.0% / 981. Pulse = timer + about 0.33 s delivery; treating every `is_final` as end-of-turn gives 48 to 55% premature turns (word-count finals). 8 streams dropped by a local network abort were re-streamed (charged as an upper bound) |
| Pulse partials replay (400 turns) | DONE (REAL, cache only) | `python -m adapters.pulse_endpointing --partials --half all`, then `--build-asr-cache`. 400/400 turns, 88.9 audio-minutes, $0.53. Feeds `run_bench --asr pulse` and the v1 refit (M5) |
| Real-Hindi pause episodes (IndicVoices extempore) | PENDING (IndicVoices terms) | `huggingface-cli download ai4bharat/IndicVoices --repo-type dataset --include "hindi/valid-*" "hindi/train-0000*" --local-dir data/indicvoices`, then `python -m bharat_turn.indicvoices_episodes --src data/indicvoices/hindi --inspect` and the same without `--inspect`. Dry-run on a fake shard: `python -m bharat_turn.fake_indicvoices && python -m bharat_turn.indicvoices_episodes --src data/_fake_indicvoices/hindi --out data/_fake_indicvoices/pauses` |
| Bharat-Turn-8k (two-party, diarised) | PENDING (data + diariser) | Label rules done and tested (`bharat_turn/labels.py`). Not on the Nov 4 path |
| Fine-tune Smart Turn v3.2 on real Hindi, mu-law + MUSAN aug | DONE (negative) | Ran 2026-10-24 on Modal L4: 0.31 h, $0.45. Train mix: IndicVoices Hindi extempore pauses 13,542 clips (1.7% end-of-turn), chirp3_hi 2,196, human en/es 1,500. Dev pauses (803 holds, 13 EoT): published v3.2 AUC 0.765, Lat@5% 0.80 s, FCR@600 11.7%; best fine-tuned epoch AUC 0.848 but Lat@5% 0.85 s, FCR@600 10.6%. No epoch beat v3.2 on the deploy metric, so nothing was exported. Caveat: only 13 dev EoT windows (IndicVoices is chunked, few true turn ends), so Lat@5% is very noisy; real two-party Hindi calls are needed. Log: Modal app ap-RPOnhkrhleGWEmyHmLNFZd |
| Lexical head + fusion | DONE (CPU v0 / v0.1) | v0.1 does not beat Smart Turn v3.2; the bar (FCR@300 < 27.4%, Lat@5% < 624 ms) is not met |
| Export ONNX, measure ms/hop | DONE (emulated) | Native x86 measurement PENDING an x86 box |
| Tune knobs on dev only, freeze, run test once | PARTIAL | Frozen knobs exist and transfer poorly. The one sealed-half run waits for v1 |
| Arbiter | NOT A RESULT | TTS proxy only (see B2) |
| Sidecar on live Pulse | PARTIAL | Key available and the live wire shapes are recorded (`docs/pulse_api_notes.md`); the sidecar itself has not been run against the live socket yet. Runs against the local mock: `python -m sidecar.mock_pulse` then `python -m sidecar.eot_sidecar --wav x.wav --base ws://127.0.0.1:8765` |

## What the student must do

1. Accept the IndicVoices terms on Hugging Face, download a few Hindi shards (command above), run the pause builder with `--inspect` once and check the task names and audio paths it prints (the `--task-regex` and `--recording-regex` defaults may need adjusting).
2. Create a Modal account (`pip install modal && modal setup`), then `modal run models/gpu/modal_app.py`. Delete the data volume afterwards (`modal volume delete pulse-eot-hi-data`; IndicVoices audio is gated).
3. DONE 2026-10-24: Pulse sweep (dev-test) and Pulse partials replay. Total Smallest spend 346.9 audio-minutes including the probe and re-streamed drops, about $2.08 at $0.006/min (no usage figure is exposed on the socket; computed from audio minutes). Next: Pulse partials for IndicVoices dev for the v1 refit (about $0.60, see M5).
4. Run the sealed eot-bench half exactly once with the final v1 and frozen knobs: `python -m eval.run_bench --detector pulse_eot_hi --half sealed --final`.

## Smoke test

`bash eval/run_all.sh --quick` (Windows: `.\run.ps1 quick`) passed on 2026-10-20 in 112 s on the 200 dev-test turns; `python -m pytest -q tests` passes (32 tests, 2026-10-24).

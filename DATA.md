# Datasets and models used

Nothing in this table is redistributed by this repo: audio, datasets and third-party model files are downloaded into `data/` or `third_party/`, which are git-ignored. Download steps are in [data/README.md](data/README.md). Licences were read on each source page (2026-10-18 to 2026-10-20).

## Datasets

| Dataset | Used for | Licence | How it is obtained | In this repo |
|---|---|---|---|---|
| `livekit/eot-bench-data`, `hi` validation (400 turns) | Test only (dev-test / sealed halves, `eval/split.py`) | CC BY 4.0 | `hf_hub_download` | Not redistributed; only turn ids in `eval/eot_bench_hi_split.csv` |
| `livekit/eot-bench-data`, `en` (`silence_spans` column only) | Hold/EoT duration histograms for dev episodes | CC BY 4.0 | parquet column read | Not redistributed (derived histogram cached under data/) |
| `livekit/eot-bench` harness and committed predictions | Metric parity, published rows | Apache-2.0 | `git clone` into third_party/ | Not redistributed |
| `pipecat-ai/smart-turn-data-v3.2-train` (Hindi rows, all Chirp3 TTS; real-human English/Spanish rows) | Training (v0, v0.1 head, GPU fine-tune replay) | CC BY 4.0 | `python data/_get_st_hi.py smart-turn-data-v3.2-train 83` | Not redistributed |
| `pipecat-ai/smart-turn-data-v3.2-test` (Hindi + human rows) | Dev split, v0.1 fusion-head fit | CC BY 4.0 | `python data/_get_st_hi.py smart-turn-data-v3.2-test 10` | Not redistributed |
| MUSAN noise (OpenSLR 17, HF mirror `FluidInference/musan`) | Noise conditions and training augmentation | CC BY 4.0 | `snapshot_download(allow_patterns=["noise/*"])` | Not redistributed |
| `ai4bharat/IndicVoices`, `hindi/` (extempore task) | PENDING: real-Hindi pause episodes for the GPU fine-tune | CC BY 4.0, gated (accept terms on HF) | `huggingface-cli download ai4bharat/IndicVoices --repo-type dataset --include "hindi/*"` | Not redistributed; built episodes stay in git-ignored `data/indicvoices_pauses/` |
| Gram Vaani (OpenSLR 118) | Not used | Academic use only | - | - |
| CALLFRIEND Hindi (LDC96S52) | Not used | Paid LDC licence | - | - |
| MUCS 2021 (OpenSLR 104) | Not used | CC BY-SA 4.0 | - | - |

## Models

| Model | Used for | Licence | How it is obtained | In this repo |
|---|---|---|---|---|
| Smart Turn v3.2 (`pipecat-ai/smart-turn-v3`, CPU int8 and fp32 ONNX) | Baseline; frozen feature in v0.1; initial weights of the GPU fine-tune | BSD-2-Clause | `hf_hub_download` | Not redistributed. `models/gpu/smart_turn_torch.py` re-implements its architecture (BSD-2 notice in LICENSE) |
| `deveshu/hinglish-turn-detector` | Baseline | MIT | `hf_hub_download` | Not redistributed (no longer needed for the mel filterbank) |
| Silero VAD v5 (`onnx-community/silero-vad`) | Streaming VAD | MIT | `hf_hub_download` | Not redistributed |
| Vosk `vosk-model-small-hi-0.22` | Streaming Hindi ASR replay | Apache-2.0 | `curl https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip` | Not redistributed |
| `openai/whisper-tiny` architecture (via transformers `WhisperEncoder`) | Encoder definition for the fine-tune | MIT | pip `transformers` | Not redistributed |
| LiveKit Turn Detector, v1, v1-mini | Not run (licence forbids standalone use outside LiveKit Agents); published rows rescored from committed predictions | LiveKit Model License | - | - |

## Artifacts this repo does ship

`models/artifacts/*.onnx`, `*.json`, `*_lexical.npz` (the v0 / v0.1 GBDT heads and lexical weights) and `arbiter.pkl` were trained in this repo on the Smart Turn v3.2 datasets above (CC BY 4.0: attribution to pipecat-ai / Daily), using Smart Turn v3.2 (BSD-2) scores as a feature. They are research prototypes.

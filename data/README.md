# Data

Everything under `data/` except this file and `_get_st_hi.py` is git-ignored. Licences are as shown on each dataset page (checked 2026-10-18).

## Downloaded automatically (public)

| What | Where it lands | Licence | How |
|---|---|---|---|
| `livekit/eot-bench-data`, `hi` validation (400 turns, 16 kHz). **Test only, never trained or tuned on** | `data/hf/livekit__eot-bench-data/data/hi/` | CC BY 4.0 (HF card) | `hf_hub_download` |
| `livekit/eot-bench-data`, `en`: only the `silence_spans` column, used for hold/EoT duration histograms | `data/cache/en_span_durations.json` | CC BY 4.0 | parquet column read |
| `livekit/eot-bench` harness + committed predictions | `third_party/eot-bench` | Apache-2.0 | `git clone` |
| `pipecat-ai/smart-turn-data-v3.2-train`, Hindi rows (shards 0 to 14 of 83: 2,252 clips, all `chirp3_1` TTS) plus real-human non-Hindi rows | `data/smart_turn_hi/` | CC BY 4.0 | `python data/_get_st_hi.py smart-turn-data-v3.2-train 83` (about 41 GB streamed; each shard is deleted after filtering; resumable) |
| `pipecat-ai/smart-turn-data-v3.2-test`, Hindi rows (shards 0 to 8 of 10) = **dev split** | `data/smart_turn_hi_test/` | CC BY 4.0 | `python data/_get_st_hi.py smart-turn-data-v3.2-test 10` |
| Smart Turn v3.2 ONNX (CPU int8 + fp32) | `data/hf/pipecat-ai__smart-turn-v3/` | BSD-2-Clause | `hf_hub_download` |
| `deveshu/hinglish-turn-detector` (int8 ONNX + mel filters) | `data/hf/deveshu__hinglish-turn-detector/` | MIT | `hf_hub_download` |
| Silero VAD v5 ONNX (`onnx-community/silero-vad`) | `data/hf/onnx-community__silero-vad/` | MIT | `hf_hub_download` |
| Vosk `vosk-model-small-hi-0.22` (open streaming Hindi ASR) | `data/vosk/` | Apache-2.0 | `curl https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip` |
| MUSAN noise (free-sound + sound-bible, 237 files), HF mirror `FluidInference/musan` | `data/musan/musan/noise/` | CC BY 4.0 (OpenSLR 17) | `snapshot_download(allow_patterns=["noise/*"])`. The OpenSLR tarball (11 GB) kept failing mid-transfer, so we used the mirror |

## Not used, and why

- **LiveKit Turn Detector (`livekit/turn-detector`, multilingual tag `v0.4.1-intl`):** this is a downloadable text model (Qwen2.5-0.5B). Its LiveKit Model License forbids use "on a standalone basis or with any other frameworks" than LiveKit Agents (sections 1 and 3.b). Running it in this harness would breach that, so it is not run. Its published Hindi rows, computed by LiveKit, are reproduced from the committed predictions in `third_party/eot-bench`.
- **LiveKit v1 / v1-mini audio models:** v1 is a cloud model (needs a LiveKit key). v1-mini needs `livekit-local-inference` under the same licence. Both use their published rows only.

## Manual steps (gated or paid)

1. **IndicVoices (`ai4bharat/IndicVoices`, CC BY 4.0, gated).** Log in to huggingface.co, open the dataset page, and accept the terms (share contact info). Then run:
   `huggingface-cli login` and `huggingface-cli download ai4bharat/IndicVoices --repo-type dataset --include "hindi/*" --local-dir data/indicvoices`.
   After that, `python -m bharat_turn.build_dataset --indicvoices-dir data/indicvoices/hindi`.
2. **pyannote.audio diarisation (alternative to Pulse batch):** accept the terms of `pyannote/speaker-diarization-3.1` on HF and set `HF_TOKEN`.
3. **Smallest Pulse API:** set `SMALLEST_API_KEY` (from the Smallest console). This unlocks `adapters/pulse_endpointing.py` and live `sidecar/eot_sidecar.py`.
4. **Gram Vaani (OpenSLR 118):** academic use only. Commercial use needs Gram Vaani's permission. It is not downloaded. Flag this to Smallest before using it for a shipped model.
5. **CALLFRIEND Hindi (LDC96S52):** paid LDC licence. Optional.
6. **MUCS 2021 (OpenSLR 104, CC BY-SA 4.0):** downloadable but not needed for v0. The share-alike licence would apply to derived text resources.

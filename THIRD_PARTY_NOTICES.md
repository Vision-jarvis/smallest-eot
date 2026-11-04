# Third-party notices

This repo is MIT-licensed (LICENSE). It contains no third-party model weights, audio or datasets; those are downloaded at run time and keep their own licences (DATA.md).

## Smart Turn v3.2 (pipecat-ai / Daily)

models/gpu/smart_turn_torch.py re-implements the SmartTurnV3Model architecture from
https://github.com/pipecat-ai/smart-turn (Smart Turn v3.2). Its licence:

```
BSD 2-Clause License

Copyright (c) 2024–2025, Daily

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

## Silero VAD and LiveKit eot-bench

Silero VAD (https://github.com/snakers4/silero-vad, MIT) is downloaded at run time and is not
redistributed. eval/metrics.py re-implements the metric definitions of LiveKit eot-bench
(https://github.com/livekit/eot-bench, Apache-2.0); no eot-bench code or data is redistributed.

## Data behind the shipped artifacts

The small models in `models/artifacts/` were trained in this repo on `pipecat-ai/smart-turn-data-v3.2-train` / `-test` (CC BY 4.0, attribution: pipecat-ai / Daily) with MUSAN noise (OpenSLR 17, CC BY 4.0), using Smart Turn v3.2 scores as a feature. Turn ids in `eval/eot_bench_hi_split.csv` refer to `livekit/eot-bench-data` (CC BY 4.0). Model weights and datasets from these sources are not covered by this repo's MIT licence; see DATA.md.

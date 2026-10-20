"""One entry point for the whole evaluation (Windows-friendly; eval/run_all.sh wraps it).

    python -m eval.run_all --quick     # 5-minute check: model + 2 baselines on the 200 dev-test Hindi turns
    python -m eval.run_all             # + SmartTurn fp32, channel conditions, ablations, arbiter, efficiency

Everything here runs on the dev-test half of eot-bench hi (eval/split.py). The sealed half is
scored only by an explicit `python -m eval.run_bench --half sealed --final ...`.

--quick runs: Silero timer, SmartTurn v3.2 int8 (one Whisper call per pause), the Vosk ASR
replay (cached after the first run) and Pulse-EoT-hi v0.1; then prints the table next to
LiveKit's published Hindi rows. The deveshu baseline is excluded from --quick because it
costs about 30 CPU-minutes (Whisper-tiny int8 with no batching).
"""

from __future__ import annotations

import argparse
import subprocess
import sys

QUICK = ["silero_timer", "smart_turn_v32", "pulse_eot_hi"]
FULL_EXTRA = ["smart_turn_v32_fp32", "hinglish_turn"]
# v0 (Hindi TTS only, no Smart Turn) and its ablations; v0.1 ablations
ABLATIONS = [
    "pulse_eot_hi:full", "pulse_eot_hi:audio_only", "pulse_eot_hi:no_lexhead", "pulse_eot_hi:no_HOLD", "pulse_eot_hi:no_VERB",
    "pulse_eot_hi:no_TAG", "pulse_eot_hi:no_DIGIT", "pulse_eot_hi:stacked_audio_only", "pulse_eot_hi:smart_turn_only",
]


def sh(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    dets = QUICK if a.quick else QUICK + FULL_EXTRA + ABLATIONS
    for d in dets:
        sh("-m", "eval.run_bench", "--detector", d)
    if not a.quick:
        for cond in ("mulaw8k", "snr10", "snr5"):
            for d in ("silero_timer", "smart_turn_v32", "pulse_eot_hi"):
                sh("-m", "eval.run_bench", "--detector", d, "--condition", cond)
        sh("-m", "eval.run_deployed")
        sh("-m", "arbiter.train_eval")
        sh("-m", "eval.bench_cpu", "--detector", "pulse_eot_hi")
    sh("-m", "eval.report")


if __name__ == "__main__":
    main()

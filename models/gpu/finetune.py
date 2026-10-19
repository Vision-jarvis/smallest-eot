"""PENDING (GPU): Hindi fine-tune of Smart Turn v3.2, the audio branch for Pulse-EoT-hi v1.

What runs today (CPU): `--prepare-only` writes the Hindi training set in Smart Turn's own
schema (audio, endpoint_bool, language, ...) to data/gpu_train/, from the Smart Turn train
split (Hindi) plus, once licensed, Bharat-Turn-8k train spans cut at hold / shift points.

What needs the GPU: training. Smart Turn publishes its training code in
github.com/pipecat-ai/smart-turn (BSD-2), which initialises from openai/whisper-tiny. We do
not re-implement it; the runner clones it and points it at data/gpu_train/ with the
hyper-parameters in finetune_smart_turn_hi.yaml, then exports int8 ONNX for models.train.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/gpu_train"
UPSTREAM = ROOT / "third_party/smart-turn"


def prepare() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / "data/smart_turn_hi").glob("hi_*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    manifest = ROOT / "bharat_turn/manifest.parquet"
    if manifest.exists():
        print("Bharat-Turn-8k manifest found: add its train-split spans with bharat_turn.build_dataset --export-clips")
    df.to_parquet(OUT / "train-hi-00000.parquet")
    print(f"wrote {len(df)} Hindi clips to {OUT}")
    return OUT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).with_name("finetune_smart_turn_hi.yaml")))
    ap.add_argument("--prepare-only", action="store_true")
    a = ap.parse_args()
    prepare()
    if a.prepare_only:
        return
    try:
        import torch
    except ImportError:
        sys.exit("PENDING: install torch with CUDA on a GPU machine (see STATUS.md)")
    if not torch.cuda.is_available():
        sys.exit("PENDING: no CUDA device. This step needs one 24 GB GPU (see STATUS.md).")
    if not UPSTREAM.exists():
        subprocess.run(["git", "clone", "https://github.com/pipecat-ai/smart-turn", str(UPSTREAM)], check=True)
    sys.exit(f"Next: run the upstream trainer in {UPSTREAM} on {OUT} with the hyper-parameters in {a.config}, "
             "then export int8 ONNX to models/artifacts/smart_turn_hi_int8.onnx.")


if __name__ == "__main__":
    main()

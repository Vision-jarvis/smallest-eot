"""One-command GPU run of the Smart Turn Hindi fine-tune on Modal.

    pip install modal && modal setup            # once, with the student's Modal account
    modal run models/gpu/modal_app.py           # upload data (first time), train, export, download

Prerequisites on this machine (all CPU): data/indicvoices_pauses/{train,dev}.parquet from
`python -m bharat_turn.indicvoices_episodes` (needs the IndicVoices terms), plus the Smart Turn
Hindi clips, MUSAN noise and the v3.2 fp32 ONNX already used by the repo (data/README.md).

What it costs: one L4 (24 GB) with 8 CPU cores and 32 GiB; models/gpu/cost_estimate.py gives
about 0.2 h / $0.26 for the default data size, and the function timeout (2 h) caps the worst
case at about $2.9 (prices from modal.com/pricing, 2026-10-20). Data upload is about 1.2 GB
the first time and is kept in the `pulse-eot-hi-data` volume (delete it after the run:
`modal volume delete pulse-eot-hi-data`; the IndicVoices audio is gated, do not share it).
Artifacts land in models/artifacts/smart_turn_hi_v1/ with run_summary.json, which records the
measured wall time and its dollar cost.
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[2]
VOL_NAME = "pulse-eot-hi-data"
REMOTE = "/data"
OUT_REMOTE = "runs/smart_turn_hi_v1"
DATA_DIRS = ["data/indicvoices_pauses", "data/smart_turn_hi", "data/smart_turn_hi_test", "data/musan/musan/noise"]
DATA_FILES = ["data/hf/pipecat-ai__smart-turn-v3/smart-turn-v3.2-gpu.onnx"]
CODE_DIRS = ["core", "eval", "models", "bharat_turn"]
TIMEOUT_H = 2.0

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")
    .pip_install("torch==2.14.1", "transformers==5.20.1", "onnx==1.23.2", "onnxruntime==1.31.0", "numpy==2.4.6",
                 "pandas==3.0.6", "pyarrow==26.0.0", "scipy==1.17.1", "scikit-learn==1.9.1", "soundfile==0.14.0", "PyYAML==6.0.3")
)
for d in CODE_DIRS:
    image = image.add_local_dir(ROOT / d, f"/repo/{d}", ignore=["**/__pycache__/**", "**/artifacts/**"])

app = modal.App("pulse-eot-hi-finetune", image=image)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)


@app.function(gpu="L4", cpu=8, memory=32 * 1024, volumes={REMOTE: vol}, timeout=int(TIMEOUT_H * 3600))
def train() -> dict:
    import json
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "models.gpu.finetune", "--config", "/repo/models/gpu/finetune_smart_turn_hi.yaml",
           "--data-root", REMOTE, "--out", f"{REMOTE}/{OUT_REMOTE}"]
    subprocess.run(cmd, cwd="/repo", check=True, env={"PYTHONPATH": "/repo", "PYTHONUTF8": "1", "PATH": "/usr/local/bin:/usr/bin:/bin"})
    vol.commit()
    return json.loads(Path(f"{REMOTE}/{OUT_REMOTE}/run_summary.json").read_text())


def _upload_missing() -> None:
    have = {e.path.strip("/") for e in vol.listdir("/", recursive=True)} if _has_any() else set()
    missing = [p for p in DATA_DIRS + DATA_FILES if p not in have]
    for p in missing:
        if not (ROOT / p).exists():
            raise SystemExit(f"missing local {p}: build it first (see the module docstring and STATUS.md)")
    if missing:
        print("uploading:", missing)
        with vol.batch_upload() as b:
            for p in missing:
                (b.put_directory if (ROOT / p).is_dir() else b.put_file)(str(ROOT / p), f"/{p}")


def _has_any() -> bool:
    try:
        return bool(vol.listdir("/"))
    except Exception:  # a brand-new volume
        return False


@app.local_entrypoint()
def main() -> None:
    import json

    _upload_missing()
    summary = train.remote()
    out = ROOT / "models/artifacts/smart_turn_hi_v1"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("run_summary.json", "smart_turn_hi_v1.onnx", "smart_turn_hi_v1_int8.onnx"):
        try:
            with (out / name).open("wb") as f:
                for chunk in vol.read_file(f"{OUT_REMOTE}/{name}"):
                    f.write(chunk)
        except Exception as e:  # nothing exported when no epoch beat v3.2 on dev
            print(f"not downloaded: {name} ({type(e).__name__})")
    print(json.dumps({k: summary.get(k) for k in ("result", "gpu_name", "wall_hours", "cost_usd_if_on_L4")}, indent=1))

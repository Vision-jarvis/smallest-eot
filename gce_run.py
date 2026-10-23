"""Plain-VM runner (one rented GPU VM, no Modal) for the Smart Turn Hindi fine-tune.

Runs the command models/gpu/modal_app.py::train runs (`python -m models.gpu.finetune --config
models/gpu/finetune_smart_turn_hi.yaml --data-root <root> --out <root>/runs/smart_turn_hi_v1`), with the data
root being this checkout (the same relative layout the Modal volume uses), then copies the three files the
Modal entrypoint downloads into models/artifacts/smart_turn_hi_v1/.

  python gce_run.py train          # GPU
  python gce_run.py smoke          # CPU: --dry-run --fake-indicvoices (pipeline test only)

Exit code 3 means a required data path is missing (job skipped, not failed); the IndicVoices pauses
(data/indicvoices_pauses/{train,dev}.parquet) must be built first, see STATUS.md.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CONFIG = "models/gpu/finetune_smart_turn_hi.yaml"
OUT = "runs/smart_turn_hi_v1"
MISSING = 3


def modal_data_paths() -> list[str]:
    """DATA_DIRS + DATA_FILES from models/gpu/modal_app.py, read without importing modal."""
    import ast

    tree = ast.parse((ROOT / "models/gpu/modal_app.py").read_text(encoding="utf-8"))
    vals = {t.id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
            for t in n.targets if isinstance(t, ast.Name) and t.id in ("DATA_DIRS", "DATA_FILES")}
    return vals["DATA_DIRS"] + vals["DATA_FILES"]


def _need(paths: list[str]) -> None:
    gone = [p for p in paths if not (ROOT / p).exists()]
    if gone:
        print(f"SKIP: missing {gone}", file=sys.stderr)
        sys.exit(MISSING)


def _cmd(data_root: Path, out: Path, extra: list[str]) -> list[str]:
    return [sys.executable, "-m", "models.gpu.finetune", "--config", str(ROOT / CONFIG), "--data-root", str(data_root),
            "--out", str(out), *extra]


def train() -> dict:
    _need(modal_data_paths() + ["data/indicvoices_pauses/train.parquet", "data/indicvoices_pauses/dev.parquet"])
    out = ROOT / OUT
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUTF8": "1"}
    subprocess.run(_cmd(ROOT, out, []), cwd=str(ROOT), check=True, env=env)
    dst = ROOT / "models/artifacts/smart_turn_hi_v1"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("run_summary.json", "smart_turn_hi_v1.onnx", "smart_turn_hi_v1_int8.onnx"):
        if (out / name).exists():
            shutil.copy2(out / name, dst / name)
        else:  # nothing exported when no epoch beat v3.2 on dev
            print(f"not exported: {name}")
    s = json.loads((out / "run_summary.json").read_text())
    return {k: s.get(k) for k in ("result", "gpu_name", "wall_hours")}


def smoke() -> dict:
    if not (ROOT / "data/_fake_indicvoices/pauses/train.parquet").exists():
        subprocess.run([sys.executable, "-m", "bharat_turn.fake_indicvoices"], cwd=str(ROOT), check=True)
        subprocess.run([sys.executable, "-m", "bharat_turn.indicvoices_episodes", "--src", "data/_fake_indicvoices/hindi",
                        "--out", "data/_fake_indicvoices/pauses"], cwd=str(ROOT), check=True)
    out = ROOT / "data/_dryrun_finetune"
    subprocess.run(_cmd(ROOT, out, ["--dry-run", "--fake-indicvoices"]), cwd=str(ROOT), check=True,
                   env={**os.environ, "PYTHONUTF8": "1"})
    s = json.loads((out / "run_summary.json").read_text())
    return {k: s.get(k) for k in ("dry_run", "device", "n_train", "steps", "first_losses", "last_losses", "result",
                                  "parity_torch_vs_onnx_fp32")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "smoke"])
    a = ap.parse_args()
    print(json.dumps(train() if a.cmd == "train" else smoke(), default=float))


if __name__ == "__main__":
    main()

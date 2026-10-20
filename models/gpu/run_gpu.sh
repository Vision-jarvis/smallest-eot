#!/usr/bin/env bash
# One-command GPU path (see STATUS.md). Needs: IndicVoices terms accepted + downloaded, a Modal account.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=python; [ -x .venv/Scripts/python ] && PY=.venv/Scripts/python; [ -x .venv/bin/python ] && PY=.venv/bin/python
export PYTHONUTF8=1
[ -f data/indicvoices_pauses/train.parquet ] || "$PY" -m bharat_turn.indicvoices_episodes --src data/indicvoices/hindi --max-pauses 6000
"$PY" -m models.gpu.cost_estimate
"$PY" -m modal run models/gpu/modal_app.py

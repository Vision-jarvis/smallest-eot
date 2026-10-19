#!/usr/bin/env bash
# Usage: bash eval/run_all.sh [--quick]   (uses ./.venv if present)
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
PY=python
[ -x .venv/Scripts/python ] && PY=.venv/Scripts/python
[ -x .venv/bin/python ] && PY=.venv/bin/python
"$PY" -m eval.run_all "$@"

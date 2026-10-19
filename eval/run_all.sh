#!/usr/bin/env bash
# Usage: bash eval/run_all.sh [--quick]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONUTF8=1
python -m eval.run_all "$@"

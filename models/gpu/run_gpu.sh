#!/usr/bin/env bash
# PENDING: one-command GPU runner for the Smart Turn Hindi fine-tune.
# Prerequisites: a CUDA box with torch+transformers, HF token with IndicVoices terms accepted,
# and bharat_turn/manifest.parquet built (python -m bharat_turn.build_dataset).
set -euo pipefail
cd "$(dirname "$0")/../.."
python -c "import torch; assert torch.cuda.is_available(), 'needs a CUDA GPU'"
python -m models.gpu.finetune --config models/gpu/finetune_smart_turn_hi.yaml

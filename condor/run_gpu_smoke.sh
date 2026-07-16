#!/usr/bin/env bash
set -euo pipefail
# shellcheck disable=SC1091
source condor/setup_environment.sh
exec python -u scripts/condor_gpu_smoke.py --model-config configs/model_qwen7b_4bit.yaml

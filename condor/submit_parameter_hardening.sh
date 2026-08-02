#!/usr/bin/env bash
set -euo pipefail
MODE=${1:---dry-run}
if [[ "$MODE" != "--dry-run" && "$MODE" != "--submit" ]]; then
  echo "Usage: $0 [--dry-run|--submit]" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
if [[ -f condor/condor.env ]]; then
  # shellcheck disable=SC1091
  source condor/condor.env
fi
SHARED_ROOT=${SLGEO_SHARED_ROOT:-/scratch/compuling/$USER/beyond-steering-vectors}
export SLGEO_SHARED_ROOT="$SHARED_ROOT"
EXECUTION_COMMIT=$(git rev-parse HEAD)
START_EPOCH=$(date +%s)
DAG=condor/runtime/parameter_hardening.dag
mkdir -p condor/logs condor/runtime "$SHARED_ROOT/results/research/plans"
python3 scripts/run_parameter_hardening_manifest.py \
  --emit-plan "$SHARED_ROOT/results/research/parameter_hardening_preflight_plan.json" \
  --resume --require-adapters
python3 scripts/generate_parameter_hardening_dag.py
python3 scripts/dag_notifications.py --source condor/parameter_hardening.dag --output "$DAG" \
  --study "Parameter Formation v1" --git-commit "$EXECUTION_COMMIT" \
  --result-path "$SHARED_ROOT/results/research/qwen7b_cat_parameter_hardening_v1" \
  --start-epoch "$START_EPOCH" --ntfy-topic "${NTFY_TOPIC:-}" \
  --container-image "${CONDOR_CONTAINER_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}"
condor_submit_dag -no_submit -f "$DAG"
if [[ "$MODE" == "--dry-run" ]]; then
  echo "READY: Phase-1 inputs and 52-node DAG validated; nothing submitted."
  exit 0
fi
condor_submit_dag -f "$DAG"

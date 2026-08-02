#!/usr/bin/env bash
set -euo pipefail
MODE=${1:---dry-run}
if [[ "$MODE" != "--dry-run" && "$MODE" != "--submit" ]]; then
  echo "Usage: $0 [--dry-run|--submit]" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
SHARED_ROOT=${SLGEO_SHARED_ROOT:-/scratch/compuling/$USER/beyond-steering-vectors}
export SLGEO_SHARED_ROOT="$SHARED_ROOT"
mkdir -p condor/logs "$SHARED_ROOT/results/research/plans"
python3 scripts/run_parameter_hardening_manifest.py \
  --emit-plan "$SHARED_ROOT/results/research/parameter_hardening_preflight_plan.json" \
  --resume --require-adapters
python3 scripts/generate_parameter_hardening_dag.py
condor_submit_dag -no_submit -f condor/parameter_hardening.dag
if [[ "$MODE" == "--dry-run" ]]; then
  echo "READY: Phase-1 inputs and 52-node DAG validated; nothing submitted."
  exit 0
fi
condor_submit_dag -f condor/parameter_hardening.dag

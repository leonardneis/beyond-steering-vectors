#!/usr/bin/env bash
set -euo pipefail

INTERVAL=${1:-30}
cd "$(dirname "$0")/.."
export SLGEO_SHARED_ROOT=${SLGEO_SHARED_ROOT:-/scratch/compuling/$USER/beyond-steering-vectors}
MANIFEST=configs/validation/cat_cross_seed_confirmatory.yaml
STATUS="$SLGEO_SHARED_ROOT/results/confirmatory/qwen7b_cat_cross_seed_v1/status.md"

while true; do
  python3 scripts/confirmatory_status.py --manifest "$MANIFEST" --condor
  echo
  cat "$STATUS"
  echo "## Live HTCondor jobs"
  condor_q -nobatch -constraint 'TaskId =!= undefined'
  echo
  echo "## Running GPU allocations"
  condor_q -run -constraint 'RequestGPUs > 0' \
    -af ClusterId ProcId TaskId RemoteHost RequestGPUs AssignedGPUs
  echo
  echo "## Held jobs"
  condor_q -hold -af ClusterId ProcId TaskId HoldReasonCode HoldReason
  sleep "$INTERVAL"
done

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
MANIFEST=configs/validation/cat_cross_seed_confirmatory.yaml

mkdir -p condor/logs "$SHARED_ROOT/preflight"
./condor/repair_scratch_group.sh "$SHARED_ROOT" "${SLGEO_QUOTA_GROUP:-compuling}"
python3 scripts/confirmatory_preflight.py \
  --shared-root "$SHARED_ROOT" \
  --report "$SHARED_ROOT/preflight/confirmatory_preflight.json"
python3 scripts/generate_condor_dag.py --validate-only
condor_submit_dag -no_submit -f condor/confirmatory.dag

if [[ "$MODE" == "--dry-run" ]]; then
  echo "READY: all prerequisites and native DAG files validated; nothing submitted."
  exit 0
fi

condor_submit_dag -f condor/confirmatory.dag | tee "$SHARED_ROOT/preflight/submission.txt"

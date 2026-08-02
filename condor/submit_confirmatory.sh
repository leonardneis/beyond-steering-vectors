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
MANIFEST=configs/validation/cat_cross_seed_confirmatory.yaml
EXECUTION_COMMIT=$(git rev-parse HEAD)
START_EPOCH=$(date +%s)
DAG=condor/runtime/confirmatory.dag

mkdir -p condor/logs condor/runtime "$SHARED_ROOT/preflight"
./condor/repair_scratch_group.sh "$SHARED_ROOT" "${SLGEO_QUOTA_GROUP:-compuling}"
python3 scripts/confirmatory_preflight.py \
  --shared-root "$SHARED_ROOT" \
  --report "$SHARED_ROOT/preflight/confirmatory_preflight.json"
python3 scripts/generate_condor_dag.py --validate-only
python3 scripts/dag_notifications.py --source condor/confirmatory.dag --output "$DAG" \
  --study "Confirmatory Baseline" --git-commit "$EXECUTION_COMMIT" \
  --result-path "$SHARED_ROOT/results/confirmatory/qwen7b_cat_cross_seed_v1" \
  --start-epoch "$START_EPOCH" --ntfy-topic "${NTFY_TOPIC:-}" \
  --container-image "${CONDOR_CONTAINER_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}"
condor_submit_dag -no_submit -f "$DAG"
./condor/validate_submit_files.sh

if [[ "$MODE" == "--dry-run" ]]; then
  echo "READY: all prerequisites and native DAG files validated; nothing submitted."
  exit 0
fi

condor_submit_dag -f "$DAG" | tee "$SHARED_ROOT/preflight/submission.txt"

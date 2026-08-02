#!/usr/bin/env bash
set -euo pipefail
PAIR_INDEX=$1
STAGE=$2
COMMAND_INDEX=$3
MANIFEST=$4
TASK_ID=$5

# shellcheck disable=SC1091
source condor/setup_environment.sh
TMP_BASE=${TMPDIR:-/tmp}
LOCAL_STAGE=$(mktemp -d "$TMP_BASE/bsv-hardening-${TASK_ID}.XXXXXX")
PLAN="$SLGEO_SHARED_ROOT/results/research/plans/${TASK_ID}.json"
trap 'rm -rf "$LOCAL_STAGE"' EXIT
child=""
terminate_child() {
  if [[ -n "$child" ]]; then
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" || true
  fi
  exit 75
}
trap terminate_child TERM INT

python -u scripts/run_parameter_hardening_manifest.py \
  --manifest "$MANIFEST" --pair-index "$PAIR_INDEX" --stage "$STAGE" \
  --command-index "$COMMAND_INDEX" --emit-plan "$PLAN" --execute --resume \
  --require-adapters --scratch-root "$LOCAL_STAGE" --retries 1 &
child=$!
set +e
wait "$child"
status=$?
set -e
child=""
exit "$status"

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
LOCAL_STAGE=$(mktemp -d "$TMP_BASE/bsv-${TASK_ID}.XXXXXX")
PLAN="$SLGEO_SHARED_ROOT/results/research/plans/${TASK_ID}.json"
child=""
tee_pid=""
ERROR_CAPTURE="$LOCAL_STAGE/stderr.log"
ERROR_FIFO="$LOCAL_STAGE/stderr.fifo"
GPU_RESOURCE_EXIT_CODE=85

terminate_child() {
  if [[ -n "$child" ]]; then
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" || true
  fi
  if [[ -n "$tee_pid" ]]; then
    wait "$tee_pid" 2>/dev/null || true
  fi
  exit 75
}
trap terminate_child TERM INT
trap 'rm -rf "$LOCAL_STAGE"' EXIT

mkfifo "$ERROR_FIFO"
tee "$ERROR_CAPTURE" <"$ERROR_FIFO" >&2 &
tee_pid=$!
python -u scripts/run_activation_behavior_dissociation_manifest.py \
  --manifest "$MANIFEST" --pair-index "$PAIR_INDEX" --stage "$STAGE" \
  --command-index "$COMMAND_INDEX" --emit-plan "$PLAN" --execute --resume \
  --require-adapters --scratch-root "$LOCAL_STAGE" --retries 1 2>"$ERROR_FIFO" &
child=$!
set +e
wait "$child"
status=$?
set -e
child=""
wait "$tee_pid"
tee_pid=""

if [[ "$status" -ne 0 && "${SLGEO_FORCE_SINGLE_GPU:-0}" == "1" ]] &&
   grep -Eiq \
     'CUDA-capable device\(s\) is/are busy or unavailable|all CUDA-capable devices are busy or unavailable|CUDA driver initialization failed|No CUDA GPUs are available' \
     "$ERROR_CAPTURE"; then
  exit "$GPU_RESOURCE_EXIT_CODE"
fi
exit "$status"

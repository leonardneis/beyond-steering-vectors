#!/usr/bin/env bash
# CTS Stage 0 node wrapper. Exit codes: 0 ok, 75 SIGTERM (retried), 85 GPU resource (rematched in place),
# 86 identity/integrity refusal (never retried), other non-zero = software error.
set -euo pipefail
COMMAND=$1
TARGET=$2
TASK_ID=$3

if [[ -z "${SLGEO_EXECUTION_GIT_COMMIT:-}" || "$SLGEO_EXECUTION_GIT_COMMIT" == "UNFROZEN" ]]; then
  echo "Refusing execution without a frozen Git commit." >&2
  exit 86
fi
# shellcheck disable=SC1091
source condor/setup_environment.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1

TMP_BASE=${TMPDIR:-/tmp}
LOCAL_STAGE=$(mktemp -d "$TMP_BASE/bsv-cts-${TASK_ID}.XXXXXX")
ERROR_CAPTURE="$LOCAL_STAGE/stderr.log"
ERROR_FIFO="$LOCAL_STAGE/stderr.fifo"
child=""
tee_pid=""
terminate_child() {
  if [[ -n "$child" ]]; then
    kill -TERM "$child" 2>/dev/null || true
    wait "$child" || true
  fi
  if [[ -n "$tee_pid" ]]; then wait "$tee_pid" 2>/dev/null || true; fi
  exit 75
}
trap terminate_child TERM INT
trap 'rm -rf "$LOCAL_STAGE"' EXIT

case "$COMMAND" in
  plan) args=(plan) ;;
  run) args=(run --shard "$TARGET") ;;
  techval) args=(techval --name "$TARGET") ;;
  techval-cpu) args=(techval-cpu) ;;
  *) echo "Unknown command $COMMAND" >&2; exit 2 ;;
esac

mkfifo "$ERROR_FIFO"
tee "$ERROR_CAPTURE" <"$ERROR_FIFO" >&2 &
tee_pid=$!
python -u scripts/cts_stage0.py "${args[@]}" 2>"$ERROR_FIFO" &
child=$!
set +e
wait "$child"
status=$?
set -e
child=""
wait "$tee_pid"
tee_pid=""
if [[ "$status" -ne 0 && "$status" -ne 86 ]] &&
   grep -Eiq 'CUDA-capable device\(s\) is/are busy or unavailable|all CUDA-capable devices are busy or unavailable|CUDA driver initialization failed|No CUDA GPUs are available' "$ERROR_CAPTURE"; then
  exit 85
fi
exit "$status"

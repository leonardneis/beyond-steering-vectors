#!/usr/bin/env bash
set -u
DAG_STATUS=$1
FAILED_COUNT=$2
DAG_ID=$3
STUDY=${4//_/ }
GIT_COMMIT=$5
START_EPOCH=$6
RESULT_PATH=$7

# Optional: a path whose existence marks a budget stop (set by DAGs with a budget gate; checked, never sent).
BUDGET_STOP_MARKER=${BSV_BUDGET_STOP_MARKER:-}

END_EPOCH=$(date +%s)
DURATION=$((END_EPOCH - START_EPOCH))
if [[ "$DAG_STATUS" == "0" && "$FAILED_COUNT" == "0" ]]; then
  RESULT_ARGS=(--result-path "$RESULT_PATH")
  FINAL_EXIT=0
else
  RESULT_ARGS=()
  FINAL_EXIT=1
fi

# notify.py derives the terminal status (SUCCESS, BUDGET_STOP, REMOVED, TECHNICAL_FAIL) from these values.
python3 scripts/notify.py \
  --study "$STUDY" --event DAG --dag-status "$DAG_STATUS" --failed-count "$FAILED_COUNT" \
  --budget-stop-marker "$BUDGET_STOP_MARKER" --dag-id "$DAG_ID" \
  --git-commit "$GIT_COMMIT" --duration-seconds "$DURATION" \
  "${RESULT_ARGS[@]}" || true

# A FINAL node determines overall DAG status. Preserve upstream failure even
# when notification delivery itself is disabled or unavailable.
exit "$FINAL_EXIT"

#!/usr/bin/env bash
set -u
DAG_STATUS=$1
FAILED_COUNT=$2
DAG_ID=$3
STUDY=${4//_/ }
GIT_COMMIT=$5
START_EPOCH=$6
RESULT_PATH=$7

END_EPOCH=$(date +%s)
DURATION=$((END_EPOCH - START_EPOCH))
if [[ "$DAG_STATUS" == "0" && "$FAILED_COUNT" == "0" ]]; then
  STATUS=SUCCESS
  RESULT_ARGS=(--result-path "$RESULT_PATH")
  FINAL_EXIT=0
else
  STATUS=FAILED
  RESULT_ARGS=()
  FINAL_EXIT=1
fi

python3 scripts/notify.py \
  --study "$STUDY" --event DAG --status "$STATUS" --dag-id "$DAG_ID" \
  --git-commit "$GIT_COMMIT" --duration-seconds "$DURATION" \
  "${RESULT_ARGS[@]}" || true

# A FINAL node determines overall DAG status. Preserve upstream failure even
# when notification delivery itself is disabled or unavailable.
exit "$FINAL_EXIT"

#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./condor/monitor_confirmatory.sh
  ./condor/monitor_confirmatory.sh --watch [SECONDS] [--events]

Options:
  --watch [SECONDS]  Refresh periodically (default: 30 seconds).
  --events           After the initial snapshot, print only state changes.
  -h, --help         Show this help.

A single numeric argument remains supported as shorthand for --watch SECONDS.
EOF
}

WATCH=0
EVENTS=0
INTERVAL=30
while (($#)); do
  case "$1" in
    --watch)
      WATCH=1
      shift
      if (($#)) && [[ "$1" =~ ^[1-9][0-9]*$ ]]; then
        INTERVAL=$1
        shift
      fi
      ;;
    --events)
      EVENTS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    [1-9][0-9]*)
      WATCH=1
      INTERVAL=$1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done
if ((EVENTS && ! WATCH)); then
  echo "--events requires --watch." >&2
  exit 2
fi

cd "$(dirname "$0")/.."
export SLGEO_SHARED_ROOT=${SLGEO_SHARED_ROOT:-/scratch/compuling/$USER/beyond-steering-vectors}
MANIFEST=configs/validation/cat_cross_seed_confirmatory.yaml
RUN_ROOT="$SLGEO_SHARED_ROOT/results/confirmatory/qwen7b_cat_cross_seed_v1"
STATUS_MD="$RUN_ROOT/status.md"
SESSION_DIR=$(mktemp -d "${TMPDIR:-/tmp}/bsv-monitor.XXXXXX")
PREVIOUS="$SESSION_DIR/previous.json"
CURRENT="$SESSION_DIR/current.json"
cleanup() {
  rm -rf "$SESSION_DIR"
}
stop_monitor() {
  echo
  echo "Monitoring stopped; the experiment continues in HTCondor."
  exit 0
}
trap cleanup EXIT
trap stop_monitor INT TERM

refresh_status() {
  python3 scripts/confirmatory_status.py \
    --manifest "$MANIFEST" --condor --snapshot-json "$CURRENT" >/dev/null
}

render_full_snapshot() {
  echo
  cat "$STATUS_MD"
  echo "## Live HTCondor jobs"
  condor_q -nobatch -constraint 'TaskId =!= undefined' ||
    echo "condor_q temporarily unavailable."
  echo
  echo "## Running GPU allocations"
  condor_q -run -constraint 'RequestGPUs > 0' \
    -af ClusterId ProcId TaskId RemoteHost RequestGPUs AssignedGPUs ||
    echo "GPU allocation query temporarily unavailable."
  echo
  echo "## Held jobs"
  condor_q -hold -af ClusterId ProcId TaskId HoldReasonCode HoldReason ||
    echo "Held-job query temporarily unavailable."
}

refresh_status
render_full_snapshot
if ((! WATCH)); then
  exit 0
fi

cp "$CURRENT" "$PREVIOUS"
echo
if ((EVENTS)); then
  echo "Event mode active; subsequent refreshes show changes only (interval: ${INTERVAL}s)."
else
  echo "Watch mode active; refreshing full snapshot every ${INTERVAL}s."
fi
echo "Press Ctrl+C to stop monitoring. The experiment is not modified."

while true; do
  sleep "$INTERVAL"
  refresh_status
  if ((EVENTS)); then
    python3 scripts/confirmatory_status_events.py "$PREVIOUS" "$CURRENT"
  else
    render_full_snapshot
  fi
  cp "$CURRENT" "$PREVIOUS"
done

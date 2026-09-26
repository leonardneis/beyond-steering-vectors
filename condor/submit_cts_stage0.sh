#!/usr/bin/env bash
# CTS Stage 0 submission (submit host). Default is a dry run.
#   --technical-validation   outcome-blind technical validation DAG
#   --scientific             scientific execution (refused unless a committed authorization file exists)
#   --submit                 actually submit (otherwise validate only)
set -euo pipefail
MODE=""
SUBMIT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --technical-validation) MODE=techval; shift ;;
    --scientific) MODE=scientific; shift ;;
    --submit) SUBMIT=1; shift ;;
    *) echo "Usage: $0 --technical-validation|--scientific [--submit]" >&2; exit 2 ;;
  esac
done
[[ -n "$MODE" ]] || { echo "Choose --technical-validation or --scientific" >&2; exit 2; }
cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd)
if [[ -f condor/condor.env ]]; then
  # shellcheck disable=SC1091
  source condor/condor.env
fi
: "${SLGEO_SHARED_ROOT:?Set SLGEO_SHARED_ROOT (persistent shared storage) in the environment or condor/condor.env}"
SHARED_ROOT=$SLGEO_SHARED_ROOT

# Code state: clean CTS checkout at a commit that keeps the frozen package byte-identical.
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing submission from a dirty or untracked worktree." >&2; exit 2
fi
if [[ "$(git rev-parse 'prereg/cts-stage0-v1^{commit}')" != "43dd95d3d476a9db9ebb9f8d0aa5639c65a1c1ab" ]]; then
  echo "Preregistration tag missing or moved." >&2; exit 2
fi
python3 -B research/cts_stage0_v1/tools/test_freeze_inputs.py
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Freeze test modified the worktree." >&2; exit 2
fi
EXECUTION_COMMIT=$(git rev-parse HEAD)

# Maintenance blackout (announced 2026-09-29 to 2026-10-01): no submission inside or within 12 h before it.
python3 - <<'PY'
import datetime as dt, sys
now = dt.datetime.now(dt.timezone.utc)
start = dt.datetime(2026, 9, 28, 22, 0, tzinfo=dt.timezone.utc) - dt.timedelta(hours=12)
end = dt.datetime(2026, 10, 1, 22, 0, tzinfo=dt.timezone.utc)
if start <= now <= end:
    sys.exit("Refusing submission inside the maintenance blackout window")
PY

if [[ "$MODE" == scientific ]]; then
  AUTH=research/cts_stage0_v1_execution/SCIENTIFIC_EXECUTION_AUTHORIZATION.json
  if [[ ! -f "$AUTH" ]]; then
    echo "Scientific execution is not authorized (no committed $AUTH)." >&2; exit 2
  fi
  echo "Scientific submission requires the authorization workflow; not implemented in this script version." >&2
  exit 2
fi

ROOT_TV="$SHARED_ROOT/results/research/qwen7b_cts_stage0_v1_technical_validation"
mkdir -p condor/logs condor/runtime
if [[ ! -f "$ROOT_TV/plan/run_record.json" ]]; then
  python3 -B scripts/cts_stage0.py submit-record --technical-validation
fi
python3 - "$ROOT_TV/plan/run_record.json" "$EXECUTION_COMMIT" <<'PY'
import json, sys
record = json.load(open(sys.argv[1]))
if record["execution_commit"] != sys.argv[2]:
    sys.exit("Existing technical-validation record belongs to another commit; use a fresh validation root")
PY
RUNTIME_DAG=condor/runtime/cts_stage0_technical_validation.dag
generator=(python3 -B scripts/generate_cts_stage0_dag.py --technical-validation --output "$RUNTIME_DAG"
  --execution-git-commit "$EXECUTION_COMMIT" --repo-root "$REPO_ROOT" --shared-root "$SHARED_ROOT"
  --start-epoch "$(date +%s)")
if [[ -n "${NTFY_TOPIC:-}" ]]; then generator+=(--ntfy-topic "$NTFY_TOPIC"); fi
"${generator[@]}"

TMP_BASE=${TMPDIR:-/tmp}
VALIDATION_DIR=$(mktemp -d "$TMP_BASE/bsv-cts-submit.XXXXXX")
trap 'rm -rf "$VALIDATION_DIR"' EXIT
macros=("BsvTaskId=cts_dry" "BsvCommand=techval" "BsvTarget=dry" "BsvRepoRoot=$REPO_ROOT"
  "BsvSharedRoot=$SHARED_ROOT" "BsvRequestCpus=4" "BsvRequestMemoryMB=32768"
  "BsvExecutionGitCommit=$EXECUTION_COMMIT" "BsvMachineRequirement=True"
  "BsvDockerImage=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755")
condor_submit -dry-run "$VALIDATION_DIR/gpu.classad" "${macros[@]}" condor/cts_stage0_task_gpu.sub >/dev/null
condor_submit -dry-run "$VALIDATION_DIR/cpu.classad" "${macros[@]}" condor/cts_stage0_task_cpu.sub >/dev/null
grep -q 'NVIDIA A100-PCIE-40GB' "$VALIDATION_DIR/gpu.classad"
grep -q '570.211.01' "$VALIDATION_DIR/gpu.classad"
condor_submit_dag -no_submit -f "$RUNTIME_DAG" >/dev/null
if [[ "$SUBMIT" -ne 1 ]]; then
  echo "READY: technical-validation DAG validated for $EXECUTION_COMMIT; nothing submitted."
  exit 0
fi
condor_submit_dag -f "$RUNTIME_DAG"

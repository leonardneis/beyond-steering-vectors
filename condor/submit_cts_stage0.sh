#!/usr/bin/env bash
# CTS Stage 0 v2 submission (submit host). Default is a dry run (validate only, nothing submitted).
#   --technical-validation   TV-v2 DAG (new attempt directory per run tag; at most 3 attempts, 6 A100-h)
#   --scientific-plan        scientific plan job (refused unless authorized and the contract is frozen)
#   --scientific             scientific DAG from the plan (refused unless authorized; budget gate on every node)
#   --submit                 actually submit (otherwise validate only)
set -euo pipefail
MODE=""
SUBMIT=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --technical-validation) MODE=techval; shift ;;
    --scientific-plan) MODE=sciplan; shift ;;
    --scientific) MODE=scientific; shift ;;
    --submit) SUBMIT=1; shift ;;
    *) echo "Usage: $0 --technical-validation|--scientific-plan|--scientific [--submit]" >&2; exit 2 ;;
  esac
done
[[ -n "$MODE" ]] || { echo "Choose --technical-validation, --scientific-plan or --scientific" >&2; exit 2; }
cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd)
if [[ -f condor/condor.env ]]; then
  # shellcheck disable=SC1091
  source condor/condor.env
fi
: "${SLGEO_SHARED_ROOT:?Set SLGEO_SHARED_ROOT (persistent shared storage) in the environment or condor/condor.env}"
SHARED_ROOT=$SLGEO_SHARED_ROOT
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
SCI_ROOT="$SHARED_ROOT/results/research/qwen7b_cts_stage0_v2"
TV_BASE="$SHARED_ROOT/results/research/qwen7b_cts_stage0_v2_technical_validation"
ACCOUNTING_ROOT="$SHARED_ROOT/results/research/qwen7b_cts_stage0_v2_accounting"
SCI_CAP=30
PLANNING_FRACTION=0.8
TV_CAP=6
TV_MAX_ATTEMPTS=3

# Code state: clean checkout; reused v1 package byte-identical to its tag; v2 contract regenerates.
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing submission from a dirty or untracked worktree." >&2; exit 2
fi
if [[ "$(git rev-parse 'prereg/cts-stage0-v1^{commit}')" != "43dd95d3d476a9db9ebb9f8d0aa5639c65a1c1ab" ]]; then
  echo "v1 preregistration tag missing or moved." >&2; exit 2
fi
if [[ -n "$(git diff --name-only 43dd95d3d476a9db9ebb9f8d0aa5639c65a1c1ab HEAD -- research/cts_stage0_v1)" ]]; then
  echo "Reused v1 package differs from its tag." >&2; exit 2
fi
python3 -B research/cts_stage0_v1/tools/test_freeze_inputs.py
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Freeze test modified the worktree." >&2; exit 2
fi
EXECUTION_COMMIT=$(git rev-parse HEAD)

# Maintenance blackout: no submission from 2026-09-28 10:00 UTC to 2026-10-01 22:00 UTC.
python3 - <<'PY'
import datetime as dt, sys
now = dt.datetime.now(dt.timezone.utc)
start = dt.datetime(2026, 9, 28, 10, 0, tzinfo=dt.timezone.utc)
end = dt.datetime(2026, 10, 1, 22, 0, tzinfo=dt.timezone.utc)
if start <= now <= end:
    sys.exit("Refusing submission inside the maintenance blackout window")
PY

dry_run_submit_files() {
  local dir; dir=$(mktemp -d "${TMPDIR:-/tmp}/bsv-cts-submit.XXXXXX")
  local macros=("BsvTaskId=cts_dry" "BsvCommand=techval" "BsvTarget=dry" "BsvRepoRoot=$REPO_ROOT"
    "BsvSharedRoot=$SHARED_ROOT" "BsvRequestCpus=4" "BsvRequestMemoryMB=32768"
    "BsvExecutionGitCommit=$EXECUTION_COMMIT" "BsvMachineRequirement=True" "BsvDockerImage=$IMAGE"
    "BsvRunTag=dry" "BsvBudgetCategory=TV")
  condor_submit -dry-run "$dir/gpu.classad" "${macros[@]}" condor/cts_stage0_task_gpu.sub >/dev/null
  condor_submit -dry-run "$dir/cpu.classad" "${macros[@]}" condor/cts_stage0_task_cpu.sub >/dev/null
  grep -q 'NVIDIA A100-PCIE-40GB' "$dir/gpu.classad"
  grep -q '570.211.01' "$dir/gpu.classad"
  grep -q 'BsvBudgetCategory' "$dir/gpu.classad"
  rm -rf "$dir"
}

mkdir -p condor/logs condor/runtime

if [[ "$MODE" == techval ]]; then
  RUN_TAG="tv-$(date -u +%Y%m%dT%H%M%SZ)"
  TV_ROOT="$TV_BASE/$RUN_TAG"
  ATTEMPTS=$(find "$ACCOUNTING_ROOT/tv_attempts" -name '*.json' 2>/dev/null | wc -l)
  if [[ "$ATTEMPTS" -ge "$TV_MAX_ATTEMPTS" ]]; then
    echo "TV-v2 attempt limit ($TV_MAX_ATTEMPTS) reached; a dated researcher decision is required." >&2; exit 2
  fi
  RUNTIME_DAG=condor/runtime/cts_stage0_v2_technical_validation.dag
  generator=(python3 -B scripts/generate_cts_stage0_dag.py --technical-validation --output "$RUNTIME_DAG"
    --execution-git-commit "$EXECUTION_COMMIT" --repo-root "$REPO_ROOT" --shared-root "$SHARED_ROOT"
    --run-tag "$RUN_TAG" --out-root "$TV_ROOT" --cap "$TV_CAP" --accounting-root "$ACCOUNTING_ROOT" --start-epoch "$(date +%s)")
  if [[ -n "${NTFY_TOPIC:-}" ]]; then generator+=(--ntfy-topic "$NTFY_TOPIC"); fi
  "${generator[@]}"
  dry_run_submit_files
  condor_submit_dag -no_submit -f "$RUNTIME_DAG" >/dev/null
  if [[ "$SUBMIT" -ne 1 ]]; then
    echo "READY: TV-v2 DAG validated for $EXECUTION_COMMIT (run tag $RUN_TAG, attempt $((ATTEMPTS + 1))/$TV_MAX_ATTEMPTS); nothing submitted."
    exit 0
  fi
  SLGEO_RUN_TAG="$RUN_TAG" python3 -B scripts/cts_stage0.py submit-record --technical-validation --run-tag "$RUN_TAG"
  python3 -B scripts/cts_stage0_budget.py tv-attempt --accounting-root "$ACCOUNTING_ROOT" --run-tag "$RUN_TAG" \
    --commit "$EXECUTION_COMMIT" --max-attempts "$TV_MAX_ATTEMPTS"
  condor_submit_dag -f "$RUNTIME_DAG"
  exit 0
fi

# Scientific modes: committed researcher authorization, frozen contract, passing TV-v2 projection.
AUTH=research/cts_stage0_v2_execution/SCIENTIFIC_EXECUTION_AUTHORIZATION.json
if [[ ! -f "$AUTH" ]]; then
  echo "Scientific execution is not authorized (no committed $AUTH)." >&2; exit 2
fi
if ! grep -q 'status: frozen' configs/validation/cts_stage0_v2.yaml; then
  echo "The v2 contract is not frozen." >&2; exit 2
fi
PROJECTION=$(python3 - "$AUTH" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["tv_projection_record"])
PY
)
python3 -B scripts/cts_stage0_budget.py authorize-check --projection "$PROJECTION" --cap "$SCI_CAP" --fraction "$PLANNING_FRACTION"
RUN_TAG="sci-v2"

if [[ "$MODE" == sciplan ]]; then
  if [[ "$SUBMIT" -ne 1 ]]; then echo "READY: scientific plan job validated; nothing submitted."; exit 0; fi
  python3 -B scripts/cts_stage0.py submit-record --run-tag "$RUN_TAG"
  condor_submit "BsvTaskId=cts_plan" "BsvCommand=plan" "BsvTarget=none" "BsvRepoRoot=$REPO_ROOT" \
    "BsvSharedRoot=$SHARED_ROOT" "BsvRequestCpus=2" "BsvRequestMemoryMB=8192" "BsvExecutionGitCommit=$EXECUTION_COMMIT" \
    "BsvMachineRequirement=True" "BsvDockerImage=$IMAGE" "BsvRunTag=$RUN_TAG" "BsvBudgetCategory=SCI" condor/cts_stage0_task_cpu.sub
  exit 0
fi

PLAN="$SCI_ROOT/plan/plan.json"
[[ -f "$PLAN" ]] || { echo "No scientific plan at $PLAN; run --scientific-plan first." >&2; exit 2; }
RUNTIME_DAG=condor/runtime/cts_stage0_v2_scientific.dag
python3 -B scripts/generate_cts_stage0_dag.py --plan "$PLAN" --output "$RUNTIME_DAG" \
  --execution-git-commit "$EXECUTION_COMMIT" --repo-root "$REPO_ROOT" --shared-root "$SHARED_ROOT" \
  --run-tag "$RUN_TAG" --out-root "$SCI_ROOT" --cap "$SCI_CAP" --start-epoch "$(date +%s)"
dry_run_submit_files
condor_submit_dag -no_submit -f "$RUNTIME_DAG" >/dev/null
if [[ "$SUBMIT" -ne 1 ]]; then echo "READY: scientific DAG validated; nothing submitted."; exit 0; fi
condor_submit_dag -f "$RUNTIME_DAG"

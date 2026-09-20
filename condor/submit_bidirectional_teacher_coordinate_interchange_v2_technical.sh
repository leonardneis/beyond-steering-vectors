#!/usr/bin/env bash
set -euo pipefail

MODE=${1:---dry-run}
if [[ "$MODE" != "--dry-run" && "$MODE" != "--submit-technical" ]]; then
  echo "Only --dry-run or --submit-technical is supported; no scientific mode exists here." >&2
  exit 2
fi
cd "$(dirname "$0")/.."
if [[ -f condor/condor.env ]]; then
  # shellcheck disable=SC1091
  source condor/condor.env
fi
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing C18-v2 technical submission from a dirty worktree." >&2
  exit 2
fi

COMMIT=$(git rev-parse HEAD)
START_EPOCH=$(date +%s)
SHARED_ROOT=${SLGEO_SHARED_ROOT:-/scratch/compuling/$USER/beyond-steering-vectors}
RUNTIME_DAG=condor/runtime/bidirectional_teacher_coordinate_interchange_v2_technical.dag
RESULT_ROOT=$SHARED_ROOT/results/research/qwen7b_cat_bidirectional_teacher_coordinate_interchange_v2/technical
mkdir -p condor/runtime condor/logs
if [[ "$MODE" == "--submit-technical" ]] && [[ -e "$RESULT_ROOT/validation.json" || -e "$RESULT_ROOT/audit.json" ]]; then
  echo "Refusing to overwrite an existing C18-v2 technical validation bundle." >&2
  exit 2
fi
python3 scripts/render_c18_v2_technical_dag.py \
  --source condor/bidirectional_teacher_coordinate_interchange_v2.dag \
  --output "$RUNTIME_DAG" --execution-git-commit "$COMMIT" \
  --start-epoch "$START_EPOCH" --ntfy-topic "${NTFY_TOPIC:-}"

TMP=$(mktemp -d "${TMPDIR:-/tmp}/c18-v2-condor-preflight.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
COMMON=(
  "BsvManifestPath=configs/validation/cat_bidirectional_teacher_coordinate_interchange_v2.yaml"
  "BsvRepoRoot=$HOME/beyond-steering-vectors"
  "BsvSharedRoot=$SHARED_ROOT"
  "BsvDockerImage=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
  "BsvExecutionGitCommit=$COMMIT"
)
condor_submit -dry-run "$TMP/cpu.classad" \
  "BsvTaskId=c18v2_technical_preflight" "BsvMode=technical_preflight" "BsvCondition=none" \
  "${COMMON[@]}" condor/bidirectional_teacher_coordinate_interchange_v2_task_cpu.sub >/dev/null
test -s "$TMP/cpu.classad"
condor_submit -dry-run "$TMP/gpu.classad" \
  "BsvTaskId=c18v2_technical_00" "BsvMode=technical" "BsvCondition=none" \
  "${COMMON[@]}" condor/bidirectional_teacher_coordinate_interchange_v2_task_gpu.sub >/dev/null
test -s "$TMP/gpu.classad"
condor_submit -dry-run "$TMP/audit.classad" \
  "BsvTaskId=c18v2_technical_audit" "BsvMode=technical_audit" "BsvCondition=none" \
  "${COMMON[@]}" condor/bidirectional_teacher_coordinate_interchange_v2_task_cpu.sub >/dev/null
test -s "$TMP/audit.classad"
condor_submit_dag -no_submit -f "$RUNTIME_DAG"
if [[ "$MODE" == "--dry-run" ]]; then
  echo "READY: C18-v2 technical-only ClassAds and DAG validated; nothing submitted."
  exit 0
fi
condor_submit_dag -f "$RUNTIME_DAG"

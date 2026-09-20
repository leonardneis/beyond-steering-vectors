#!/usr/bin/env bash
set -euo pipefail
MODE=${1:---dry-run}
if [[ "$MODE" != "--dry-run" && "$MODE" != "--submit-technical" ]]; then
  echo "Only --dry-run or --submit-technical is permitted before scientific authorization." >&2
  exit 2
fi
cd "$(dirname "$0")/.."
if [[ -f condor/condor.env ]]; then
  # shellcheck disable=SC1091
  source condor/condor.env
fi
# The login-node manifest preflight needs the same content-addressed dependency
# environment and storage remapping as the later Docker-universe tasks.
# shellcheck disable=SC1091
source condor/setup_environment.sh
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing C18 technical submission from a dirty worktree." >&2
  exit 2
fi
COMMIT=$(git rev-parse HEAD)
START_EPOCH=$(date +%s)
RUNTIME_DAG=condor/runtime/bidirectional_teacher_coordinate_interchange.dag
mkdir -p condor/runtime condor/logs
python3 scripts/run_bidirectional_teacher_coordinate_interchange_manifest.py \
  --mode technical --require-runtime-inputs \
  --emit-plan condor/runtime/c18_technical_plan.json >/dev/null
generator=(python3 scripts/generate_bidirectional_teacher_coordinate_interchange_dag.py \
  --mode technical --execution-git-commit "$COMMIT" --start-epoch "$START_EPOCH" \
  --output "$RUNTIME_DAG")
if [[ -n "${NTFY_TOPIC:-}" ]]; then generator+=(--ntfy-topic "$NTFY_TOPIC"); fi
"${generator[@]}"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/c18-condor-preflight.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
condor_submit -dry-run "$TMP/technical.classad" \
  "BsvTaskId=c18_preflight" "BsvMode=technical" "BsvCondition=none" \
  "BsvManifestPath=configs/validation/cat_bidirectional_teacher_coordinate_interchange_v1.yaml" \
  "BsvRepoRoot=$HOME/beyond-steering-vectors" \
  "BsvSharedRoot=${SLGEO_SHARED_ROOT:-/scratch/compuling/$USER/beyond-steering-vectors}" \
  "BsvDockerImage=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755" \
  "BsvExecutionGitCommit=$COMMIT" condor/bidirectional_teacher_coordinate_interchange_task_gpu.sub >/dev/null
test -s "$TMP/technical.classad"
condor_submit -dry-run "$TMP/audit.classad" \
  "BsvTaskId=c18_audit_preflight" "BsvMode=technical_audit" "BsvCondition=none" \
  "BsvManifestPath=configs/validation/cat_bidirectional_teacher_coordinate_interchange_v1.yaml" \
  "BsvRepoRoot=$HOME/beyond-steering-vectors" \
  "BsvSharedRoot=${SLGEO_SHARED_ROOT:-/scratch/compuling/$USER/beyond-steering-vectors}" \
  "BsvDockerImage=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755" \
  "BsvExecutionGitCommit=$COMMIT" condor/bidirectional_teacher_coordinate_interchange_task_cpu.sub >/dev/null
test -s "$TMP/audit.classad"
condor_submit -dry-run "$TMP/notify.classad" \
  "BsvRepoRoot=$HOME/beyond-steering-vectors" "BsvStudyName=c18_preflight" \
  "BsvExecutionGitCommit=$COMMIT" "BsvStartEpoch=$START_EPOCH" \
  "BsvResultPath=${SLGEO_SHARED_ROOT:-/scratch/compuling/$USER/beyond-steering-vectors}/results/research/qwen7b_cat_bidirectional_teacher_coordinate_interchange_v1/technical_validation" \
  "BsvNtfyTopic=" \
  "BsvDockerImage=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755" \
  "DAG_STATUS=0" "FAILED_COUNT=0" "DAGManJobId=0" \
  condor/dag_notification.sub >/dev/null
test -s "$TMP/notify.classad"
condor_submit_dag -no_submit -f "$RUNTIME_DAG"
if [[ "$MODE" == "--dry-run" ]]; then
  echo "READY: C18 technical-only ClassAd and DAG validated; nothing submitted."
  exit 0
fi
condor_submit_dag -f "$RUNTIME_DAG"

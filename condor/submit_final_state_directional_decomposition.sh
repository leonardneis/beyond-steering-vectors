#!/usr/bin/env bash
set -euo pipefail
MODE=--dry-run
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run|--submit) MODE=$1; shift ;;
    *) echo "Usage: $0 [--dry-run|--submit]" >&2; exit 2 ;;
  esac
done
cd "$(dirname "$0")/.."
if [[ -f condor/condor.env ]]; then
  # shellcheck disable=SC1091
  source condor/condor.env
fi
SHARED_ROOT=${SLGEO_SHARED_ROOT:-/scratch/compuling/$USER/beyond-steering-vectors}
IMAGE=${CONDOR_CONTAINER_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}
export SLGEO_SHARED_ROOT="$SHARED_ROOT"
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing submission from a dirty or untracked public worktree." >&2
  exit 2
fi
EXECUTION_COMMIT=$(git rev-parse HEAD)
PARENT_COMMIT=$(git rev-list -n 1 study/activation-behavior-dissociation-v1 2>/dev/null || true)
if [[ "$PARENT_COMMIT" != "90e8fdbe30e4709a554a314ba1ae663b3a2d49f1" ]]; then
  echo "Frozen parent study tag is absent or points to the wrong commit." >&2
  exit 2
fi
PREREG_COMMIT=$(git rev-list -n 1 prereg/final-state-directional-causal-decomposition-v1 2>/dev/null || true)
if [[ "$PREREG_COMMIT" != "$EXECUTION_COMMIT" ]]; then
  echo "Execution commit must equal the frozen preregistration tag." >&2
  exit 2
fi
START_EPOCH=$(date +%s)
mkdir -p condor/logs condor/runtime "$SHARED_ROOT/results/research/plans"
python3 scripts/run_final_state_directional_decomposition_manifest.py \
  --emit-plan "$SHARED_ROOT/results/research/final_state_directional_decomposition_preflight.json" \
  --resume --require-adapters

RUNTIME_DAG=condor/runtime/final_state_directional_decomposition.dag
generator=(python3 scripts/generate_final_state_directional_decomposition_dag.py --output "$RUNTIME_DAG" --container-image "$IMAGE" --execution-git-commit "$EXECUTION_COMMIT" --start-epoch "$START_EPOCH")
if [[ -n "${NTFY_TOPIC:-}" ]]; then generator+=(--ntfy-topic "$NTFY_TOPIC"); fi
"${generator[@]}"

TMP_BASE=${TMPDIR:-/tmp}
VALIDATION_DIR=$(mktemp -d "$TMP_BASE/bsv-fsd-submit-validation.XXXXXX")
trap 'rm -rf "$VALIDATION_DIR"' EXIT
macros=(
  "BsvTaskId=fsd_preflight" "BsvPairIndex=0" "BsvStage=state_runs"
  "BsvCommandIndex=0" "BsvManifestPath=configs/validation/cat_final_state_directional_causal_decomposition_v1.yaml"
  "BsvRepoRoot=$HOME/beyond-steering-vectors" "BsvSharedRoot=$SHARED_ROOT"
  "BsvDockerImage=$IMAGE" "BsvRequestCpus=4" "BsvRequestMemoryMB=16384"
  "BsvRetirementSeconds=86400" "BsvGpuResourceAttempts=4"
  "BsvExecutionGitCommit=$EXECUTION_COMMIT"
)
condor_submit -dry-run "$VALIDATION_DIR/cpu.classad" "${macros[@]}" \
  condor/final_state_directional_decomposition_task_cpu.sub >/dev/null
condor_submit -dry-run "$VALIDATION_DIR/gpu.classad" "${macros[@]}" \
  "BsvMinGpuMemoryMB=16384" condor/final_state_directional_decomposition_task_gpu.sub >/dev/null
test -s "$VALIDATION_DIR/cpu.classad"
test -s "$VALIDATION_DIR/gpu.classad"
condor_submit -dry-run "$VALIDATION_DIR/notify.classad" \
  "BsvRepoRoot=$HOME/beyond-steering-vectors" "BsvStudyName=preflight" \
  "BsvExecutionGitCommit=$EXECUTION_COMMIT" "BsvStartEpoch=$START_EPOCH" \
  "BsvResultPath=$SHARED_ROOT/results/research/qwen7b_cat_final_state_directional_causal_decomposition_v1" \
  "BsvNtfyTopic=" "DAG_STATUS=0" "FAILED_COUNT=0" "DAGManJobId=0" \
  condor/dag_notification.sub >/dev/null
test -s "$VALIDATION_DIR/notify.classad"
condor_submit_dag -no_submit -f "$RUNTIME_DAG"
if [[ "$MODE" == "--dry-run" ]]; then
  echo "READY: inputs, notification ClassAds, native submit files, and three-node DAG validated; nothing submitted."
  exit 0
fi
condor_submit_dag -f "$RUNTIME_DAG"

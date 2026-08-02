#!/usr/bin/env bash
set -euo pipefail
MODE=${1:---dry-run}
if [[ "$MODE" != "--dry-run" && "$MODE" != "--submit" ]]; then
  echo "Usage: $0 [--dry-run|--submit]" >&2
  exit 2
fi
cd "$(dirname "$0")/.."
if [[ -f condor/condor.env ]]; then
  # shellcheck disable=SC1091
  source condor/condor.env
fi
SHARED_ROOT=${SLGEO_SHARED_ROOT:-/scratch/compuling/$USER/beyond-steering-vectors}
IMAGE=${CONDOR_CONTAINER_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}
export SLGEO_SHARED_ROOT="$SHARED_ROOT"
EXECUTION_COMMIT=$(git rev-parse HEAD)
START_EPOCH=$(date +%s)
DAG=condor/runtime/activation_behavior_dissociation.dag
mkdir -p condor/logs condor/runtime "$SHARED_ROOT/results/research/plans"
python3 scripts/run_activation_behavior_dissociation_manifest.py \
  --emit-plan "$SHARED_ROOT/results/research/activation_behavior_dissociation_preflight.json" \
  --resume --require-adapters
python3 scripts/generate_activation_behavior_dissociation_dag.py \
  --container-image "$IMAGE"
python3 scripts/dag_notifications.py --source condor/activation_behavior_dissociation.dag --output "$DAG" \
  --study "Activation--Behavior Dissociation v1" --git-commit "$EXECUTION_COMMIT" \
  --result-path "$SHARED_ROOT/results/research/qwen7b_cat_activation_behavior_dissociation_v1" \
  --start-epoch "$START_EPOCH" --ntfy-topic "${NTFY_TOPIC:-}"

TMP_BASE=${TMPDIR:-/tmp}
VALIDATION_DIR=$(mktemp -d "$TMP_BASE/bsv-abd-submit-validation.XXXXXX")
trap 'rm -rf "$VALIDATION_DIR"' EXIT
macros=(
  "BsvTaskId=abd_preflight" "BsvPairIndex=0" "BsvStage=behavior_runs"
  "BsvCommandIndex=0" "BsvManifestPath=configs/validation/cat_activation_behavior_dissociation_v1.yaml"
  "BsvRepoRoot=$HOME/beyond-steering-vectors" "BsvSharedRoot=$SHARED_ROOT"
  "BsvDockerImage=$IMAGE" "BsvRequestCpus=4" "BsvRequestMemoryMB=16384"
  "BsvRetirementSeconds=86400" "BsvGpuResourceAttempts=4"
)
condor_submit -dry-run "$VALIDATION_DIR/cpu.classad" "${macros[@]}" \
  condor/activation_behavior_dissociation_task_cpu.sub >/dev/null
condor_submit -dry-run "$VALIDATION_DIR/gpu.classad" "${macros[@]}" \
  "BsvMinGpuMemoryMB=16384" condor/activation_behavior_dissociation_task_gpu.sub >/dev/null
test -s "$VALIDATION_DIR/cpu.classad"
test -s "$VALIDATION_DIR/gpu.classad"
condor_submit_dag -no_submit -f "$DAG"
if [[ "$MODE" == "--dry-run" ]]; then
  echo "READY: inputs, native submit files, and 10-node DAG validated; nothing submitted."
  exit 0
fi
condor_submit_dag -f "$DAG"

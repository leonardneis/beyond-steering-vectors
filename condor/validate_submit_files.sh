#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
SHARED_ROOT=${SLGEO_SHARED_ROOT:-/scratch/compuling/$USER/beyond-steering-vectors}
IMAGE=${CONDOR_CONTAINER_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}
TMP_BASE=${TMPDIR:-/tmp}
VALIDATION_DIR=$(mktemp -d "$TMP_BASE/bsv-condor-submit-validation.XXXXXX")
trap 'rm -rf "$VALIDATION_DIR"' EXIT

# DAGMan passes VARS entries to condor_submit as macros. Native dry-runs with the
# same macro names catch submit-command collisions that condor_submit_dag
# -no_submit alone cannot detect.
task_macros=(
  "BsvTaskId=preflight_submit"
  "BsvPairIndex=0"
  "BsvStage=prepare"
  "BsvCommandIndex=0"
  "BsvManifestPath=configs/validation/cat_cross_seed_confirmatory.yaml"
  "BsvRepoRoot=$HOME/beyond-steering-vectors"
  "BsvSharedRoot=$SHARED_ROOT"
  "BsvDockerImage=$IMAGE"
  "BsvRequestCpus=2"
  "BsvRequestMemoryMB=8192"
  "BsvRetirementSeconds=7200"
)

condor_submit -dry-run "$VALIDATION_DIR/task_cpu.classad" \
  "${task_macros[@]}" condor/task_cpu.sub >/dev/null
test -s "$VALIDATION_DIR/task_cpu.classad"

condor_submit -dry-run "$VALIDATION_DIR/task_gpu.classad" \
  "${task_macros[@]}" "BsvMinGpuMemoryMB=16384" condor/task_gpu.sub >/dev/null
test -s "$VALIDATION_DIR/task_gpu.classad"

finalize_macros=(
  "BsvTaskId=finalize_confirmatory"
  "BsvManifestPath=configs/validation/cat_cross_seed_confirmatory.yaml"
  "BsvRepoRoot=$HOME/beyond-steering-vectors"
  "BsvSharedRoot=$SHARED_ROOT"
  "BsvDockerImage=$IMAGE"
  "BsvRequestCpus=4"
  "BsvRequestMemoryMB=16384"
  "BsvRetirementSeconds=7200"
)
condor_submit -dry-run "$VALIDATION_DIR/finalize.classad" \
  "${finalize_macros[@]}" condor/finalize.sub >/dev/null
test -s "$VALIDATION_DIR/finalize.classad"

for template in gpu_smoke environment_smoke; do
  condor_submit -dry-run "$VALIDATION_DIR/$template.classad" \
    "condor/$template.sub" >/dev/null
  test -s "$VALIDATION_DIR/$template.classad"
done

echo "Validated all HTCondor node submit files with native condor_submit dry-runs."

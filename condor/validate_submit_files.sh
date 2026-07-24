#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
SHARED_ROOT=${SLGEO_SHARED_ROOT:-/scratch/compuling/$USER/beyond-steering-vectors}
IMAGE=${CONDOR_CONTAINER_IMAGE:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}
TMP_BASE=${TMPDIR:-/tmp}
VALIDATION_DIR=$(mktemp -d "$TMP_BASE/slgeo-condor-submit-validation.XXXXXX")
trap 'rm -rf "$VALIDATION_DIR"' EXIT

# DAGMan passes VARS entries to condor_submit as macros. Native dry-runs with the
# same macro names catch submit-command collisions that condor_submit_dag
# -no_submit alone cannot detect.
macros=(
  "TaskId=preflight_submit"
  "PairIndex=0"
  "Stage=prepare"
  "CommandIndex=0"
  "Manifest=configs/validation/cat_cross_seed_confirmatory.yaml"
  "RepoRoot=$HOME/beyond-steering-vectors"
  "SharedRoot=$SHARED_ROOT"
  "DockerImage=$IMAGE"
  "RequestCpus=2"
  "RequestMemoryMB=8192"
  "MinGpuMemoryMB=16384"
  "RetirementSeconds=7200"
)

for template in task_cpu task_gpu finalize; do
  condor_submit -dry-run "$VALIDATION_DIR/$template.classad" \
    "${macros[@]}" "condor/$template.sub" >/dev/null
  test -s "$VALIDATION_DIR/$template.classad"
done

for template in gpu_smoke environment_smoke; do
  condor_submit -dry-run "$VALIDATION_DIR/$template.classad" \
    "condor/$template.sub" >/dev/null
  test -s "$VALIDATION_DIR/$template.classad"
done

echo "Validated all HTCondor node submit files with native condor_submit dry-runs."

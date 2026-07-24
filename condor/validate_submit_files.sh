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
  "SlgeoTaskId=preflight_submit"
  "SlgeoPairIndex=0"
  "SlgeoStage=prepare"
  "SlgeoCommandIndex=0"
  "SlgeoManifestPath=configs/validation/cat_cross_seed_confirmatory.yaml"
  "SlgeoRepoRoot=$HOME/beyond-steering-vectors"
  "SlgeoSharedRoot=$SHARED_ROOT"
  "SlgeoDockerImage=$IMAGE"
  "SlgeoRequestCpus=2"
  "SlgeoRequestMemoryMB=8192"
  "SlgeoMinGpuMemoryMB=16384"
  "SlgeoRetirementSeconds=7200"
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

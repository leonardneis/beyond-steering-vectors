#!/bin/bash
set -euo pipefail
PAIR_INDEX=$1
STAGE=$2
COMMAND_INDEX=$3
source slurm/setup_environment.sh

SCRATCH_ROOT=""
for candidate in "${SLURM_TMPDIR:-}" "${TMPDIR:-}" "${SCRATCH:-}"; do
  if [[ -n "$candidate" && -d "$candidate" && -w "$candidate" ]]; then
    SCRATCH_ROOT=$candidate
    break
  fi
done

PLAN="results/confirmatory/plans/${STAGE}_pair${PAIR_INDEX}_cmd${COMMAND_INDEX}.json"
args=(scripts/run_confirmatory_manifest.py --manifest "$CONFIRMATORY_MANIFEST"
  --pair-index "$PAIR_INDEX" --stage "$STAGE" --command-index "$COMMAND_INDEX"
  --emit-plan "$PLAN" --execute --resume)
if [[ -n "$SCRATCH_ROOT" ]]; then args+=(--scratch-root "$SCRATCH_ROOT"); fi

python -u "${args[@]}"

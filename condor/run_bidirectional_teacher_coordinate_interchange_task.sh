#!/usr/bin/env bash
set -euo pipefail
MODE=$1
CONDITION=$2
MANIFEST=$3
TASK_ID=$4

# shellcheck disable=SC1091
source condor/setup_environment.sh
if [[ -z "${SLGEO_EXECUTION_GIT_COMMIT:-}" || "$SLGEO_EXECUTION_GIT_COMMIT" == "UNFROZEN" ]]; then
  echo "Refusing C18 execution without a frozen commit." >&2
  exit 2
fi
if [[ "$(git rev-parse HEAD)" != "$SLGEO_EXECUTION_GIT_COMMIT" ]] || [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "C18 execution checkout is not the exact clean frozen commit." >&2
  exit 2
fi
ROOT="$SLGEO_SHARED_ROOT/results/research/qwen7b_cat_bidirectional_teacher_coordinate_interchange_v1"
if [[ "$MODE" == "technical_preflight" ]]; then
  mkdir -p "$ROOT/technical"
  python -u scripts/run_bidirectional_teacher_coordinate_interchange_manifest.py \
    --manifest "$MANIFEST" --mode technical --require-runtime-inputs \
    --emit-plan "$ROOT/technical/preflight.json" >/dev/null
elif [[ "$MODE" == "technical" ]]; then
  mkdir -p "$ROOT/technical"
  python -u scripts/validate_bidirectional_teacher_coordinate_interchange.py \
    --manifest "$MANIFEST" --output "$ROOT/technical/validation.json"
elif [[ "$MODE" == "technical_audit" ]]; then
  python -u scripts/audit_c18_technical_validation.py \
    --manifest "$MANIFEST" --validation "$ROOT/technical/validation.json" \
    --preflight "$ROOT/technical/preflight.json" \
    --output "$ROOT/technical/audit.json"
elif [[ "$MODE" == "scientific" ]]; then
  mkdir -p "$ROOT/sealed/raw"
  python -u scripts/run_bidirectional_teacher_coordinate_interchange.py \
    --manifest "$MANIFEST" --condition "$CONDITION" \
    --authorization condor/runtime/c18_scientific_authorization.json \
    --technical-audit "$ROOT/technical/audit.json" \
    --output "$ROOT/sealed/raw/$CONDITION.npz.sealed"
else
  echo "Unknown C18 task mode: $MODE" >&2
  exit 2
fi

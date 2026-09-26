#!/usr/bin/env bash
# Phenotype Anchor v1 submission (submit host). Default is a dry run (validate only, nothing submitted).
#   --technical-validation [--driver X]  TV-P1 DAG (new attempt directory per run tag; at most 3 attempts, 4 A100-h).
#                                        The NVIDIA driver comes from the manifest (execution.nvidia_driver) or, while
#                                        that is still FILL_FROM_TV, from --driver X; the attempt record stores it.
#   --scientific-plan                    scientific plan job (refused unless authorized and the contract is frozen)
#   --scientific                         scientific DAG from the plan (refused unless authorized; budget gate on every node)
#   --submit                             actually submit (otherwise validate only)
set -euo pipefail
MODE=""
SUBMIT=0
DRIVER_ARG=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --technical-validation) MODE=techval; shift ;;
    --scientific-plan) MODE=sciplan; shift ;;
    --scientific) MODE=scientific; shift ;;
    --driver) DRIVER_ARG=${2:?--driver needs a version}; shift 2 ;;
    --submit) SUBMIT=1; shift ;;
    *) echo "Usage: $0 --technical-validation [--driver X]|--scientific-plan|--scientific [--submit]" >&2; exit 2 ;;
  esac
done
[[ -n "$MODE" ]] || { echo "Choose --technical-validation, --scientific-plan or --scientific" >&2; exit 2; }
if [[ -n "$DRIVER_ARG" && "$MODE" != techval ]]; then
  echo "--driver is accepted for the technical validation only; scientific runs use the manifest value." >&2; exit 2
fi
cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd)
if [[ -f condor/condor.env ]]; then
  # shellcheck disable=SC1091
  source condor/condor.env
fi
: "${SLGEO_SHARED_ROOT:?Set SLGEO_SHARED_ROOT (persistent shared storage) in the environment or condor/condor.env}"
SHARED_ROOT=$SLGEO_SHARED_ROOT
CONFIG=configs/validation/phenotype_anchor_v1.yaml
IMAGE="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755"
SCI_ROOT="$SHARED_ROOT/results/research/qwen7b_phenotype_anchor_v1"
TV_BASE="$SCI_ROOT/technical_validation"
ACCOUNTING_ROOT="$SCI_ROOT/accounting"
CAP_FILE="$ACCOUNTING_ROOT/cap.json"
TV_CAP=4
TV_MAX_ATTEMPTS=3
PLACEHOLDER=FILL_FROM_TV
manifest_value() { python3 -B scripts/phenotype_budget.py manifest-value --manifest "$CONFIG" --field "$1"; }

# Code state: clean checkout, pushed commit, prompt manifest matches its pinned hash, image matches the manifest.
if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  echo "Refusing submission from a dirty or untracked worktree." >&2; exit 2
fi
if [[ -z "$(git branch -r --contains HEAD 2>/dev/null)" ]]; then
  echo "Refusing submission of a commit that is not on any remote branch (push first)." >&2; exit 2
fi
python3 - "$CONFIG" <<'PY'
import hashlib, re, sys
text = open(sys.argv[1], encoding="utf-8").read()
path = re.search(r"^\s+prompt_manifest:\s*(\S+)", text, re.M).group(1)
pinned = re.search(r"^\s+prompt_manifest_sha256:\s*([0-9a-f]{64})", text, re.M).group(1)
if hashlib.sha256(open(path, "rb").read()).hexdigest() != pinned:
    sys.exit("Prompt manifest differs from its pinned SHA-256")
PY
if [[ "$(manifest_value execution.container_image)" != "$IMAGE" ]]; then
  echo "Container image differs from the manifest (execution.container_image)." >&2; exit 2
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

# NVIDIA driver pin: never hard-coded; the manifest value, or --driver for the TV while the manifest is a placeholder.
MANIFEST_DRIVER=$(manifest_value execution.nvidia_driver)
if [[ "$MODE" == techval ]]; then
  if [[ "$MANIFEST_DRIVER" != "$PLACEHOLDER" && -n "$MANIFEST_DRIVER" ]]; then
    if [[ -n "$DRIVER_ARG" && "$DRIVER_ARG" != "$MANIFEST_DRIVER" ]]; then
      echo "--driver $DRIVER_ARG contradicts execution.nvidia_driver $MANIFEST_DRIVER." >&2; exit 2
    fi
    DRIVER=$MANIFEST_DRIVER; DRIVER_SOURCE=manifest
  else
    [[ -n "$DRIVER_ARG" ]] || { echo "execution.nvidia_driver is $PLACEHOLDER; give the TV driver pin with --driver X." >&2; exit 2; }
    DRIVER=$DRIVER_ARG; DRIVER_SOURCE=command-line
  fi
else
  DRIVER=$MANIFEST_DRIVER; DRIVER_SOURCE=manifest
fi
if ! [[ "$DRIVER" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
  echo "Invalid NVIDIA driver pin '$DRIVER'." >&2; exit 2
fi

dry_run_submit_files() {
  local dir; dir=$(mktemp -d "${TMPDIR:-/tmp}/bsv-phenotype-submit.XXXXXX")
  local macros=("BsvTaskId=phenotype_dry" "BsvCommand=tv" "BsvTarget=dry" "BsvRepoRoot=$REPO_ROOT"
    "BsvSharedRoot=$SHARED_ROOT" "BsvRequestCpus=4" "BsvRequestMemoryMB=32768"
    "BsvExecutionGitCommit=$EXECUTION_COMMIT" "BsvMachineRequirement=True" "BsvDockerImage=$IMAGE"
    "BsvRunTag=dry" "BsvBudgetCategory=TV" "BsvNvidiaDriver=$DRIVER")
  condor_submit -dry-run "$dir/gpu.classad" "${macros[@]}" condor/phenotype_task_gpu.sub >/dev/null
  condor_submit -dry-run "$dir/cpu.classad" "${macros[@]}" condor/phenotype_task_cpu.sub >/dev/null
  grep -q 'NVIDIA A100-PCIE-40GB' "$dir/gpu.classad"
  grep -qF "$DRIVER" "$dir/gpu.classad"
  grep -q 'BsvBudgetCategory' "$dir/gpu.classad"
  rm -rf "$dir"
}

mkdir -p condor/logs condor/runtime

if [[ "$MODE" == techval ]]; then
  RUN_TAG="tv-$(date -u +%Y%m%dT%H%M%SZ)"
  TV_ROOT="$TV_BASE/$RUN_TAG"
  ATTEMPTS=$(find "$ACCOUNTING_ROOT/tv_attempts" -name '*.json' 2>/dev/null | wc -l)
  if [[ "$ATTEMPTS" -ge "$TV_MAX_ATTEMPTS" ]]; then
    echo "TV attempt limit ($TV_MAX_ATTEMPTS) reached; a dated researcher decision is required." >&2; exit 2
  fi
  RUNTIME_DAG=condor/runtime/phenotype_anchor_v1_technical_validation.dag
  generator=(python3 -B scripts/generate_phenotype_dag.py --technical-validation --output "$RUNTIME_DAG"
    --execution-git-commit "$EXECUTION_COMMIT" --repo-root "$REPO_ROOT" --shared-root "$SHARED_ROOT"
    --run-tag "$RUN_TAG" --out-root "$TV_ROOT" --cap "$TV_CAP" --accounting-root "$ACCOUNTING_ROOT"
    --nvidia-driver "$DRIVER" --start-epoch "$(date +%s)")
  if [[ -n "${NTFY_TOPIC:-}" ]]; then generator+=(--ntfy-topic "$NTFY_TOPIC"); fi
  "${generator[@]}"
  dry_run_submit_files
  condor_submit_dag -no_submit -f "$RUNTIME_DAG" >/dev/null
  if [[ "$SUBMIT" -ne 1 ]]; then
    echo "READY: TV-P1 DAG validated for $EXECUTION_COMMIT (run tag $RUN_TAG, driver $DRIVER from $DRIVER_SOURCE, attempt $((ATTEMPTS + 1))/$TV_MAX_ATTEMPTS); nothing submitted."
    exit 0
  fi
  python3 -B scripts/phenotype_budget.py tv-attempt --accounting-root "$ACCOUNTING_ROOT" --run-tag "$RUN_TAG" \
    --commit "$EXECUTION_COMMIT" --max-attempts "$TV_MAX_ATTEMPTS" --nvidia-driver "$DRIVER" --driver-source "$DRIVER_SOURCE"
  condor_submit_dag -f "$RUNTIME_DAG"
  exit 0
fi

# Scientific modes: frozen contract (tagged), no placeholders, committed researcher authorization with the cap.
if [[ "$(manifest_value contract.status)" != "frozen" ]]; then
  echo "The Phenotype Anchor contract is not frozen (contract.status is draft)." >&2; exit 2
fi
for field in execution.nvidia_driver execution.packages; do
  value=$(manifest_value "$field")
  if [[ -z "$value" || "$value" == "$PLACEHOLDER" ]]; then
    echo "$field is still a $PLACEHOLDER placeholder; record the TV-P1 values first." >&2; exit 2
  fi
done
if ! git rev-parse -q --verify 'prereg/phenotype-anchor-v1^{commit}' >/dev/null; then
  echo "Preregistration tag prereg/phenotype-anchor-v1 missing." >&2; exit 2
fi
if [[ -n "$(git diff --name-only 'prereg/phenotype-anchor-v1' HEAD -- research/phenotype_anchor_v1)" ]]; then
  echo "The preregistration package differs from its tag." >&2; exit 2
fi
AUTH=research/phenotype_anchor_v1/SCIENTIFIC_EXECUTION_AUTHORIZATION.json
if [[ ! -f "$AUTH" ]]; then
  echo "Scientific execution is not authorized (no committed $AUTH)." >&2; exit 2
fi
read -r PROJECTION AUTH_CAP < <(python3 - "$AUTH" <<'PY'
import json, sys
record = json.load(open(sys.argv[1]))
print(record["tv_projection_record"], float(record["cap_a100_h"]))
PY
)
# The cap record is written once at the first authorized submission; later calls verify it (an extension of the cap
# is a researcher decision). A dry run writes nothing.
write_cap() {
  python3 -B scripts/phenotype_budget.py write-cap --projection "$PROJECTION" --cap "$AUTH_CAP" --accounting-root "$ACCOUNTING_ROOT"
}
RUN_TAG="sci-p1"

if [[ "$MODE" == sciplan ]]; then
  if [[ "$SUBMIT" -ne 1 ]]; then echo "READY: scientific plan job validated; nothing submitted."; exit 0; fi
  write_cap
  condor_submit "BsvTaskId=phenotype_plan" "BsvCommand=plan" "BsvTarget=none" "BsvRepoRoot=$REPO_ROOT" \
    "BsvSharedRoot=$SHARED_ROOT" "BsvRequestCpus=2" "BsvRequestMemoryMB=8192" "BsvExecutionGitCommit=$EXECUTION_COMMIT" \
    "BsvMachineRequirement=True" "BsvDockerImage=$IMAGE" "BsvRunTag=$RUN_TAG" "BsvBudgetCategory=SCI" \
    "BsvNvidiaDriver=$DRIVER" condor/phenotype_task_cpu.sub
  exit 0
fi

PLAN="$SCI_ROOT/plan.json"
[[ -f "$PLAN" ]] || { echo "No scientific plan at $PLAN; run --scientific-plan first." >&2; exit 2; }
RUNTIME_DAG=condor/runtime/phenotype_anchor_v1_scientific.dag
generator=(python3 -B scripts/generate_phenotype_dag.py --plan "$PLAN" --output "$RUNTIME_DAG"
  --execution-git-commit "$EXECUTION_COMMIT" --repo-root "$REPO_ROOT" --shared-root "$SHARED_ROOT"
  --run-tag "$RUN_TAG" --out-root "$SCI_ROOT" --cap-file "$CAP_FILE" --nvidia-driver "$DRIVER" --start-epoch "$(date +%s)")
if [[ -n "${NTFY_TOPIC:-}" ]]; then generator+=(--ntfy-topic "$NTFY_TOPIC"); fi
"${generator[@]}"
dry_run_submit_files
condor_submit_dag -no_submit -f "$RUNTIME_DAG" >/dev/null
if [[ "$SUBMIT" -ne 1 ]]; then echo "READY: scientific DAG validated; nothing submitted."; exit 0; fi
write_cap
condor_submit_dag -f "$RUNTIME_DAG"

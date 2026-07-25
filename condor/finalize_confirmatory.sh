#!/usr/bin/env bash
set -euo pipefail
MANIFEST=$1
TASK_ID=$2
# shellcheck disable=SC1091
source condor/setup_environment.sh

ROOT=$(python - "$MANIFEST" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from run_confirmatory_manifest import apply_storage_overrides
from slgeo.io import load_yaml
manifest = apply_storage_overrides(load_yaml(Path(sys.argv[1])))
print(manifest["output_root"])
PY
)
MARKER="$ROOT/orchestration/finalize.complete.json"
AGGREGATE="$ROOT/aggregate.json"
PLOTS="$ROOT/plots"
REPORTS="$ROOT/reports"
CHECKSUMS="$ROOT/final_artifacts.sha256"
mkdir -p "$ROOT/orchestration"
if [[ -f "$MARKER" ]]; then
  echo "Final aggregation already complete: $MARKER"
  exit 0
fi

TMP_BASE=${TMPDIR:-/tmp}
LOCAL_STAGE=$(mktemp -d "$TMP_BASE/slgeo-finalize.XXXXXX")
trap 'rm -rf "$LOCAL_STAGE"' EXIT
PUBLISH_TAG="${CONDOR_CLUSTER_ID:-local}.${CONDOR_PROC_ID:-0}.$$"
CHECKSUMS_INCOMING="$ROOT/final_artifacts.sha256.incoming.$PUBLISH_TAG"

publish_file() {
  local source=$1 incoming=$2 final=$3
  [[ ! -e "$incoming" && ! -e "$final" ]]
  cp --no-preserve=ownership "$source" "$incoming"
  mv "$incoming" "$final"
}

[[ -f "$AGGREGATE" ]] || { echo "Missing existing aggregate: $AGGREGATE" >&2; exit 2; }
[[ -d "$PLOTS" ]] || { echo "Missing existing plots: $PLOTS" >&2; exit 2; }
[[ -d "$REPORTS" ]] || { echo "Missing existing reports: $REPORTS" >&2; exit 2; }
./condor/repair_scratch_group.sh "$ROOT" "${SLGEO_QUOTA_GROUP:-compuling}"

if python -c 'import pytest' >/dev/null 2>&1; then
  python -m pytest -q
else
  echo "pytest is unavailable in the runtime image; running the read-only artifact audit"
  python scripts/audit_confirmatory_artifacts.py --root "$ROOT"
fi
(
  cd "$ROOT"
  find aggregate.json plots reports -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum
) > "$LOCAL_STAGE/final_artifacts.sha256"
publish_file "$LOCAL_STAGE/final_artifacts.sha256" "$CHECKSUMS_INCOMING" "$CHECKSUMS"
python - "$MARKER" "$AGGREGATE" "$CHECKSUMS" "$REPORTS" <<'PY'
import hashlib,json,os,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
marker,aggregate,checksums,reports=map(Path,sys.argv[1:])
digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
payload={"schema_version":1,"status":"complete","ended_at":datetime.now(timezone.utc).isoformat(),
         "condor_cluster_id":os.getenv("CONDOR_CLUSTER_ID"),"condor_proc_id":os.getenv("CONDOR_PROC_ID"),
         "finalizer_script_sha256":digest(Path("condor/finalize_confirmatory.sh")),
         "renderer_script_sha256":digest(Path("scripts/render_confirmatory_final_artifacts.py")),
         "aggregate":str(aggregate),"aggregate_sha256":digest(aggregate),
         "reports":str(reports),
         "checksums":str(checksums),"checksums_sha256":digest(checksums)}
temporary=marker.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
temporary.replace(marker)
PY
python scripts/confirmatory_status.py --manifest "$MANIFEST" --condor

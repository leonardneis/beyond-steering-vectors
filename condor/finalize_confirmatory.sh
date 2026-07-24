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
AGGREGATE_INCOMING="$ROOT/aggregate.incoming.$PUBLISH_TAG"
PLOTS_INCOMING="$ROOT/plots.incoming.$PUBLISH_TAG"
CHECKSUMS_INCOMING="$ROOT/final_artifacts.sha256.incoming.$PUBLISH_TAG"
if [[ ! -f "$AGGREGATE" ]]; then
  python -u scripts/aggregate_confirmatory_seeds.py \
    --seed-roots "$ROOT/seed_1" "$ROOT/seed_2" "$ROOT/seed_3" \
    --output "$LOCAL_STAGE/aggregate.json" --bootstrap-samples 10000
  mv "$LOCAL_STAGE/aggregate.json" "$AGGREGATE_INCOMING"
  mv "$AGGREGATE_INCOMING" "$AGGREGATE"
fi
if [[ ! -d "$PLOTS" ]]; then
  python -u scripts/plot_confirmatory_aggregate.py --input "$AGGREGATE" --output-dir "$LOCAL_STAGE/plots"
  mv "$LOCAL_STAGE/plots" "$PLOTS_INCOMING"
  mv "$PLOTS_INCOMING" "$PLOTS"
fi
./condor/repair_scratch_group.sh "$ROOT" "${SLGEO_QUOTA_GROUP:-compuling}"

python -m pytest -q
sha256sum "$AGGREGATE" "$PLOTS"/*.png > "$LOCAL_STAGE/final_artifacts.sha256"
mv "$LOCAL_STAGE/final_artifacts.sha256" "$CHECKSUMS_INCOMING"
mv "$CHECKSUMS_INCOMING" "$CHECKSUMS"
python - "$MARKER" "$AGGREGATE" "$CHECKSUMS" <<'PY'
import hashlib,json,os,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
marker,aggregate,checksums=map(Path,sys.argv[1:])
digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
payload={"schema_version":1,"status":"complete","ended_at":datetime.now(timezone.utc).isoformat(),
         "condor_cluster_id":os.getenv("CONDOR_CLUSTER_ID"),"condor_proc_id":os.getenv("CONDOR_PROC_ID"),
         "aggregate":str(aggregate),"aggregate_sha256":digest(aggregate),
         "checksums":str(checksums),"checksums_sha256":digest(checksums)}
temporary=marker.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
temporary.replace(marker)
PY
python scripts/confirmatory_status.py --manifest "$MANIFEST" --condor

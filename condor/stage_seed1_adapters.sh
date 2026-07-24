#!/usr/bin/env bash
set -euo pipefail

ARCHIVE=${1:?Usage: stage_seed1_adapters.sh ARCHIVE SHARED_ROOT}
SHARED_ROOT=${2:?Usage: stage_seed1_adapters.sh ARCHIVE SHARED_ROOT}
EXPECTED_SHA256=543ceeb0a3ec4e00cc06eed6243de965612d7f18324fd19586d05f2a91c7b60e
GROUP=${SLGEO_QUOTA_GROUP:-compuling}
TARGET="$SHARED_ROOT/results/reference_reproduction_4080"
INCOMING="$TARGET.incoming"

case "$(realpath "$SHARED_ROOT")" in
  /scratch/*) ;;
  *) echo "Shared root must be below /scratch: $SHARED_ROOT" >&2; exit 2 ;;
esac
actual=$(sha256sum "$ARCHIVE" | awk '{print $1}')
if [[ "$actual" != "$EXPECTED_SHA256" ]]; then
  echo "Adapter archive checksum mismatch: $actual != $EXPECTED_SHA256" >&2
  exit 1
fi

if [[ -e "$INCOMING" ]]; then
  resolved=$(realpath "$INCOMING")
  test "$resolved" = "$TARGET.incoming"
  rm -rf -- "$resolved"
fi
mkdir -p "$INCOMING"
tar -xf "$ARCHIVE" -C "$INCOMING"
./condor/repair_scratch_group.sh "$INCOMING" "$GROUP"

for condition in qwen7b_cat_subliminal_10k_3epochs qwen7b_neutral_10k_3epochs; do
  adapter="$INCOMING/$condition/student_lora"
  test -s "$adapter/adapter_config.json"
  test -s "$adapter/adapter_model.safetensors"
done

if [[ -e "$TARGET" ]]; then
  mv "$TARGET" "$TARGET.partial-$(date -u +%Y%m%dT%H%M%SZ)"
fi
mv "$INCOMING" "$TARGET"
echo "Published checksum-verified Seed-1 adapters to $TARGET"

#!/usr/bin/env bash
set -euo pipefail

SHARED_ROOT=${1:?Usage: stage_qwen_cache.sh SHARED_ROOT [REVISION]}
REVISION=${2:-a09a35458c702b33eeacc393d103063234e8bc28}
MODEL=Qwen/Qwen2.5-7B-Instruct
CACHE_ROOT="$SHARED_ROOT/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
INCOMING="$CACHE_ROOT/snapshots/$REVISION.incoming"
FINAL="$CACHE_ROOT/snapshots/$REVISION"

mkdir -p "$INCOMING" "$CACHE_ROOT/refs"

download() {
  local file=$1
  wget -q -c -O "$INCOMING/$file" \
    "https://huggingface.co/$MODEL/resolve/$REVISION/$file"
}

metadata=(
  config.json
  generation_config.json
  merges.txt
  model.safetensors.index.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
)
shards=(
  model-00001-of-00004.safetensors
  model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors
  model-00004-of-00004.safetensors
)

for file in "${metadata[@]}"; do
  download "$file"
done

pids=()
for file in "${shards[@]}"; do
  download "$file" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

expected=$((${#metadata[@]} + ${#shards[@]}))
actual=$(find "$INCOMING" -maxdepth 1 -type f -size +0c | wc -l)
if [[ "$actual" -ne "$expected" ]]; then
  echo "Expected $expected non-empty files, found $actual in $INCOMING" >&2
  exit 1
fi

if [[ -d "$FINAL" ]]; then
  mv "$FINAL" "$FINAL.partial-$(date -u +%Y%m%dT%H%M%SZ)"
fi
mv "$INCOMING" "$FINAL"
# huggingface_hub compares this value verbatim and does not strip a newline.
printf '%s' "$REVISION" > "$CACHE_ROOT/refs/main.incoming"
mv "$CACHE_ROOT/refs/main.incoming" "$CACHE_ROOT/refs/main"
echo "Published $MODEL@$REVISION to $FINAL"

#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:?Usage: repair_scratch_group.sh SCRATCH_ROOT [QUOTA_GROUP]}
QUOTA_GROUP=${2:-compuling}
case "$(realpath "$ROOT")" in
  /scratch/*) ;;
  *) echo "Refusing to modify a path outside /scratch: $ROOT" >&2; exit 2 ;;
esac
if ! id -nG | tr ' ' '\n' | grep -Fxq "$QUOTA_GROUP"; then
  echo "Current user is not a member of quota group $QUOTA_GROUP" >&2
  exit 2
fi

chgrp -R "$QUOTA_GROUP" "$ROOT"
find "$ROOT" -type d -print0 | xargs -0 --no-run-if-empty chmod g+s

if find "$ROOT" ! -group "$QUOTA_GROUP" -print -quit | grep -q .; then
  echo "Some entries below $ROOT are not owned by group $QUOTA_GROUP" >&2
  exit 1
fi
if find "$ROOT" -type d ! -perm -2000 -print -quit | grep -q .; then
  echo "Some directories below $ROOT do not inherit the quota group" >&2
  exit 1
fi
echo "Normalized group=$QUOTA_GROUP and setgid inheritance below $ROOT"

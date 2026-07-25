#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:?Usage: repair_scratch_group.sh SCRATCH_ROOT [QUOTA_GROUP]}
QUOTA_GROUP=${2:-compuling}
case "$(realpath "$ROOT")" in
  /scratch/*) ;;
  *) echo "Refusing to modify a path outside /scratch: $ROOT" >&2; exit 2 ;;
esac

# LDAP group names are not always resolvable inside SIC Docker jobs even though
# HTCondor passes the corresponding supplementary numeric GIDs through. Prefer
# the named lookup, but safely fall back to the already-established group of the
# explicitly supplied Scratch root.
QUOTA_GID=$(getent group "$QUOTA_GROUP" 2>/dev/null | awk -F: 'NR == 1 {print $3}' || true)
if [[ -z "$QUOTA_GID" && "$QUOTA_GROUP" =~ ^[0-9]+$ ]]; then
  QUOTA_GID=$QUOTA_GROUP
fi
if [[ -z "$QUOTA_GID" ]]; then
  QUOTA_GID=$(stat -c %g "$ROOT")
  echo "Group name $QUOTA_GROUP is unavailable; using Scratch root GID $QUOTA_GID"
fi
if ! id -G | tr ' ' '\n' | grep -Fxq "$QUOTA_GID"; then
  echo "Current user is not a member of quota GID $QUOTA_GID" >&2
  exit 2
fi

chgrp -R "$QUOTA_GID" "$ROOT"
find "$ROOT" -type d -print0 | xargs -0 --no-run-if-empty chmod g+s

if find "$ROOT" ! -gid "$QUOTA_GID" -print -quit | grep -q .; then
  echo "Some entries below $ROOT are not owned by group GID $QUOTA_GID" >&2
  exit 1
fi
if find "$ROOT" -type d ! -perm -2000 -print -quit | grep -q .; then
  echo "Some directories below $ROOT do not inherit the quota group" >&2
  exit 1
fi
echo "Normalized group=$QUOTA_GROUP (GID $QUOTA_GID) and setgid inheritance below $ROOT"

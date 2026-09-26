"""Standard-library-only atomic publication and canonical JSON (usable on the submit host)."""

from __future__ import annotations

from .errors import FinalFailure

import datetime as _dt
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any


class ArtifactError(RuntimeError, FinalFailure):
    pass


def canonical_json(data: Any) -> bytes:
    return (json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode()


def pretty_json(data: Any) -> bytes:
    return (json.dumps(data, sort_keys=True, indent=1, ensure_ascii=True, allow_nan=False) + "\n").encode()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:  # pragma: no cover - Windows cannot open directories
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: str | Path, data: bytes, *, write_once: bool = False) -> str:
    """Publish ``data`` at ``path`` atomically; return its SHA-256. ``write_once`` refuses overwrite."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if write_once and path.exists():
        raise ArtifactError(f"Refusing to overwrite write-once artifact {path}")
    incoming = path.with_name(f"{path.name}.{uuid.uuid4().hex}.incoming")
    expected = sha256_bytes(data)
    with incoming.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if sha256_file(incoming) != expected:
        incoming.unlink(missing_ok=True)
        raise ArtifactError(f"Short or corrupted write detected for {path}")
    if write_once:
        # No check-then-rename window: a hard link creates the final name only if it does not exist yet.
        try:
            os.link(incoming, path)
        except FileExistsError:
            incoming.unlink(missing_ok=True)
            raise ArtifactError(f"Refusing to overwrite write-once artifact {path}") from None
        except OSError:
            # Filesystem without hard links: re-check immediately before the rename (narrow window only).
            if path.exists():
                incoming.unlink(missing_ok=True)
                raise ArtifactError(f"Refusing to overwrite write-once artifact {path}") from None
            os.replace(incoming, path)
        else:
            incoming.unlink()
    else:
        os.replace(incoming, path)
    _fsync_dir(path.parent)
    return expected


def atomic_write_json(path: str | Path, data: Any, *, write_once: bool = False) -> str:
    return atomic_write_bytes(path, pretty_json(data), write_once=write_once)

"""Atomic, hash-verified artifact publication, shard markers, resume and attempt records.

Protocol (pre-implementation audit 5, §3): write to a unique ``.incoming`` name in the destination
directory, flush + fsync, re-read and re-hash, ``os.replace`` to the final name, fsync the directory.
A shard is complete only when its marker exists and every file it lists re-hashes to the marker value;
otherwise the whole shard is quarantined and recomputed. Attempt records are one immutable file each.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import socket
import uuid
import zipfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np


from .atomic import (  # noqa: E402,F401  (re-exported)
    ArtifactError,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json,
    pretty_json,
    sha256_bytes,
    sha256_file,
    utc_now,
)


def npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Byte-deterministic uncompressed npz (sorted keys, fixed zip timestamps; readable by np.load)."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for key in sorted(arrays):
            array = np.asarray(arrays[key])
            if array.dtype == object:
                raise ArtifactError(f"Object arrays are not allowed in artifacts ({key})")
            member = io.BytesIO()
            np.lib.format.write_array(member, array, allow_pickle=False)
            info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            archive.writestr(info, member.getvalue())
    return buffer.getvalue()


def atomic_write_npz(path: str | Path, arrays: Mapping[str, np.ndarray], *, write_once: bool = False) -> str:
    return atomic_write_bytes(path, npz_bytes(arrays), write_once=write_once)


def load_npz_verified(path: str | Path, expected_sha256: str) -> dict[str, np.ndarray]:
    path = Path(path)
    if sha256_file(path) != expected_sha256:
        raise ArtifactError(f"Hash mismatch for {path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_json_verified(path: str | Path, expected_sha256: str | None = None) -> Any:
    path = Path(path)
    raw = path.read_bytes()
    if expected_sha256 is not None and sha256_bytes(raw) != expected_sha256:
        raise ArtifactError(f"Hash mismatch for {path}")
    return json.loads(raw)


class Shard:
    """A shard directory with files and a completion marker."""

    def __init__(self, root: str | Path, shard_id: str, spec_sha256: str, run_identity: Mapping[str, Any]):
        self.root = Path(root)
        self.shard_id = shard_id
        self.directory = self.root / "shards" / shard_id
        self.marker = self.root / "markers" / f"{shard_id}.json"
        self.spec_sha256 = spec_sha256
        self.run_identity = dict(run_identity)

    def is_complete(self) -> bool:
        """True only if the marker verifies completely against the current plan and run identity."""
        if not self.marker.is_file():
            return False
        try:
            marker = json.loads(self.marker.read_bytes())
        except (OSError, json.JSONDecodeError):
            return False
        if not isinstance(marker, dict) or not isinstance(marker.get("files"), dict):
            return False
        if marker.get("shard_id") != self.shard_id or marker.get("spec_sha256") != self.spec_sha256:
            return False
        if marker.get("run_identity") != self.run_identity:
            return False
        try:
            for name, info in marker["files"].items():
                path = self.directory / name
                if not path.is_file() or path.stat().st_size != info["bytes"] or sha256_file(path) != info["sha256"]:
                    return False
        except (KeyError, TypeError, AttributeError):
            return False
        return bool(marker["files"])

    def quarantine(self) -> Path | None:
        """Move any partial state of this shard aside (never deleted, never reused)."""
        targets = [path for path in (self.directory, self.marker) if path.exists()]
        if not targets:
            return None
        stamp = utc_now().replace(":", "").replace(".", "")
        destination = self.root / "quarantine" / f"{stamp}-{uuid.uuid4().hex[:8]}" / self.shard_id
        destination.mkdir(parents=True, exist_ok=False)
        for path in targets:
            shutil.move(str(path), str(destination / path.name))
        return destination

    def publish(self, files: Mapping[str, bytes], extra: Mapping[str, Any]) -> dict[str, Any]:
        """Write every file atomically, then the marker last."""
        if self.marker.exists():
            raise ArtifactError(f"Shard {self.shard_id} already has a marker")
        listing = {}
        for name, data in sorted(files.items()):
            digest = atomic_write_bytes(self.directory / name, data, write_once=True)
            listing[name] = {"sha256": digest, "bytes": len(data)}
        marker = {
            "shard_id": self.shard_id,
            "spec_sha256": self.spec_sha256,
            "run_identity": self.run_identity,
            "files": listing,
            "completed_utc": utc_now(),
            **dict(extra),
        }
        atomic_write_json(self.marker, marker, write_once=True)
        return marker


def attempt_record(root: str | Path, shard_id: str, payload: Mapping[str, Any]) -> Path:
    """One immutable JSON per attempt event under ``orchestration/attempts/<shard>/``."""
    cluster = os.environ.get("CONDOR_CLUSTER_ID", "local")
    proc = os.environ.get("CONDOR_PROC_ID", "0")
    name = f"{utc_now().replace(':', '')}.{cluster}.{proc}.{uuid.uuid4().hex[:8]}.json"
    path = Path(root) / "orchestration" / "attempts" / shard_id / name
    atomic_write_json(path, {"shard_id": shard_id, "host": socket.gethostname(), **dict(payload)}, write_once=True)
    return path

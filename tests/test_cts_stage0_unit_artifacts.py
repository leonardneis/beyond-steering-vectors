from __future__ import annotations

from pathlib import Path
import io
import json
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.cts_stage0 import artifacts as art
from slgeo.cts_stage0 import synthetic
from slgeo.cts_stage0.artifacts import ArtifactError, Shard

IDENTITY = {"execution_commit": "abc", "manifest_sha256": "m"}


def test_atomic_write_returns_hash_and_leaves_no_incoming(tmp_path):
    path = tmp_path / "a" / "b" / "x.bin"
    digest = art.atomic_write_bytes(path, b"hello")
    assert path.read_bytes() == b"hello"
    assert digest == art.sha256_bytes(b"hello") == art.sha256_file(path)
    assert not list(tmp_path.rglob("*.incoming"))
    art.atomic_write_bytes(path, b"again")  # overwrite allowed without write_once
    assert path.read_bytes() == b"again"


def test_write_once(tmp_path):
    path = tmp_path / "once.json"
    art.atomic_write_json(path, {"a": 1}, write_once=True)
    before = path.read_bytes()
    with pytest.raises(ArtifactError, match="write-once"):
        art.atomic_write_json(path, {"a": 2}, write_once=True)
    assert path.read_bytes() == before
    assert not list(tmp_path.rglob("*.incoming"))


def test_json_serializers_reject_nan_and_are_canonical():
    assert art.canonical_json({"b": 1, "a": [1, 2]}) == b'{"a":[1,2],"b":1}\n'
    with pytest.raises(ValueError):
        art.canonical_json({"x": float("nan")})
    with pytest.raises(ValueError):
        art.pretty_json({"x": float("inf")})


def test_npz_bytes_deterministic_and_loadable(tmp_path):
    arrays = {"z": np.arange(6.0).reshape(2, 3), "a": np.array(["p1", "p2"]), "i": np.array([1, 2], dtype=np.int64)}
    first = art.npz_bytes(arrays)
    second = art.npz_bytes(dict(reversed(list(arrays.items()))))
    assert first == second
    with np.load(io.BytesIO(first), allow_pickle=False) as data:
        assert sorted(data.files) == ["a", "i", "z"]
        np.testing.assert_array_equal(data["z"], arrays["z"])
        assert data["a"].tolist() == ["p1", "p2"]
    path = tmp_path / "x.npz"
    digest = art.atomic_write_npz(path, arrays)
    loaded = art.load_npz_verified(path, digest)
    np.testing.assert_array_equal(loaded["i"], arrays["i"])
    with pytest.raises(ArtifactError, match="Hash mismatch"):
        art.load_npz_verified(path, "0" * 64)
    with pytest.raises(ArtifactError, match="Object arrays"):
        art.npz_bytes({"o": np.array([{"a": 1}], dtype=object)})


def test_load_json_verified(tmp_path):
    path = tmp_path / "x.json"
    digest = art.atomic_write_json(path, {"k": [1, 2]})
    assert art.load_json_verified(path, digest) == {"k": [1, 2]}
    assert art.load_json_verified(path) == {"k": [1, 2]}
    with pytest.raises(ArtifactError):
        art.load_json_verified(path, "f" * 64)


def _published(tmp_path) -> Shard:
    shard = Shard(tmp_path, "s1", "spec", IDENTITY)
    shard.publish({"data.npz": art.npz_bytes({"x": np.arange(3.0)}), "info.json": art.pretty_json({"n": 3})}, {"stage": "t"})
    return shard


def test_shard_complete_after_publish(tmp_path):
    shard = Shard(tmp_path, "s1", "spec", IDENTITY)
    assert not shard.is_complete()
    marker = _published(tmp_path).marker
    assert shard.is_complete()
    content = json.loads(marker.read_bytes())
    assert content["run_identity"] == IDENTITY and set(content["files"]) == {"data.npz", "info.json"}
    assert content["stage"] == "t"
    with pytest.raises(ArtifactError, match="already has a marker"):
        shard.publish({"data.npz": b"x"}, {})


def test_shard_partial_and_quarantine(tmp_path):
    shard = Shard(tmp_path, "s1", "spec", IDENTITY)
    shard.directory.mkdir(parents=True)
    (shard.directory / "partial.npz").write_bytes(b"truncated")
    assert not shard.is_complete()
    destination = shard.quarantine()
    assert destination is not None and (destination / "s1" / "partial.npz").is_file()
    assert not shard.directory.exists()
    assert shard.quarantine() is None
    _published(tmp_path)
    assert shard.is_complete()


def test_shard_corruption_detected(tmp_path):
    shard = _published(tmp_path)
    path = shard.directory / "data.npz"
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF  # same size, different hash
    path.write_bytes(bytes(data))
    assert not shard.is_complete()


def test_shard_truncation_and_missing_file_detected(tmp_path):
    shard = _published(tmp_path)
    (shard.directory / "info.json").write_bytes(b"{}")
    assert not shard.is_complete()
    shard2 = _published(tmp_path / "second")
    (shard2.directory / "data.npz").unlink()
    assert not shard2.is_complete()


def test_shard_other_identity_or_spec_not_complete(tmp_path):
    _published(tmp_path)
    assert not Shard(tmp_path, "s1", "spec", {**IDENTITY, "execution_commit": "other"}).is_complete()
    assert not Shard(tmp_path, "s1", "other-spec", IDENTITY).is_complete()
    assert Shard(tmp_path, "s1", "spec", dict(IDENTITY)).is_complete()


def test_corrupt_marker_not_complete(tmp_path):
    shard = _published(tmp_path)
    shard.marker.write_bytes(b"{not json")
    assert not shard.is_complete()


def test_malformed_marker_entry_not_complete(tmp_path):
    shard = _published(tmp_path)
    marker = json.loads(shard.marker.read_bytes())
    del marker["files"]["data.npz"]["bytes"]
    shard.marker.write_bytes(json.dumps(marker).encode())
    assert shard.is_complete() is False


def test_marker_without_files_not_complete(tmp_path):
    shard = Shard(tmp_path, "empty", "spec", IDENTITY)
    shard.publish({}, {})
    assert not shard.is_complete()


def test_attempt_records_distinct_and_immutable(tmp_path, monkeypatch):
    monkeypatch.setenv("CONDOR_CLUSTER_ID", "77")
    monkeypatch.setenv("CONDOR_PROC_ID", "3")
    first = art.attempt_record(tmp_path, "s1", {"event": "start"})
    second = art.attempt_record(tmp_path, "s1", {"event": "start"})
    assert first != second and first.is_file() and second.is_file()
    assert first.parent == tmp_path / "orchestration" / "attempts" / "s1"
    assert ".77.3." in first.name
    record = json.loads(first.read_bytes())
    assert record["shard_id"] == "s1" and record["event"] == "start" and "host" in record
    with pytest.raises(ArtifactError):
        art.atomic_write_json(first, {"event": "overwrite"}, write_once=True)
    assert json.loads(first.read_bytes())["event"] == "start"


def test_synthetic_artifact_drill(tmp_path):
    result = synthetic.artifact_drill(tmp_path)
    assert result["pass"], result["checks"]
    assert all(result["checks"].values())

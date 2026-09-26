from __future__ import annotations

from pathlib import Path
import hashlib
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.cts_stage0 import provenance as pv
from slgeo.cts_stage0.provenance import ProvenanceError

HAS_GIT = shutil.which("git") is not None


@pytest.mark.skipif(not HAS_GIT, reason="git not installed")
@pytest.mark.parametrize("content", [b"", b"print('x')\n", b"a\r\nb\r\n", bytes(range(256))])
def test_git_blob_sha1_matches_git_hash_object(tmp_path, content):
    path = tmp_path / "f.bin"
    path.write_bytes(content)
    expected = subprocess.run(
        ["git", "hash-object", "--no-filters", str(path)], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert pv.git_blob_sha1(content) == expected


@pytest.mark.skipif(not HAS_GIT, reason="git not installed")
def test_git_blob_sha1_matches_index_of_tracked_file():
    listing = subprocess.run(
        ["git", "ls-files", "-s", "--", "src/slgeo/training.py"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout
    blob = listing.split()[1]
    dirty = subprocess.run(["git", "diff", "--quiet", "--", "src/slgeo/training.py"], cwd=ROOT).returncode
    if dirty:
        pytest.skip("training.py modified in the working tree")
    raw = subprocess.run(["git", "cat-file", "blob", blob], cwd=ROOT, capture_output=True, check=True).stdout
    assert pv.git_blob_sha1(raw) == blob


def _repo(tmp_path: Path) -> tuple[Path, list[dict]]:
    files = {
        "src/slgeo/a.py": b"A = 1\n",
        "src/slgeo/cts_stage0/b.py": b"B = 2\n",
        "scripts/run.py": b"print('run')\n",
        "configs/validation/c.yaml": b"k: v\n",
    }
    tracked = []
    for relative, data in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        tracked.append({"path": relative, "mode": "100644", "blob_sha1": pv.git_blob_sha1(data)})
    return tmp_path, tracked


def test_verify_tracked_blobs_clean(tmp_path):
    root, tracked = _repo(tmp_path)
    # __pycache__ files are ignored.
    (root / "src" / "slgeo" / "__pycache__").mkdir()
    (root / "src" / "slgeo" / "__pycache__" / "junk.py").write_bytes(b"x")
    assert pv.verify_tracked_blobs(root, tracked) == len(tracked)


def test_verify_tracked_blobs_detects_modified_file(tmp_path):
    root, tracked = _repo(tmp_path)
    (root / "src" / "slgeo" / "cts_stage0" / "b.py").write_bytes(b"B = 3\n")
    with pytest.raises(ProvenanceError, match="differs from the execution commit"):
        pv.verify_tracked_blobs(root, tracked)


def test_verify_tracked_blobs_detects_crlf_rewrite(tmp_path):
    root, tracked = _repo(tmp_path)
    (root / "configs" / "validation" / "c.yaml").write_bytes(b"k: v\r\n")
    with pytest.raises(ProvenanceError):
        pv.verify_tracked_blobs(root, tracked)


def test_verify_tracked_blobs_detects_untracked_py(tmp_path):
    root, tracked = _repo(tmp_path)
    (root / "src" / "slgeo" / "cts_stage0" / "extra.py").write_bytes(b"X = 1\n")
    with pytest.raises(ProvenanceError, match="Untracked source file"):
        pv.verify_tracked_blobs(root, tracked)
    (root / "src" / "slgeo" / "cts_stage0" / "extra.py").unlink()
    (root / "scripts" / "helper.py").write_bytes(b"")
    with pytest.raises(ProvenanceError, match="Untracked source file"):
        pv.verify_tracked_blobs(root, tracked)


def test_verify_tracked_blobs_detects_missing_file(tmp_path):
    root, tracked = _repo(tmp_path)
    (root / "configs" / "validation" / "c.yaml").unlink()
    with pytest.raises(ProvenanceError, match="missing"):
        pv.verify_tracked_blobs(root, tracked)


def test_verify_tracked_blobs_skips_symlinks(tmp_path):
    root, tracked = _repo(tmp_path)
    tracked.append({"path": "configs/link.yaml", "mode": "120000", "blob_sha1": "0" * 40})
    assert pv.verify_tracked_blobs(root, tracked) == len(tracked)


def test_lf_sha256_crlf_independent(tmp_path):
    lf, crlf = tmp_path / "lf.yaml", tmp_path / "crlf.yaml"
    lf.write_bytes(b"a: 1\nb: 2\n")
    crlf.write_bytes(b"a: 1\r\nb: 2\r\n")
    assert pv.lf_sha256(lf) == pv.lf_sha256(crlf) == hashlib.sha256(b"a: 1\nb: 2\n").hexdigest()
    other = tmp_path / "other.yaml"
    other.write_bytes(b"a: 1\nb: 3\n")
    assert pv.lf_sha256(other) != pv.lf_sha256(lf)


def test_checked_prefixes_cover_code_and_package():
    for prefix in ("src/", "scripts/", "configs/", "research/cts_stage0_v1/", "research/cts_stage0_v1_execution/"):
        assert prefix in pv.CHECKED_PREFIXES

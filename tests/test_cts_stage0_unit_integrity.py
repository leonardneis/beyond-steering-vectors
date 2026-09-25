"""D/C leak scan. D/C fingerprints are used only inside assertions on counts; no text is printed."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.cts_stage0 import artifacts as art
from slgeo.cts_stage0.integrity import dc_leak_scan
from slgeo.cts_stage0.package import FrozenPackage


@pytest.fixture(scope="module")
def package() -> FrozenPackage:
    return FrozenPackage.from_repo(ROOT)


@pytest.fixture(scope="module")
def fingerprints(package):
    return package.dc_fingerprints()


def _clean(root: Path, package: FrozenPackage) -> None:
    s0 = sorted(package.partition_ids()["S0"])
    art.atomic_write_json(root / "a" / "report.json", {"prompt_ids": s0, "values": [1.0, 2.0]})
    art.atomic_write_npz(root / "b" / "scores.npz", {"prompt_ids": np.array(s0), "x": np.zeros((2, 3))})
    (root / "log.txt").write_text("stage ok\n", encoding="utf-8")


def _scan(root: Path, fingerprints) -> dict:
    return dc_leak_scan([root], fingerprints)


def test_fingerprints_shape(fingerprints, package):
    ids = package.partition_ids()
    assert fingerprints["ids"] == set(ids["D"]) | set(ids["C"])
    assert len(fingerprints["sha256"]) == 666
    assert not fingerprints["ids"] & set(ids["S0"])


def test_clean_outputs_pass(tmp_path, fingerprints, package):
    _clean(tmp_path, package)
    result = _scan(tmp_path, fingerprints)
    assert result == {"files_scanned": 3, "files_with_hits": 0, "unscannable_files": 0, "leftover_incoming_files": 0, "pass": True}


def test_empty_output_does_not_pass(tmp_path, fingerprints):
    assert _scan(tmp_path, fingerprints)["pass"] is False
    assert dc_leak_scan([tmp_path / "missing"], fingerprints)["files_scanned"] == 0


def test_planted_d_id_in_json(tmp_path, fingerprints, package):
    _clean(tmp_path, package)
    d_id = sorted(package.partition_ids()["D"])[0]
    art.atomic_write_json(tmp_path / "leak.json", {"ids": ["x", d_id]})
    result = _scan(tmp_path, fingerprints)
    assert result["files_with_hits"] == 1 and result["pass"] is False


def test_planted_c_id_in_text_log(tmp_path, fingerprints, package):
    _clean(tmp_path, package)
    c_id = sorted(package.partition_ids()["C"])[-1]
    (tmp_path / "job.log").write_text(f"scoring {c_id} now\n", encoding="utf-8")
    assert _scan(tmp_path, fingerprints)["files_with_hits"] == 1


def test_planted_d_sha256(tmp_path, fingerprints, package):
    _clean(tmp_path, package)
    digest = sorted(fingerprints["sha256"])[0]
    art.atomic_write_json(tmp_path / "leak.json", {"hash": digest})
    assert _scan(tmp_path, fingerprints)["files_with_hits"] == 1


def test_planted_d_text_in_json(tmp_path, fingerprints, package):
    _clean(tmp_path, package)
    text = sorted(fingerprints["texts"])[0]
    art.atomic_write_json(tmp_path / "leak.json", {"note": "prefix " + text + " suffix"})
    assert _scan(tmp_path, fingerprints)["files_with_hits"] == 1


def test_planted_d_text_in_npz_string_array(tmp_path, fingerprints, package):
    _clean(tmp_path, package)
    text = sorted(fingerprints["texts"])[-1]
    art.atomic_write_npz(tmp_path / "leak.npz", {"x": np.zeros(2), "names": np.array(["ok", text])})
    result = _scan(tmp_path, fingerprints)
    assert result["files_with_hits"] == 1 and result["files_scanned"] == 4


def test_planted_d_id_as_npz_key(tmp_path, fingerprints, package):
    _clean(tmp_path, package)
    d_id = sorted(package.partition_ids()["D"])[1]
    art.atomic_write_npz(tmp_path / "leak.npz", {d_id: np.zeros(2)})
    assert _scan(tmp_path, fingerprints)["files_with_hits"] == 1


def test_s0_ids_are_not_flagged(tmp_path, fingerprints, package):
    s0 = sorted(package.partition_ids()["S0"])
    (tmp_path / "s0.txt").write_text(" ".join(s0), encoding="utf-8")
    assert _scan(tmp_path, fingerprints)["files_with_hits"] == 0


def test_id_match_is_whole_token(tmp_path, fingerprints, package):
    d_id = sorted(package.partition_ids()["D"])[0]
    (tmp_path / "near.txt").write_text(f"x{d_id}9", encoding="utf-8")
    assert _scan(tmp_path, fingerprints)["files_with_hits"] == 0

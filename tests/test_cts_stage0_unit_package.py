from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.cts_stage0 import package as pkgmod
from slgeo.cts_stage0.package import (
    ANIMAL_FAMILIES,
    FrozenPackage,
    FrozenPackageError,
    list_hash,
    load_extraction_prompts,
    sha256_path,
    sha256_text,
)

PACKAGE_DIR = ROOT / pkgmod.PACKAGE_RELATIVE_PATH
EXTRACTION_FILE = ROOT / "data" / "generated" / "reference_qwen7b_cat_subliminal_30k.jsonl"


@pytest.fixture(scope="module")
def package() -> FrozenPackage:
    return FrozenPackage.from_repo(ROOT)


def _copy(tmp_path: Path) -> Path:
    target = tmp_path / "pkg"
    shutil.copytree(PACKAGE_DIR, target, ignore=shutil.ignore_patterns("__pycache__"))
    return target


def _regenerate_manifest(root: Path, monkeypatch) -> None:
    """Rewrite MANIFEST.json consistently with the (modified) copy and pin its hash in the module."""
    manifest_path = root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative in manifest["files"]:
        manifest["files"][relative] = sha256_path(root / relative)
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    monkeypatch.setattr(pkgmod, "MANIFEST_SHA256", sha256_path(manifest_path))


def _rewrite_prompt_line(root: Path, predicate, mutate) -> str:
    """Mutate the first prompts.jsonl row matching ``predicate``; other lines stay byte-identical."""
    path = root / "cts_stage0_prompts.jsonl"
    lines = path.read_bytes().split(b"\n")
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        row = json.loads(line)
        if predicate(row):
            mutate(row)
            lines[index] = json.dumps(row, ensure_ascii=False).encode("utf-8")
            path.write_bytes(b"\n".join(lines))
            return row["prompt_id"]
    raise AssertionError("no row matched")


# --- MANIFEST verification -------------------------------------------------------------------------


def test_real_package_verifies(package):
    assert package.root == PACKAGE_DIR.resolve()
    assert sha256_path(PACKAGE_DIR / "MANIFEST.json") == pkgmod.MANIFEST_SHA256
    assert set(package.file_hashes) == set(package.manifest["files"])
    assert "cts_stage0_decision_spec.json" in package.file_hashes


def test_unmodified_copy_verifies(tmp_path):
    FrozenPackage(_copy(tmp_path))


def test_flipped_byte_raises(tmp_path):
    root = _copy(tmp_path)
    target = root / "DECISION_MATRIX.md"
    data = bytearray(target.read_bytes())
    data[10] ^= 0x01
    target.write_bytes(bytes(data))
    with pytest.raises(FrozenPackageError, match="hash mismatch"):
        FrozenPackage(root)


def test_flipped_byte_in_spec_raises(tmp_path):
    root = _copy(tmp_path)
    target = root / "cts_stage0_decision_spec.json"
    data = bytearray(target.read_bytes())
    data[-3] ^= 0x20
    target.write_bytes(bytes(data))
    with pytest.raises(FrozenPackageError):
        FrozenPackage(root)


def test_extra_file_raises(tmp_path):
    root = _copy(tmp_path)
    (root / "extra.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FrozenPackageError, match="extra"):
        FrozenPackage(root)


def test_extra_file_in_subdirectory_raises(tmp_path):
    root = _copy(tmp_path)
    (root / "tools" / "helper.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(FrozenPackageError, match="extra"):
        FrozenPackage(root)


def test_missing_file_raises(tmp_path):
    root = _copy(tmp_path)
    (root / "PROMPT_AUTHORING.md").unlink()
    with pytest.raises(FrozenPackageError, match="missing"):
        FrozenPackage(root)


def test_missing_manifest_raises(tmp_path):
    root = _copy(tmp_path)
    (root / "MANIFEST.json").unlink()
    with pytest.raises(FrozenPackageError, match="MANIFEST.json missing"):
        FrozenPackage(root)


def test_modified_manifest_raises(tmp_path):
    root = _copy(tmp_path)
    path = root / "MANIFEST.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(FrozenPackageError, match="frozen manifest hash"):
        FrozenPackage(root)


def test_regenerated_manifest_without_patched_hash_raises(tmp_path, monkeypatch):
    root = _copy(tmp_path)
    (root / "DECISION_MATRIX.md").write_bytes(b"changed")
    _regenerate_manifest(root, monkeypatch)
    monkeypatch.undo()  # the pinned MANIFEST_SHA256 is the real one again
    with pytest.raises(FrozenPackageError, match="frozen manifest hash"):
        FrozenPackage(root)


# --- S0 prompts ------------------------------------------------------------------------------------


def test_s0_prompts(package):
    records = package.s0_prompts()
    assert len(records) == 334
    ids = [record.prompt_id for record in records]
    assert ids == sorted(ids)
    assert len(set(ids)) == 334
    assert sum(record.is_animal_family for record in records) == 300
    for family in ANIMAL_FAMILIES:
        assert sum(record.subfamily == family for record in records) == 100
    assert sorted(ids) == sorted(package.partition_ids()["S0"])
    assert all(sha256_text(record.prompt) == record.sha256 for record in records)
    assert list_hash([r.sha256 for r in records]) == package.partition["sets"]["S0"]["prompt_sha256_list_sha256"]


def test_partition_ids_disjoint(package):
    ids = package.partition_ids()
    assert {k: len(v) for k, v in ids.items()} == {"S0": 334, "D": 333, "C": 333}
    assert not (set(ids["S0"]) & set(ids["D"]))
    assert not (set(ids["S0"]) & set(ids["C"]))
    assert not (set(ids["D"]) & set(ids["C"]))


def test_d_row_marked_s0_hits_manifest_guard(tmp_path):
    root = _copy(tmp_path)
    d_ids = set(FrozenPackage(root).partition_ids()["D"])
    _rewrite_prompt_line(root, lambda row: row["prompt_id"] in d_ids, lambda row: row.__setitem__("set", "S0"))
    with pytest.raises(FrozenPackageError, match="cts_stage0_prompts.jsonl"):
        FrozenPackage(root)


def test_d_row_marked_s0_hits_s0_list_guard(tmp_path, monkeypatch):
    root = _copy(tmp_path)
    d_ids = set(FrozenPackage(root).partition_ids()["D"])
    _rewrite_prompt_line(root, lambda row: row["prompt_id"] in d_ids, lambda row: row.__setitem__("set", "S0"))
    _regenerate_manifest(root, monkeypatch)
    copy = FrozenPackage(root)  # manifest now consistent
    with pytest.raises(FrozenPackageError, match="S0 row not in the S0 partition list"):
        copy.s0_prompts()


def test_s0_text_change_hits_prompt_hash_guard(tmp_path, monkeypatch):
    root = _copy(tmp_path)
    _rewrite_prompt_line(root, lambda row: row["set"] == "S0", lambda row: row.__setitem__("prompt", row["prompt"] + " x"))
    _regenerate_manifest(root, monkeypatch)
    with pytest.raises(FrozenPackageError, match="Prompt hash mismatch"):
        FrozenPackage(root).s0_prompts()


def test_s0_row_relabelled_d_hits_guard(tmp_path, monkeypatch):
    root = _copy(tmp_path)
    _rewrite_prompt_line(root, lambda row: row["set"] == "S0", lambda row: row.__setitem__("set", "D"))
    _regenerate_manifest(root, monkeypatch)
    with pytest.raises(FrozenPackageError):
        FrozenPackage(root).s0_prompts()


# --- V prompts, personas ---------------------------------------------------------------------------


def test_validation_prompts(package):
    records = package.validation_prompts()
    assert len(records) == 40
    ids = [r.prompt_id for r in records]
    assert ids == sorted(ids) and len(set(ids)) == 40
    assert not set(ids) & set(package.partition_ids()["S0"])


def test_validation_hash_guard(tmp_path, monkeypatch):
    root = _copy(tmp_path)
    path = root / "cts_stage0_validation_prompts.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    row = json.loads(lines[0])
    row["prompt"] = row["prompt"] + " x"
    lines[0] = json.dumps(row, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _regenerate_manifest(root, monkeypatch)
    with pytest.raises(FrozenPackageError, match="V prompt hash mismatch"):
        FrozenPackage(root).validation_prompts()


def test_personas(package):
    assert len(package.personas) == 44
    assert package.persona("P_default")["system_prompt"] is None
    with pytest.raises(FrozenPackageError, match="Unknown persona"):
        package.persona("P_nope")


# --- extraction prompts ----------------------------------------------------------------------------


def _write_rows(path: Path, prompts: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for prompt in prompts:
            handle.write(json.dumps({"prompt": prompt, "completion": "0"}) + "\n")


def _fresh_package_with_extraction(monkeypatch, **overrides) -> FrozenPackage:
    fresh = FrozenPackage.from_repo(ROOT)
    spec = dict(fresh.manifest["external_inputs"]["extraction_prompts"])
    spec.update(overrides)
    fresh.manifest = json.loads(json.dumps(fresh.manifest))
    fresh.manifest["external_inputs"]["extraction_prompts"] = spec
    return fresh


def test_extraction_file_hash_mismatch(tmp_path, package):
    path = tmp_path / "extraction.jsonl"
    _write_rows(path, [f"p{i}" for i in range(1024)])
    with pytest.raises(FrozenPackageError, match="Extraction file hash mismatch"):
        load_extraction_prompts(package, path)


def test_extraction_list_hash_mismatch(tmp_path, monkeypatch):
    path = tmp_path / "extraction.jsonl"
    _write_rows(path, [f"p{i}" for i in range(1030)])
    fresh = _fresh_package_with_extraction(monkeypatch, file_sha256=sha256_path(path))
    with pytest.raises(FrozenPackageError, match="list hash mismatch"):
        load_extraction_prompts(fresh, path)


def test_extraction_too_few_rows(tmp_path, monkeypatch):
    path = tmp_path / "extraction.jsonl"
    _write_rows(path, [f"p{i}" for i in range(10)])
    fresh = _fresh_package_with_extraction(monkeypatch, file_sha256=sha256_path(path))
    with pytest.raises(FrozenPackageError, match="fewer than 1024"):
        load_extraction_prompts(fresh, path)


def test_extraction_consistent_tmp_file_loads_first_1024_rows(tmp_path, monkeypatch):
    prompts = [f"p{i}" for i in range(1030)]
    path = tmp_path / "extraction.jsonl"
    _write_rows(path, prompts)
    expected_hash = hashlib.sha256(json.dumps(prompts[:1024], ensure_ascii=False).encode("utf-8")).hexdigest()
    fresh = _fresh_package_with_extraction(monkeypatch, file_sha256=sha256_path(path), prompts_json_sha256=expected_hash)
    assert load_extraction_prompts(fresh, path) == tuple(prompts[:1024])


@pytest.mark.skipif(not EXTRACTION_FILE.is_file(), reason="extraction file not staged")
def test_real_extraction_file(package):
    prompts = load_extraction_prompts(package, EXTRACTION_FILE)
    assert len(prompts) == 1024
    assert all(isinstance(p, str) and p for p in prompts)

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from slgeo.analysis.c18_v2_authorization import (
    AUTHORIZATION_DECISION, AUTHORIZATION_RELATIVE_PATH, BLOCKED_EXECUTION_BASE_COMMIT,
    EXECUTION_CONTROL_PATHS, FROZEN_EXECUTION_COMMIT, SUCCESSOR_RELATIONSHIP, TECHNICAL_BINDINGS,
    _verify_execution_successor, canonical_scientific_paths, validate_scientific_authorization,
    validate_sealing_preconditions,
)
from slgeo.analysis.c18_v2_execution import EXPERIMENT_ID
from slgeo.analysis.c18_v2_manifest import (
    apply_storage_overrides, resolve_scientific_artifact, sha256_file, tree_digest,
)


ROOT = Path(__file__).resolve().parents[1]
EXECUTION_COMMIT = "1" * 40


def write(path: Path, content: bytes = b"frozen\n") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def fixture(tmp_path: Path) -> tuple[dict, dict, Path, Path]:
    historical = ROOT / "research/bidirectional_teacher_coordinate_interchange_v2/SCIENTIFIC_EXECUTION_AUTHORIZATION.json"
    target = tmp_path / "research/bidirectional_teacher_coordinate_interchange_v2/SCIENTIFIC_EXECUTION_AUTHORIZATION.json"
    write(target, historical.read_bytes())
    public = {}
    for label in ("preregistration", "decision_matrix", "calibration_summary"):
        path = tmp_path / f"{label}.md"
        public[label] = {"path": path.name, "sha256": write(path, label.encode())}
    frozen = {}
    for label in ("model_config", "prompt_file", "token_inventory", "batch_plan",
                  "selection_plan", "teacher_tensor", "orthogonal_directions"):
        path = tmp_path / f"{label}.bin"
        frozen[label] = {"path": path.name, "sha256": write(path, label.encode())}
    frozen["model"] = {"revision": "a" * 40}
    frozen["adapters"] = {}
    for condition in ("subliminal", "neutral"):
        path = tmp_path / f"adapter_{condition}"
        write(path / "adapter_config.json", condition.encode())
        frozen["adapters"][condition] = {"path": path.name, "tree_sha256": tree_digest(path)}
    frozen["fsd"] = {}
    for label in ("manifest", "subliminal_states", "neutral_states", "aggregate"):
        path = tmp_path / f"fsd_{label}.bin"
        frozen["fsd"][label] = {"path": path.name, "sha256": write(path, label.encode())}
    lock = tmp_path / "requirements.txt"
    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "public_contract": public,
        "frozen_inputs": frozen,
        "calibration": {
            "contract_sha256": "a" * 64, "population_sha256": "b" * 64,
            "seal_sha256": "c" * 64, "audit_sha256": "d" * 64,
        },
        "execution": {
            "container_image": "example/image@sha256:" + "e" * 64,
            "requirements_lock": {"path": lock.name, "sha256": write(lock, b"locked")},
        },
    }
    manifest_path = tmp_path / "manifest.yaml"
    write(manifest_path, b"frozen manifest\n")
    technical = tmp_path / "technical"
    documents = {
        "preflight.json": {"execution_commit": EXECUTION_COMMIT},
        "validation.json": {"execution_git_commit": EXECUTION_COMMIT},
        "validation.json.provenance.json": {},
        "audit.json": {"status": "PASS", "manifest_sha256": sha256_file(manifest_path),
                       "execution_commit": EXECUTION_COMMIT},
    }
    for name, document in documents.items():
        write(technical / name, json.dumps(document).encode())
    write(technical / "SHA256SUMS", b"synthetic technical bundle\n")
    bindings = {
        "preregistration_sha256": public["preregistration"]["sha256"],
        "decision_matrix_sha256": public["decision_matrix"]["sha256"],
        "manifest_sha256": sha256_file(manifest_path),
        "calibration_summary_sha256": public["calibration_summary"]["sha256"],
        "calibration_contract_sha256": "a" * 64,
        "calibration_population_sha256": "b" * 64,
        "calibration_seal_sha256": "c" * 64,
        "calibration_audit_sha256": "d" * 64,
        **{key: sha256_file(technical / name) for key, name in TECHNICAL_BINDINGS.items()},
    }
    inputs = {
        "model_config_sha256": frozen["model_config"]["sha256"],
        "model_revision": "a" * 40,
        **{f"{label}_sha256": frozen[label]["sha256"] for label in
           ("prompt_file", "token_inventory", "batch_plan", "selection_plan",
            "teacher_tensor", "orthogonal_directions")},
        "subliminal_adapter_tree_sha256": frozen["adapters"]["subliminal"]["tree_sha256"],
        "neutral_adapter_tree_sha256": frozen["adapters"]["neutral"]["tree_sha256"],
        **{f"fsd_{label}_sha256": frozen["fsd"][label]["sha256"] for label in
           ("manifest", "subliminal_states", "neutral_states", "aggregate")},
        "requirements_lock_sha256": manifest["execution"]["requirements_lock"]["sha256"],
    }
    record = {
        "schema_version": 2, "experiment_id": EXPERIMENT_ID,
        "decision": AUTHORIZATION_DECISION, "authorized": True,
        "authorized_at": "2026-09-21T12:00:00+02:00",
        "authorization_basis_commit": "2" * 40, "execution_commit": EXECUTION_COMMIT,
        "execution_commit_policy": {
            "frozen_execution_commit": FROZEN_EXECUTION_COMMIT,
            "blocked_execution_base_commit": BLOCKED_EXECUTION_BASE_COMMIT,
            "relationship": SUCCESSOR_RELATIONSHIP,
            "permitted_changed_paths": sorted(EXECUTION_CONTROL_PATHS),
        },
        "bindings": bindings,
        "container": {"image": "example/image", "digest": "sha256:" + "e" * 64},
        "scientific_inputs": inputs,
        "scope": {"seed": 2, "scientific_contract_changes_authorized": False,
                  "outcome_release_authorized": False,
                  "seed_1_or_3_execution_authorized": False,
                  "merge_tag_or_release_authorized": False},
        "supersedes": {
            "authorization_record_sha256": "f8f008c061101d828f3c9a7662e07a9489ada4380826a676b7b13963844ce584",
            "failed_execution_report_status": "C18 V2 EXECUTION BLOCKED",
        },
    }
    return manifest, record, manifest_path, technical


def validate(record, manifest, manifest_path, technical, **kwargs):
    return validate_scientific_authorization(
        record, manifest, manifest_path, root=manifest_path.parent,
        execution_commit=kwargs.pop("execution_commit", EXECUTION_COMMIT),
        technical_directory=technical, **{"verify_git": False, **kwargs},
    )


def move_adapters_to_shared(manifest: dict, repository: Path, shared: Path) -> None:
    for condition, item in manifest["frozen_inputs"]["adapters"].items():
        source = repository / item["path"]
        logical = f"results/adapters/{condition}"
        target = shared / logical
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        item["path"] = logical


def test_exact_frozen_contract_and_matching_external_authorization_pass(tmp_path):
    manifest, record, manifest_path, technical = fixture(tmp_path)
    assert validate(record, manifest, manifest_path, technical)["authorization"] == "PASS"


@pytest.mark.parametrize(("mutation", "message"), [
    (lambda r: r.update(authorized=False), "not authorized"),
    (lambda r: r["bindings"].update(manifest_sha256="0" * 64), "frozen scientific contract"),
    (lambda r: r.update(experiment_id="wrong-study"), "Study-ID"),
    (lambda r: r.pop("scope"), "incomplete"),
])
def test_invalid_authorization_records_stop(tmp_path, mutation, message):
    manifest, record, manifest_path, technical = fixture(tmp_path)
    mutation(record)
    with pytest.raises(RuntimeError, match=message):
        validate(record, manifest, manifest_path, technical)


def test_wrong_execution_commit_stops(tmp_path):
    manifest, record, manifest_path, technical = fixture(tmp_path)
    with pytest.raises(RuntimeError, match="execution commit differs"):
        validate(record, manifest, manifest_path, technical, execution_commit="3" * 40)


def test_changed_preregistration_stops(tmp_path):
    manifest, record, manifest_path, technical = fixture(tmp_path)
    (tmp_path / "preregistration.md").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="public scientific contract changed"):
        validate(record, manifest, manifest_path, technical)


def test_changed_scientific_input_stops(tmp_path):
    manifest, record, manifest_path, technical = fixture(tmp_path)
    (tmp_path / "prompt_file.bin").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="scientific input differs"):
        validate(record, manifest, manifest_path, technical)


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True)


def test_direct_runner_without_authorization_stops_before_outcomes(tmp_path):
    result = run_script(
        "scripts/run_bidirectional_teacher_coordinate_interchange_v2.py",
        "--condition", "neutral", "--authorization", str(tmp_path / "absent.json"),
        "--technical-audit", str(tmp_path / "audit.json"), "--output", str(tmp_path / "raw.sealed"),
        "--execution-git-commit", EXECUTION_COMMIT,
    )
    assert result.returncode != 0 and "authorization record is absent" in result.stderr
    assert not (tmp_path / "raw.sealed").exists()


def test_scientific_dag_generation_without_shared_root_stops(tmp_path):
    env = {key: value for key, value in os.environ.items() if key != "SLGEO_SHARED_ROOT"}
    result = subprocess.run(
        [sys.executable, "scripts/generate_bidirectional_teacher_coordinate_interchange_v2_dag.py",
         "--mode", "scientific", "--execution-git-commit", EXECUTION_COMMIT,
         "--output", str(tmp_path / "scientific.dag")],
        cwd=ROOT, capture_output=True, text=True, env=env,
    )
    assert result.returncode != 0 and "SLGEO_SHARED_ROOT is required" in result.stderr
    assert not (tmp_path / "scientific.dag").exists()


def test_aggregation_without_upstream_authorization_stops(tmp_path):
    result = run_script(
        "scripts/aggregate_bidirectional_teacher_coordinate_interchange_v2.py",
        "--authorization", str(tmp_path / "absent.json"),
        "--technical-directory", str(tmp_path), "--execution-git-commit", EXECUTION_COMMIT,
        "--release-token", str(tmp_path / "release.json"),
        "--subliminal", "absent", "--neutral", "absent", "--output", str(tmp_path / "aggregate.json"),
    )
    assert result.returncode != 0 and "authorization record is absent" in result.stderr
    assert not (tmp_path / "aggregate.json").exists()


def test_scientific_launchers_read_the_authorization_outside_the_checkout():
    for relative in (
        "condor/run_bidirectional_teacher_coordinate_interchange_v2_task.sh",
        "scripts/run_bidirectional_teacher_coordinate_interchange_v2_manifest.py",
        "scripts/generate_bidirectional_teacher_coordinate_interchange_v2_dag.py",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "SCIENTIFIC_EXECUTION_AUTHORIZATION_V" not in text
        assert "research/bidirectional_teacher_coordinate_interchange_v2/SCIENTIFIC" not in text
    wrapper = (ROOT / "condor/run_bidirectional_teacher_coordinate_interchange_v2_task.sh").read_text(encoding="utf-8")
    assert '--authorization "$ROOT/authorization/SCIENTIFIC_EXECUTION_AUTHORIZATION.json"' in wrapper
    assert '--technical-audit "$ROOT/technical/audit.json"' in wrapper
    assert str(AUTHORIZATION_RELATIVE_PATH.as_posix()) == "authorization/SCIENTIFIC_EXECUTION_AUTHORIZATION.json"


def test_canonical_scientific_paths_are_below_the_shared_output_root(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    manifest = {"output": {"root": "results/research/c18", "technical_namespace": "technical"}}
    monkeypatch.delenv("SLGEO_SHARED_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="SLGEO_SHARED_ROOT is required"):
        canonical_scientific_paths(manifest, tmp_path)
    monkeypatch.setenv("SLGEO_SHARED_ROOT", str(shared))
    record, technical = canonical_scientific_paths(apply_storage_overrides(manifest), tmp_path)
    output = (shared / "results/research/c18").resolve()
    assert record == output / "authorization/SCIENTIFIC_EXECUTION_AUTHORIZATION.json"
    assert technical == output / "technical"
    with pytest.raises(RuntimeError, match="output root is not canonical"):
        canonical_scientific_paths({"output": {"root": str(tmp_path / "elsewhere"),
                                               "technical_namespace": "technical"}}, tmp_path)


def test_gpu_submit_description_pins_the_frozen_execution_identity():
    text = (ROOT / "condor/bidirectional_teacher_coordinate_interchange_v2_task_gpu.sub").read_text(encoding="utf-8")
    lines = dict(line.split(" = ", 1) for line in text.splitlines() if " = " in line)
    assert lines["request_GPUs"] == "1"
    assert lines["require_gpus"] == 'DeviceName == "NVIDIA A100-PCIE-40GB" && NvidiaDriver == "570.211.01"'
    assert lines["requirements"] == (
        'UidDomain == "cs.uni-saarland.de" && GPUs_DeviceName == "NVIDIA A100-PCIE-40GB" '
        '&& GPUs_NvidiaDriver == "570.211.01"'
    )
    assert "condor/bidirectional_teacher_coordinate_interchange_v2_task_gpu.sub" in EXECUTION_CONTROL_PATHS


@pytest.mark.parametrize(("key", "observed"), [
    ("gpu_class", "NVIDIA A100-SXM4-80GB"),
    ("nvidia_driver", "550.163.01"),
    ("container_image", "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"),
    ("torch", "2.5.1+cu121"),
])
def test_wrong_runtime_execution_identity_stops(monkeypatch, key, observed):
    from slgeo.analysis import c18_execution
    from slgeo.io import load_yaml

    manifest = load_yaml(ROOT / "configs/validation/cat_bidirectional_teacher_coordinate_interchange_v2.yaml")
    execution = manifest["execution"]
    identity = {name: execution[name] for name in (
        "python", "torch", "cuda_runtime", "transformers", "peft", "bitsandbytes", "numpy",
        "gpu_class", "attention_backend", "container_image", "nvidia_driver")}
    monkeypatch.setattr(c18_execution, "runtime_identity", lambda _backend: dict(identity))
    assert c18_execution.assert_runtime_identity(manifest) == identity
    monkeypatch.setattr(c18_execution, "runtime_identity", lambda _backend: {**identity, key: observed})
    with pytest.raises(RuntimeError, match=f"execution identity mismatch for {key}"):
        c18_execution.assert_runtime_identity(manifest)


def test_sealing_preconditions_fail_before_computation(tmp_path):
    from cryptography.fernet import Fernet

    output = tmp_path / "subliminal.npz.sealed"
    with pytest.raises(RuntimeError, match="sealing key is absent"):
        validate_sealing_preconditions(b"", output)
    with pytest.raises(RuntimeError, match="sealing key is malformed"):
        validate_sealing_preconditions(b"not-a-fernet-key", output)
    key = Fernet.generate_key()
    validate_sealing_preconditions(key, output)
    for existing in (output, tmp_path / "subliminal.npz.sealed.provenance.json"):
        write(existing, b"partial")
        with pytest.raises(RuntimeError, match="sealed scientific output already exists"):
            validate_sealing_preconditions(key, output)
        existing.unlink()


def test_real_execution_lineage_passes_git_free_successor_check(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _forbidden_subprocess)
    monkeypatch.setenv("PATH", "")
    # 4ea9af1 is the audited pre-repair descendant; the check needs only Git objects.
    _verify_execution_successor(ROOT, "4ea9af1d58ee73eeaf0da5a64a454e0a029aed60")
    with pytest.raises(RuntimeError, match="not a descendant"):
        _verify_execution_successor(ROOT, FROZEN_EXECUTION_COMMIT)


def _forbidden_subprocess(*_args, **_kwargs):
    raise AssertionError("the successor check must not start a subprocess")


def _tree(repository, files: dict[str, bytes], modes: dict[str, int]) -> bytes:
    from dulwich.objects import Blob, Tree

    tree = Tree()
    children: dict[str, dict[str, bytes]] = {}
    for name, content in files.items():
        head, _, rest = name.partition("/")
        if rest:
            children.setdefault(head, {})[rest] = content
            continue
        blob = Blob.from_string(content)
        repository.object_store.add_object(blob)
        tree.add(name.encode(), modes.get(name, 0o100644), blob.id)
    for name, nested in children.items():
        tree.add(name.encode(), 0o040000, _tree(repository, nested, {}))
    repository.object_store.add_object(tree)
    return tree.id


def _commit(repository, files: dict[str, bytes], parents: list[bytes], modes: dict[str, int] | None = None) -> bytes:
    from dulwich.objects import Commit

    tree_id = _tree(repository, files, modes or {})
    commit = Commit()
    commit.tree, commit.parents = tree_id, parents
    commit.author = commit.committer = b"C18 test <c18@example.invalid>"
    commit.commit_time = commit.author_time = 1_700_000_000 + len(parents)
    commit.commit_timezone = commit.author_timezone = 0
    commit.message = b"synthetic"
    repository.object_store.add_object(commit)
    return commit.id


@pytest.fixture
def synthetic_lineage(tmp_path, monkeypatch):
    """Synthetic frozen -> blocked -> successor chain whose control set is one flat path."""
    from dulwich.repo import Repo
    from slgeo.analysis import c18_v2_authorization as authorization

    repository = Repo.init(str(tmp_path / "repo"), mkdir=True)
    frozen = _commit(repository, {"contract.md": b"frozen", "control.sh": b"v0"}, [])
    blocked = _commit(repository, {"contract.md": b"frozen", "control.sh": b"v1"}, [frozen])
    monkeypatch.setattr(authorization, "FROZEN_EXECUTION_COMMIT", frozen.decode())
    monkeypatch.setattr(authorization, "BLOCKED_EXECUTION_BASE_COMMIT", blocked.decode())
    monkeypatch.setattr(authorization, "EXECUTION_CONTROL_PATHS", frozenset({"control.sh", "new_control.sh"}))
    monkeypatch.setattr(subprocess, "run", _forbidden_subprocess)
    monkeypatch.setenv("PATH", "")
    return repository, frozen, blocked


def test_git_free_successor_check_accepts_control_only_descendant(synthetic_lineage):
    repository, _frozen, blocked = synthetic_lineage
    successor = _commit(repository, {"contract.md": b"frozen", "control.sh": b"v2", "new_control.sh": b"x"}, [blocked])
    _verify_execution_successor(Path(repository.path), successor.decode())


@pytest.mark.parametrize(("files", "message"), [
    ({"contract.md": b"changed", "control.sh": b"v2"}, "outside the authorized control layer"),
    ({"control.sh": b"v2"}, "outside the authorized control layer"),
    ({"contract.md": b"frozen", "moved_control.sh": b"v1"}, "outside the authorized control layer"),
    # Rename of a scientific file onto a permitted path: only the old path reveals it.
    ({"new_control.sh": b"frozen", "control.sh": b"v2"}, "outside the authorized control layer"),
    ({"contract.md": b"frozen", "control.sh": b"v1"}, "does not differ from the blocked base"),
])
def test_git_free_successor_check_rejects_scientific_or_empty_changes(synthetic_lineage, files, message):
    repository, _frozen, blocked = synthetic_lineage
    successor = _commit(repository, files, [blocked])
    with pytest.raises(RuntimeError, match=message):
        _verify_execution_successor(Path(repository.path), successor.decode())


def test_git_free_successor_check_rejects_mode_and_nested_changes(synthetic_lineage):
    repository, _frozen, blocked = synthetic_lineage
    mode_only = _commit(repository, {"contract.md": b"frozen", "control.sh": b"v2"}, [blocked],
                        modes={"contract.md": 0o100755})
    with pytest.raises(RuntimeError, match="outside the authorized control layer"):
        _verify_execution_successor(Path(repository.path), mode_only.decode())
    nested = _commit(repository, {"contract.md": b"frozen", "control.sh": b"v2", "research/x.md": b"x"}, [blocked])
    with pytest.raises(RuntimeError, match="outside the authorized control layer"):
        _verify_execution_successor(Path(repository.path), nested.decode())


def test_git_free_successor_check_never_resolves_a_ref_named_like_a_commit(synthetic_lineage):
    repository, _frozen, blocked = synthetic_lineage
    valid = _commit(repository, {"contract.md": b"frozen", "control.sh": b"v2"}, [blocked])
    fake = "e" * 40
    # A ref substitution would redirect the unknown SHA to a passing successor.
    repository.refs[f"refs/heads/{fake}".encode()] = valid
    repository.refs[f"refs/tags/{fake}".encode()] = valid
    with pytest.raises(RuntimeError, match="Git verification failed"):
        _verify_execution_successor(Path(repository.path), fake)


def test_git_free_successor_check_rejects_non_descendant_and_unknown_commits(synthetic_lineage, tmp_path):
    repository, frozen, _blocked = synthetic_lineage
    sibling = _commit(repository, {"contract.md": b"frozen", "control.sh": b"v2"}, [frozen])
    with pytest.raises(RuntimeError, match="not a descendant"):
        _verify_execution_successor(Path(repository.path), sibling.decode())
    with pytest.raises(RuntimeError, match="Git verification failed"):
        _verify_execution_successor(Path(repository.path), "f" * 40)
    with pytest.raises(RuntimeError, match="Git verification failed"):
        _verify_execution_successor(tmp_path / "not-a-repository", "f" * 40)


def test_authorization_gate_uses_the_git_free_successor_check(tmp_path, monkeypatch):
    from slgeo.analysis import c18_v2_authorization as authorization

    manifest, record, manifest_path, technical = fixture(tmp_path)
    calls = []
    monkeypatch.setattr(authorization, "_verify_execution_successor", lambda root, commit: calls.append((root, commit)))
    monkeypatch.setattr(subprocess, "run", _forbidden_subprocess)
    validate(record, manifest, manifest_path, technical, verify_git=True)
    assert calls == [(manifest_path.parent.resolve(), EXECUTION_COMMIT)]


def test_wrong_authorization_decision_and_stale_policy_stop(tmp_path):
    manifest, record, manifest_path, technical = fixture(tmp_path)
    stale = json.loads(json.dumps(record))
    stale["decision"] = "C18 V2 SCIENTIFIC EXECUTION RE-AUTHORIZED"
    with pytest.raises(RuntimeError, match="researcher decision differs"):
        validate(stale, manifest, manifest_path, technical)
    stale = json.loads(json.dumps(record))
    stale["execution_commit_policy"]["permitted_changed_paths"].remove(
        "condor/bidirectional_teacher_coordinate_interchange_v2_task_gpu.sub")
    with pytest.raises(RuntimeError, match="successor policy differs"):
        validate(stale, manifest, manifest_path, technical)


@pytest.mark.parametrize("historical", ["SCIENTIFIC_EXECUTION_AUTHORIZATION_V2.json", "SCIENTIFIC_EXECUTION_AUTHORIZATION_V3.json"])
def test_historical_tracked_records_cannot_authorize_this_lineage(historical):
    from slgeo.io import load_yaml

    manifest_path = ROOT / "configs/validation/cat_bidirectional_teacher_coordinate_interchange_v2.yaml"
    record = json.loads((ROOT / "research/bidirectional_teacher_coordinate_interchange_v2" / historical).read_text(encoding="utf-8"))
    with pytest.raises(RuntimeError, match="STOP"):
        validate_scientific_authorization(
            record, load_yaml(manifest_path), manifest_path, root=ROOT,
            execution_commit=record["execution_commit"], technical_directory=ROOT / "absent",
            verify_runtime_files=False, verify_git=False,
        )


def test_wrong_technical_bundle_commit_stops(tmp_path):
    manifest, record, manifest_path, technical = fixture(tmp_path)
    write(technical / "preflight.json", json.dumps({"execution_commit": "4" * 40}).encode())
    record["bindings"]["technical_preflight_sha256"] = sha256_file(technical / "preflight.json")
    with pytest.raises(RuntimeError, match="technical artifact execution commit differs: preflight.json"):
        validate(record, manifest, manifest_path, technical)


def test_changed_technical_bundle_bytes_stop(tmp_path):
    manifest, record, manifest_path, technical = fixture(tmp_path)
    write(technical / "audit.json", json.dumps({"status": "PASS"}).encode())
    with pytest.raises(RuntimeError, match="technical validation binding differs: audit.json"):
        validate(record, manifest, manifest_path, technical)


def generator(tmp_path, *extra: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/generate_bidirectional_teacher_coordinate_interchange_v2_dag.py",
         "--mode", "scientific", *extra],
        cwd=ROOT, capture_output=True, text=True, env={**os.environ, **(env or {})},
    )


def test_scientific_dag_with_foreign_shared_root_stops(tmp_path):
    output = ROOT / "condor/runtime/test_c18v2_scientific.dag"
    result = generator(tmp_path, "--execution-git-commit", EXECUTION_COMMIT, "--output", str(output),
                       env={"SLGEO_SHARED_ROOT": str(tmp_path / "shared"), "USER": "c18test"})
    assert result.returncode != 0 and "differs from the node shared root" in result.stderr
    assert not output.exists()


def test_scientific_dag_outside_ignored_runtime_directory_stops(tmp_path):
    user = "c18test"
    shared = f"/scratch/compuling/{user}/beyond-steering-vectors"
    result = generator(tmp_path, "--execution-git-commit", EXECUTION_COMMIT,
                       "--output", str(tmp_path / "scientific.dag"),
                       env={"SLGEO_SHARED_ROOT": shared, "USER": user})
    assert result.returncode != 0 and "below ignored condor/runtime/" in result.stderr
    assert not (tmp_path / "scientific.dag").exists()


def test_scientific_dag_for_a_different_execution_commit_stops(tmp_path):
    user = "c18test"
    output = ROOT / "condor/runtime/test_c18v2_scientific.dag"
    result = generator(tmp_path, "--execution-git-commit", EXECUTION_COMMIT, "--output", str(output),
                       env={"SLGEO_SHARED_ROOT": f"/scratch/compuling/{user}/beyond-steering-vectors", "USER": user})
    assert result.returncode != 0 and "submit checkout differs" in result.stderr
    assert not output.exists()


def test_both_adapters_pass_when_present_only_under_shared_root(tmp_path, monkeypatch):
    manifest, record, manifest_path, technical = fixture(tmp_path)
    shared = tmp_path / "shared"
    move_adapters_to_shared(manifest, tmp_path, shared)
    monkeypatch.setenv("SLGEO_SHARED_ROOT", str(shared))
    assert validate(record, manifest, manifest_path, technical)["authorization"] == "PASS"


@pytest.mark.parametrize("condition", ["subliminal", "neutral"])
def test_missing_shared_adapter_stops(tmp_path, monkeypatch, condition):
    manifest, record, manifest_path, technical = fixture(tmp_path)
    shared = tmp_path / "shared"
    move_adapters_to_shared(manifest, tmp_path, shared)
    shutil.rmtree(shared / manifest["frozen_inputs"]["adapters"][condition]["path"])
    monkeypatch.setenv("SLGEO_SHARED_ROOT", str(shared))
    with pytest.raises(RuntimeError, match=f"scientific adapter differs: {condition}"):
        validate(record, manifest, manifest_path, technical)


def test_repository_adapter_is_not_a_fallback_for_shared_artifact(tmp_path, monkeypatch):
    manifest, record, manifest_path, technical = fixture(tmp_path)
    shared = tmp_path / "shared"
    move_adapters_to_shared(manifest, tmp_path, shared)
    logical = manifest["frozen_inputs"]["adapters"]["subliminal"]["path"]
    shutil.rmtree(shared / logical)
    write(tmp_path / logical / "adapter_config.json", b"repository decoy")
    monkeypatch.setenv("SLGEO_SHARED_ROOT", str(shared))
    with pytest.raises(RuntimeError, match="scientific adapter differs: subliminal"):
        validate(record, manifest, manifest_path, technical)


@pytest.mark.parametrize("condition", ["subliminal", "neutral"])
def test_wrong_adapter_hash_at_canonical_shared_path_stops(tmp_path, monkeypatch, condition):
    manifest, record, manifest_path, technical = fixture(tmp_path)
    shared = tmp_path / "shared"
    move_adapters_to_shared(manifest, tmp_path, shared)
    logical = manifest["frozen_inputs"]["adapters"][condition]["path"]
    write(shared / logical / "adapter_config.json", b"wrong adapter")
    monkeypatch.setenv("SLGEO_SHARED_ROOT", str(shared))
    with pytest.raises(RuntimeError, match=f"scientific adapter differs: {condition}"):
        validate(record, manifest, manifest_path, technical)


def test_wrong_shared_root_stops_without_repository_fallback(tmp_path, monkeypatch):
    manifest, record, manifest_path, technical = fixture(tmp_path)
    shared = tmp_path / "shared"
    move_adapters_to_shared(manifest, tmp_path, shared)
    monkeypatch.setenv("SLGEO_SHARED_ROOT", str(tmp_path / "wrong-shared"))
    with pytest.raises(RuntimeError, match="scientific adapter differs: subliminal"):
        validate(record, manifest, manifest_path, technical)


@pytest.mark.parametrize("logical", ["results/../secret", "runs/x/../../secret", "data/./x"])
def test_path_traversal_is_rejected(tmp_path, logical):
    with pytest.raises(ValueError, match="traversal"):
        resolve_scientific_artifact(tmp_path, logical, shared_root=tmp_path / "shared")


def test_repository_and_shared_storage_classes_are_deterministic(tmp_path):
    shared = tmp_path / "shared"
    assert resolve_scientific_artifact(tmp_path, "research/contract.json", shared_root=shared) == (
        tmp_path / "research/contract.json"
    ).resolve()
    for prefix in ("data", "results", "runs"):
        assert resolve_scientific_artifact(tmp_path, f"{prefix}/artifact.bin", shared_root=shared) == (
            shared / prefix / "artifact.bin"
        ).resolve()


def test_runner_override_and_authorization_resolver_are_identical(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    monkeypatch.setenv("SLGEO_SHARED_ROOT", str(shared))
    logical = "results/adapters/subliminal"
    overridden = apply_storage_overrides({"path": logical})["path"]
    assert Path(overridden) == resolve_scientific_artifact(tmp_path, logical)


def _load_script(name: str, monkeypatch):
    import importlib.util

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(f"c18_test_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scientific_dag_without_canonical_authorization_record_stops(tmp_path, monkeypatch):
    generator_module = _load_script("generate_bidirectional_teacher_coordinate_interchange_v2_dag", monkeypatch)
    user = "c18test"
    monkeypatch.setenv("USER", user)
    monkeypatch.setenv("SLGEO_SHARED_ROOT", f"/scratch/compuling/{user}/beyond-steering-vectors")
    checked = []
    monkeypatch.setattr(generator_module, "validate_checkout", lambda root, commit: checked.append(commit))
    output = ROOT / "condor/runtime/test_c18v2_scientific_absent_record.dag"
    monkeypatch.setattr(sys, "argv", ["generate", "--mode", "scientific", "--execution-git-commit",
                                      EXECUTION_COMMIT, "--output", str(output)])
    with pytest.raises(RuntimeError, match="scientific authorization record is absent"):
        generator_module.main()
    assert checked == [EXECUTION_COMMIT]
    assert not output.exists()


def test_scientific_dag_uses_the_node_shared_root_constant():
    generator_module_text = (ROOT / "scripts/generate_bidirectional_teacher_coordinate_interchange_v2_dag.py").read_text(encoding="utf-8")
    assert generator_module_text.count("/scratch/compuling/") == 1


def test_runner_sealing_preconditions_stop_before_inputs_and_model(tmp_path, monkeypatch):
    runner = _load_script("run_bidirectional_teacher_coordinate_interchange_v2", monkeypatch)
    from cryptography.fernet import Fernet

    def forbidden(*_args, **_kwargs):
        raise AssertionError("scientific inputs or model touched before sealing preconditions")

    monkeypatch.setattr(runner, "load_and_validate_scientific_authorization", lambda *a, **k: {"authorization": "PASS"})
    for name in ("validate_public_inputs", "assert_runtime_identity", "load_model"):
        monkeypatch.setattr(runner, name, forbidden)
    monkeypatch.setenv("SLGEO_C18_V2_SEAL_KEY", Fernet.generate_key().decode("ascii"))
    output = tmp_path / "neutral.npz.sealed"
    write(output, b"partial")
    monkeypatch.setattr(sys, "argv", ["run", "--condition", "neutral", "--authorization", str(tmp_path / "a.json"),
                                      "--technical-audit", str(tmp_path / "audit.json"), "--output", str(output),
                                      "--execution-git-commit", EXECUTION_COMMIT])
    with pytest.raises(RuntimeError, match="sealed scientific output already exists"):
        runner.main()
    monkeypatch.setenv("SLGEO_C18_V2_SEAL_KEY", "malformed")
    with pytest.raises(RuntimeError, match="sealing key is malformed"):
        runner.main()
    assert output.read_bytes() == b"partial"

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from slgeo.analysis.c18_v2_authorization import (
    AUTHORIZATION_DECISION, BLOCKED_EXECUTION_BASE_COMMIT, EXECUTION_CONTROL_PATHS, FROZEN_EXECUTION_COMMIT,
    SUCCESSOR_RELATIONSHIP, TECHNICAL_BINDINGS, validate_scientific_authorization,
)
from slgeo.analysis.c18_v2_execution import EXPERIMENT_ID
from slgeo.analysis.c18_v2_manifest import sha256_file, tree_digest


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
        technical_directory=technical, verify_git=False, **kwargs,
    )


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


def test_scientific_dag_generation_without_authorization_stops(tmp_path):
    result = run_script(
        "scripts/generate_bidirectional_teacher_coordinate_interchange_v2_dag.py",
        "--mode", "scientific", "--execution-git-commit", EXECUTION_COMMIT,
        "--output", str(tmp_path / "scientific.dag"),
    )
    assert result.returncode != 0 and "authorization record is absent" in result.stderr
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

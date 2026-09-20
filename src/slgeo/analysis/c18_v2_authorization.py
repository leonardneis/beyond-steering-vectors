"""Fail-closed external authorization for the frozen C18-v2 contract."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from .c18_v2_execution import EXPERIMENT_ID
from .c18_v2_manifest import EXECUTION_CONTROL_SUCCESSOR_PATHS, sha256_file, tree_digest


FROZEN_EXECUTION_COMMIT = "0e1bd7eca2d6003327dd95559b5107e5930b465b"
BLOCKED_EXECUTION_BASE_COMMIT = "4a930b17ae92b0df28b216d525c89ca39e0a86f4"
AUTHORIZATION_DECISION = "C18 V2 SCIENTIFIC EXECUTION AUTHORIZED"
SUCCESSOR_RELATIONSHIP = "descendant_with_execution_control_only_changes"
EXECUTION_CONTROL_PATHS = EXECUTION_CONTROL_SUCCESSOR_PATHS
TECHNICAL_BINDINGS = {
    "technical_preflight_sha256": "preflight.json",
    "technical_gpu_validation_sha256": "validation.json",
    "technical_validation_provenance_sha256": "validation.json.provenance.json",
    "technical_sha256sums_sha256": "SHA256SUMS",
    "technical_independent_audit_sha256": "audit.json",
}
HISTORICAL_AUTHORIZATION_SHA256 = "f8f008c061101d828f3c9a7662e07a9489ada4380826a676b7b13963844ce584"
HISTORICAL_AUTHORIZATION_PATH = Path(
    "research/bidirectional_teacher_coordinate_interchange_v2/SCIENTIFIC_EXECUTION_AUTHORIZATION.json"
)


def _stop(message: str) -> None:
    raise RuntimeError(f"STOP: {message}")


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        _stop(f"{label} is incomplete or contains unexpected fields")


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=False, capture_output=True, text=True,
    )
    if result.returncode:
        _stop(f"Git verification failed: {' '.join(args)}")
    return result.stdout.strip()


def _expected_scientific_inputs(manifest: Mapping[str, Any]) -> dict[str, str]:
    frozen = manifest["frozen_inputs"]
    return {
        "model_config_sha256": frozen["model_config"]["sha256"],
        "model_revision": frozen["model"]["revision"],
        "prompt_file_sha256": frozen["prompt_file"]["sha256"],
        "token_inventory_sha256": frozen["token_inventory"]["sha256"],
        "batch_plan_sha256": frozen["batch_plan"]["sha256"],
        "selection_plan_sha256": frozen["selection_plan"]["sha256"],
        "teacher_tensor_sha256": frozen["teacher_tensor"]["sha256"],
        "orthogonal_directions_sha256": frozen["orthogonal_directions"]["sha256"],
        "subliminal_adapter_tree_sha256": frozen["adapters"]["subliminal"]["tree_sha256"],
        "neutral_adapter_tree_sha256": frozen["adapters"]["neutral"]["tree_sha256"],
        "fsd_manifest_sha256": frozen["fsd"]["manifest"]["sha256"],
        "fsd_subliminal_states_sha256": frozen["fsd"]["subliminal_states"]["sha256"],
        "fsd_neutral_states_sha256": frozen["fsd"]["neutral_states"]["sha256"],
        "fsd_aggregate_sha256": frozen["fsd"]["aggregate"]["sha256"],
        "requirements_lock_sha256": manifest["execution"]["requirements_lock"]["sha256"],
    }


def _verify_scientific_files(manifest: Mapping[str, Any], root: Path) -> None:
    frozen = manifest["frozen_inputs"]
    for label in ("model_config", "prompt_file", "token_inventory", "batch_plan",
                  "selection_plan", "teacher_tensor", "orthogonal_directions"):
        item = frozen[label]
        path = root / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            _stop(f"scientific input differs: {label}")
    for condition, item in frozen["adapters"].items():
        if tree_digest(root / item["path"]) != item["tree_sha256"]:
            _stop(f"scientific adapter differs: {condition}")
    for label, item in frozen["fsd"].items():
        path = root / item["path"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            _stop(f"scientific FSD input differs: {label}")
    lock = manifest["execution"]["requirements_lock"]
    if sha256_file(root / lock["path"]) != lock["sha256"]:
        _stop("requirements lock differs")


def validate_scientific_authorization(
    record: Mapping[str, Any], manifest: Mapping[str, Any], manifest_path: str | Path,
    *, root: str | Path, execution_commit: str,
    technical_directory: str | Path | None = None,
    verify_runtime_files: bool = True, verify_git: bool = True,
) -> dict[str, str]:
    """Validate a post-repair authorization without mutating the frozen manifest."""
    base = Path(root).resolve()
    manifest_file = Path(manifest_path)
    if not manifest_file.is_absolute():
        manifest_file = base / manifest_file
    _require_exact_keys(record, {
        "schema_version", "experiment_id", "decision", "authorized", "authorized_at",
        "authorization_basis_commit", "execution_commit", "execution_commit_policy",
        "bindings", "container", "scientific_inputs", "scope", "supersedes",
    }, "authorization record")
    if record["schema_version"] != 2 or record["authorized"] is not True:
        _stop("scientific execution is not authorized")
    if record["experiment_id"] != EXPERIMENT_ID or manifest.get("experiment_id") != EXPERIMENT_ID:
        _stop("authorization Study-ID differs")
    if record["decision"] != AUTHORIZATION_DECISION:
        _stop("researcher decision differs")
    if not re.fullmatch(r"[0-9a-f]{40}", str(record["authorization_basis_commit"])):
        _stop("authorization basis commit differs")
    try:
        authorized_at = datetime.fromisoformat(str(record["authorized_at"]))
    except ValueError as exc:
        raise RuntimeError("STOP: authorization timestamp differs") from exc
    if authorized_at.tzinfo is None:
        _stop("authorization timestamp lacks a timezone")
    if not re.fullmatch(r"[0-9a-f]{40}", execution_commit) or record["execution_commit"] != execution_commit:
        _stop("authorization execution commit differs")
    policy = record["execution_commit_policy"]
    if policy != {
        "frozen_execution_commit": FROZEN_EXECUTION_COMMIT,
        "blocked_execution_base_commit": BLOCKED_EXECUTION_BASE_COMMIT,
        "relationship": SUCCESSOR_RELATIONSHIP,
        "permitted_changed_paths": sorted(EXECUTION_CONTROL_PATHS),
    }:
        _stop("execution-commit successor policy differs")
    bindings = record["bindings"]
    expected_binding_keys = {
        "preregistration_sha256", "decision_matrix_sha256", "manifest_sha256",
        "calibration_summary_sha256", "calibration_contract_sha256",
        "calibration_population_sha256", "calibration_seal_sha256",
        "calibration_audit_sha256", *TECHNICAL_BINDINGS,
    }
    _require_exact_keys(bindings, expected_binding_keys, "authorization bindings")
    public = manifest["public_contract"]
    calibration = manifest["calibration"]
    expected_bindings = {
        "preregistration_sha256": public["preregistration"]["sha256"],
        "decision_matrix_sha256": public["decision_matrix"]["sha256"],
        "manifest_sha256": sha256_file(manifest_file),
        "calibration_summary_sha256": public["calibration_summary"]["sha256"],
        "calibration_contract_sha256": calibration["contract_sha256"],
        "calibration_population_sha256": calibration["population_sha256"],
        "calibration_seal_sha256": calibration["seal_sha256"],
        "calibration_audit_sha256": calibration["audit_sha256"],
    }
    if any(bindings.get(key) != value for key, value in expected_bindings.items()):
        _stop("authorization is not bound to the frozen scientific contract")
    inputs = _expected_scientific_inputs(manifest)
    if record["scientific_inputs"] != inputs:
        _stop("authorization scientific-input bindings differ")
    expected_container = manifest["execution"]["container_image"].rsplit("@", 1)
    if record["container"] != {"image": expected_container[0], "digest": expected_container[1]}:
        _stop("authorization container identity differs")
    if record["scope"] != {
        "seed": 2, "scientific_contract_changes_authorized": False,
        "outcome_release_authorized": False,
        "seed_1_or_3_execution_authorized": False,
        "merge_tag_or_release_authorized": False,
    }:
        _stop("authorization scope differs")
    supersedes = record["supersedes"]
    if supersedes != {
            "authorization_record_sha256": HISTORICAL_AUTHORIZATION_SHA256,
        "failed_execution_report_status": "C18 V2 EXECUTION BLOCKED",
    }:
        _stop("historical failed authorization is not superseded append-only")
    if technical_directory is None:
        _stop("technical validation directory is required")
    technical = Path(technical_directory)
    if not technical.is_absolute():
        technical = base / technical
    for binding, filename in TECHNICAL_BINDINGS.items():
        path = technical / filename
        if not path.is_file() or sha256_file(path) != bindings[binding]:
            _stop(f"technical validation binding differs: {filename}")
    audit = json.loads((technical / "audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "PASS" or audit.get("manifest_sha256") != bindings["manifest_sha256"]:
        _stop("exact-manifest independent technical audit has not passed")
    for filename in ("preflight.json", "validation.json", "audit.json"):
        item = json.loads((technical / filename).read_text(encoding="utf-8"))
        observed = (item.get("execution_commit") or item.get("execution_git_commit")
                    or item.get("freeze", {}).get("execution_commit"))
        if observed != execution_commit:
            _stop(f"technical artifact execution commit differs: {filename}")
    if verify_runtime_files:
        historical = base / HISTORICAL_AUTHORIZATION_PATH
        if not historical.is_file() or sha256_file(historical) != HISTORICAL_AUTHORIZATION_SHA256:
            _stop("historical authorization audit trail differs")
        for item in public.values():
            path = base / item["path"]
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                _stop("public scientific contract changed")
        _verify_scientific_files(manifest, base)
    if verify_git:
        _git(base, "merge-base", "--is-ancestor", FROZEN_EXECUTION_COMMIT, execution_commit)
        _git(base, "merge-base", "--is-ancestor", BLOCKED_EXECUTION_BASE_COMMIT, execution_commit)
        changed = set(filter(None, _git(
            base, "diff", "--name-only", BLOCKED_EXECUTION_BASE_COMMIT, execution_commit,
        ).splitlines()))
        if not changed or not changed <= EXECUTION_CONTROL_PATHS:
            _stop("execution commit changes files outside the authorized control layer")
    return {"authorization": "PASS", "execution_commit": execution_commit,
            "manifest_sha256": bindings["manifest_sha256"]}


def load_and_validate_scientific_authorization(
    path: str | Path, manifest: Mapping[str, Any], manifest_path: str | Path, **kwargs: Any,
) -> dict[str, str]:
    authorization_path = Path(path)
    if not authorization_path.is_absolute():
        authorization_path = Path(kwargs["root"]) / authorization_path
    if not authorization_path.is_file():
        _stop("scientific authorization record is absent")
    try:
        record = json.loads(authorization_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("STOP: scientific authorization record is unreadable") from exc
    return validate_scientific_authorization(record, manifest, manifest_path, **kwargs)

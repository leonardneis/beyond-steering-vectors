"""Manifest and public-input validation for frozen C18 v2."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Mapping

import numpy as np

from .c18_v2_execution import EXPERIMENT_ID, validate_batch_plan
from .selection_plans import iter_selection_sets


SHARED_ARTIFACT_PREFIXES = frozenset({"data", "results", "runs"})


EXECUTION_CONTROL_SUCCESSOR_PATHS = frozenset({
    "condor/run_bidirectional_teacher_coordinate_interchange_v2_task.sh",
    "scripts/aggregate_bidirectional_teacher_coordinate_interchange_v2.py",
    "scripts/audit_bidirectional_teacher_coordinate_interchange_v2.py",
    "scripts/audit_c18_v2_technical_validation.py",
    "scripts/generate_bidirectional_teacher_coordinate_interchange_v2_dag.py",
    "scripts/run_bidirectional_teacher_coordinate_interchange_v2.py",
    "scripts/run_bidirectional_teacher_coordinate_interchange_v2_manifest.py",
    "scripts/validate_bidirectional_teacher_coordinate_interchange_v2.py",
    "src/slgeo/analysis/c18_v2_authorization.py",
    "src/slgeo/analysis/c18_v2_manifest.py",
    "tests/test_c18_v2_authorization.py",
    "research/bidirectional_teacher_coordinate_interchange_v2/EXECUTION_CONTROL_REPAIR.md",
    "research/bidirectional_teacher_coordinate_interchange_v2/SCIENTIFIC_AUTHORIZATION_V3_REPORT.md",
    "research/bidirectional_teacher_coordinate_interchange_v2/SCIENTIFIC_EXECUTION_AUTHORIZATION_V2.json",
    "research/bidirectional_teacher_coordinate_interchange_v2/SCIENTIFIC_EXECUTION_AUTHORIZATION_V3.json",
    "research/bidirectional_teacher_coordinate_interchange_v2/SCIENTIFIC_REAUTHORIZATION_REPORT.md",
})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(path: str | Path) -> str | None:
    root = Path(path)
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    for child in sorted(item for item in root.rglob("*") if item.is_file()
                        and item.name != "run_provenance.json" and not item.name.endswith(".provenance.json")):
        digest.update(child.relative_to(root).as_posix().encode("utf-8"))
        digest.update(sha256_file(child).encode("ascii"))
    return digest.hexdigest()


def resolve_scientific_artifact(
    repository_root: str | Path, logical_path: str | Path, *, shared_root: str | Path | None = None,
) -> Path:
    """Resolve one manifest path under the canonical repository/shared-root contract."""
    repository = Path(repository_root).resolve()
    configured_shared = shared_root if shared_root is not None else os.getenv("SLGEO_SHARED_ROOT")
    shared = Path(configured_shared).resolve() if configured_shared else None
    raw = str(logical_path)
    if not raw or "\x00" in raw:
        raise ValueError("scientific artifact path is empty or invalid")
    supplied = Path(raw)
    if supplied.is_absolute():
        if shared is None:
            raise ValueError("absolute scientific artifact path requires SLGEO_SHARED_ROOT")
        resolved = supplied.resolve()
        try:
            relative = resolved.relative_to(shared)
        except ValueError as exc:
            raise ValueError("scientific artifact path escapes SLGEO_SHARED_ROOT") from exc
        if not relative.parts or relative.parts[0] not in SHARED_ARTIFACT_PREFIXES:
            raise ValueError("absolute scientific artifact path has an unauthorized storage class")
        return resolved
    normalized_raw = raw.replace("\\", "/")
    raw_parts = normalized_raw.split("/")
    logical = PurePosixPath(normalized_raw)
    if logical.is_absolute() or not logical.parts or any(part in ("", ".", "..") for part in raw_parts):
        raise ValueError("scientific artifact path traversal is forbidden")
    base = shared if logical.parts[0] in SHARED_ARTIFACT_PREFIXES and shared is not None else repository
    resolved = base.joinpath(*logical.parts).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError("scientific artifact path escapes its canonical root") from exc
    return resolved


def apply_storage_overrides(manifest: Mapping[str, Any]) -> dict[str, Any]:
    shared_root = os.getenv("SLGEO_SHARED_ROOT")
    if not shared_root:
        return dict(manifest)
    root = shared_root.rstrip("/\\")

    def rewrite(value: Any) -> Any:
        if isinstance(value, str) and PurePosixPath(value.replace("\\", "/")).parts[:1] \
                and PurePosixPath(value.replace("\\", "/")).parts[0] in SHARED_ARTIFACT_PREFIXES:
            return str(resolve_scientific_artifact(Path.cwd(), value, shared_root=root))
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value
    return rewrite(dict(manifest))


def selection_inventory(path: str | Path) -> list[dict[str, object]]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = list(iter_selection_sets(plan, set_names=("top_k", "norm_matched_control"), k_values=(20,)))
    if Counter(row["set_name"] for row in rows) != Counter({"top_k": 1, "norm_matched_control": 25}):
        raise ValueError("C18-v2 selection counts differ")
    output = []
    for row in rows:
        modules = list(row["modules"])
        if len(modules) != 20 or len(set(modules)) != 20:
            raise ValueError("C18-v2 set is not 20 unique modules")
        set_id = "top" if row["set_name"] == "top_k" else f"norm_{int(row['draw_id']):02d}"
        output.append({**row, "set_id": set_id})
    if len({row["set_id"] for row in output}) != 26:
        raise ValueError("C18-v2 set IDs differ")
    return output


def validate_manifest_contract(manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version") != 2 or manifest.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("C18-v2 manifest schema/experiment ID differs")
    if manifest.get("scientific_execution_authorized") is not False:
        raise ValueError("C18-v2 manifest must remain scientifically unauthorized")
    design = manifest["design"]
    expected = {
        "seed": 2, "k": 20, "mode": "necessity",
        "parameter_backgrounds": {0: "ablated", 1: "full"},
        "donor_backgrounds": {0: "ablated", 1: "full"},
        "cells": ["Y00", "Y01", "Y10", "Y11"],
        "teacher_slots": list(range(1, 28)), "hook_blocks": list(range(27)),
        "epsilon": 0.04405641704135471,
    }
    for key, value in expected.items():
        if design.get(key) != value:
            raise ValueError(f"frozen C18-v2 {key} differs")
    if design["bootstrap"] != {"prng": "PCG64", "seed": 20260804, "draws": 20000,
                                  "stratification": {"families": 3, "sample_per_family": 24}}:
        raise ValueError("C18-v2 bootstrap contract differs")
    execution = manifest["execution"]
    for key, value in {
        "gpu_class": "NVIDIA A100-PCIE-40GB", "nvidia_driver": "570.211.01",
        "attention_backend": "sdpa", "residual_dtype": "float16",
        "position_ids": "logical_attention_mask_cumsum", "cache_position": "rank1_physical_arange",
        "padding_side": "left", "lm_head_evaluation": "canonical_b6_same_shape",
    }.items():
        if execution.get(key) != value:
            raise ValueError(f"C18-v2 execution field differs: {key}")
    if execution["scientific_batch_plan"]["batch_size"] != 6:
        raise ValueError("C18-v2 batch size differs")
    if "@sha256:" not in execution["container_image"]:
        raise ValueError("C18-v2 container is not digest pinned")
    calibration = manifest["calibration"]
    expected_hashes = {
        "contract_sha256": "a6fbce42d08672f8e02e2276e7ff935d10ecdffc6c6a127110c2c24ea0aa97e5",
        "population_sha256": "624b4a0c4e013f0a40ee99e6b3a62a567b635704ddc1f5e6d5257fe3fc3d4e86",
        "seal_sha256": "d5fdae2ea87f40bd1155d74de6d48bf5b49ee7e2414504c26e05bc64c024d8ff",
        "audit_sha256": "b6e8808dbff887c7ff403ab7eb2d1211d092e71b098274f3ce927c82926d345f",
    }
    if any(calibration.get(key) != value for key, value in expected_hashes.items()):
        raise ValueError("C18-v2 calibration identity differs")
    if manifest["release_gates"].get("require_explicit_scientific_authorization") is not True:
        raise ValueError("C18-v2 scientific authorization gate absent")


def validate_public_inputs(
    manifest: Mapping[str, Any], root: str | Path, *, require_runtime_inputs: bool = False,
    read_sensitive: bool = False, allow_execution_control_successor: bool = False,
) -> dict[str, str]:
    base = Path(root)
    checked = {}
    public = manifest["public_contract"]
    for label, item in public.items():
        path = base / item["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"C18-v2 {label} hash differs")
        checked[label] = "PASS"
    frozen = manifest["frozen_inputs"]
    for label in ("model_config", "prompt_file", "token_inventory", "batch_plan", "selection_plan",
                  "teacher_tensor", "orthogonal_directions"):
        item = frozen[label]
        path = base / item["path"]
        if not path.is_file():
            if require_runtime_inputs:
                raise FileNotFoundError(path)
            checked[label] = "MISSING"
            continue
        digest = sha256_file(path)
        if label == "prompt_file" and digest != item["sha256"] and not require_runtime_inputs:
            canonical = ("\n".join(path.read_text(encoding="utf-8").splitlines()) + "\n").encode("utf-8")
            digest = hashlib.sha256(canonical).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"C18-v2 {label} hash differs")
        checked[label] = "PASS"
    if read_sensitive and checked.get("token_inventory") == checked.get("batch_plan") == "PASS":
        validate_batch_plan(
            json.loads((base / frozen["token_inventory"]["path"]).read_text(encoding="utf-8")),
            json.loads((base / frozen["batch_plan"]["path"]).read_text(encoding="utf-8")),
        )
    if read_sensitive and checked.get("selection_plan") == "PASS":
        selection_inventory(base / frozen["selection_plan"]["path"])
    if checked.get("orthogonal_directions") == "PASS":
        with np.load(base / frozen["orthogonal_directions"]["path"], allow_pickle=False) as data:
            if data["directions"].shape != (5, 27, 3584) or data["directions"].dtype != np.float64:
                raise ValueError("C18-v2 orthogonal directions differ")
            directions = data["directions"].copy()
        if not np.isfinite(directions).all():
            raise ValueError("C18-v2 orthogonal directions are nonfinite")
        if not np.allclose(np.linalg.norm(directions, axis=-1), 1.0, rtol=0, atol=1e-12):
            raise ValueError("C18-v2 orthogonal directions are not unit normalized")
        if read_sensitive:
            import torch

            teacher_artifact = torch.load(
                base / frozen["teacher_tensor"]["path"], map_location="cpu", weights_only=True,
            )
            teacher = teacher_artifact[frozen["teacher_tensor"]["key"]]
            if tuple(teacher.shape) != tuple(frozen["teacher_tensor"]["shape"]):
                raise ValueError("C18-v2 teacher tensor shape differs")
            teacher = teacher[1:28].double()
            norms = torch.linalg.vector_norm(teacher, dim=-1)
            if not torch.isfinite(teacher).all() or torch.any(norms <= 1e-12):
                raise ValueError("C18-v2 teacher slots 1--27 are nonfinite or degenerate")
            teacher = (teacher / norms[:, None]).numpy()
            if np.max(np.abs(np.einsum("fsh,sh->fs", directions, teacher))) > 1e-12:
                raise ValueError("C18-v2 controls are not teacher-orthogonal")
    for condition, item in frozen["adapters"].items():
        path = base / item["path"]
        has_adapter = path.is_dir() and (path / "adapter_config.json").is_file() and any(
            (path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
        )
        digest = tree_digest(path) if has_adapter else None
        if digest is None:
            if require_runtime_inputs:
                raise FileNotFoundError(path)
            checked[f"adapter_{condition}"] = "MISSING"
        elif digest != item["tree_sha256"]:
            raise ValueError(f"C18-v2 adapter hash differs: {condition}")
        else:
            checked[f"adapter_{condition}"] = "PASS"
    for label in ("manifest", "subliminal_states", "neutral_states", "aggregate"):
        item = frozen["fsd"][label]
        path = base / item["path"]
        if not path.is_file():
            if require_runtime_inputs:
                raise FileNotFoundError(path)
            checked[f"fsd_{label}"] = "MISSING"
        elif sha256_file(path) != item["sha256"]:
            raise ValueError(f"C18-v2 FSD hash differs: {label}")
        else:
            checked[f"fsd_{label}"] = "PASS"
    model = frozen["model"]
    cache_root = Path(
        os.getenv("HUGGINGFACE_HUB_CACHE")
        or os.getenv("HF_HUB_CACHE")
        or Path(os.getenv("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    )
    model_root = cache_root / ("models--" + model["id"].replace("/", "--"))
    snapshot = model_root / "snapshots" / model["revision"]
    ref = model_root / "refs" / "main"
    if not snapshot.is_dir() or not ref.is_file():
        if require_runtime_inputs:
            raise FileNotFoundError(snapshot)
        checked["model_snapshot"] = "MISSING"
    else:
        if ref.read_text(encoding="utf-8") != model["revision"]:
            raise ValueError("C18-v2 model main revision differs")
        for name, expected in model["snapshot_files_sha256"].items():
            path = snapshot / name
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"C18-v2 model snapshot hash differs: {name}")
        checked["model_snapshot"] = "PASS"
    requirements = manifest["execution"]["requirements_lock"]
    if sha256_file(base / requirements["path"]) != requirements["sha256"]:
        raise ValueError("C18-v2 dependency-lock hash differs")
    checked["requirements_lock"] = "PASS"
    for label, item in manifest["implementation"].items():
        if label == "modules":
            for module_label, module_item in item.items():
                path = base / module_item["path"]
                if allow_execution_control_successor and module_item["path"] in EXECUTION_CONTROL_SUCCESSOR_PATHS:
                    checked[f"implementation_{module_label}"] = "CONTROL_SUCCESSOR"
                    continue
                if sha256_file(path) != module_item["sha256"]:
                    raise ValueError(f"C18-v2 implementation hash differs: {module_label}")
                checked[f"implementation_{module_label}"] = "PASS"
            continue
        path = base / item["path"]
        if allow_execution_control_successor and item["path"] in EXECUTION_CONTROL_SUCCESSOR_PATHS:
            checked[f"implementation_{label}"] = "CONTROL_SUCCESSOR"
            continue
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"C18-v2 implementation hash differs: {label}")
        checked[f"implementation_{label}"] = "PASS"
    return checked


def expected_raw_ids(manifest: Mapping[str, Any], root: str | Path) -> dict[str, set[str]]:
    frozen = manifest["frozen_inputs"]
    tokens = json.loads((Path(root) / frozen["token_inventory"]["path"]).read_text(encoding="utf-8"))
    sets = selection_inventory(Path(root) / frozen["selection_plan"]["path"])
    base = {f"{c}|{s['set_id']}|{p['prompt_id']}" for c in ("subliminal", "neutral")
            for s in sets for p in tokens["rows"]}
    return {
        "Y": {f"{item}|Y{a}{b}" for item in base for a in (0, 1) for b in (0, 1)},
        "Z": {f"{item}|Z{r}{b}" for item in base for r in (0, 1) for b in (0, 1)},
        "W": {f"{item}|W{a}{b}{r}" for item in base for a in (0, 1) for b in (0, 1) for r in (0, 1)},
        "orthogonal": {f"{item}|O{k}|Y{a}{b}" for item in base for k in range(5)
                       for a, b in ((0, 1), (1, 0))},
    }

"""Manifest validation and inventory expansion for C18."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .selection_plans import iter_selection_sets
from .teacher_coordinate_interchange import sha256_file


EXPERIMENT_ID = "qwen7b_cat_bidirectional_teacher_coordinate_interchange_v1"


def apply_storage_overrides(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Map only data/results/runs paths into the ignored cluster storage root."""
    shared_root = os.getenv("SLGEO_SHARED_ROOT")
    if not shared_root:
        return dict(manifest)
    root = shared_root.rstrip("/\\")

    def rewrite(value: Any) -> Any:
        if isinstance(value, str) and any(
            value == prefix or value.startswith(prefix + "/")
            for prefix in ("data", "results", "runs")
        ):
            return root + "/" + value.replace("\\", "/")
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, dict):
            return {key: rewrite(item) for key, item in value.items()}
        return value

    return rewrite(dict(manifest))


def tree_digest(path: str | Path) -> str | None:
    root = Path(path)
    if root.is_file():
        return sha256_file(root)
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    for child in sorted(
        item for item in root.rglob("*")
        if item.is_file() and item.name != "run_provenance.json" and not item.name.endswith(".provenance.json")
    ):
        digest.update(child.relative_to(root).as_posix().encode("utf-8"))
        digest.update(sha256_file(child).encode("ascii"))
    return digest.hexdigest()


def load_prompt_records(path: str | Path) -> list[dict[str, str]]:
    raw = Path(path).read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    required = {"prompt_id", "family", "prompt"}
    if len(records) != 72 or any(required - set(row) for row in records):
        raise ValueError("C18 prompt inventory must contain 72 complete records")
    for key in ("prompt_id", "prompt"):
        values = [str(row[key]) for row in records]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate prompt field: {key}")
    expected = {"direct_preference": 24, "identity_affinity": 24, "hypothetical_choice": 24}
    if Counter(str(row["family"]) for row in records) != Counter(expected):
        raise ValueError("C18 prompt-family inventory differs")
    return [{key: str(row[key]) for key in required} for row in records]


def selection_inventory(path: str | Path) -> list[dict[str, Any]]:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = list(iter_selection_sets(
        plan, set_names=("top_k", "norm_matched_control"), k_values=(20,)
    ))
    counts = Counter(row["set_name"] for row in rows)
    if counts != Counter({"top_k": 1, "norm_matched_control": 25}):
        raise ValueError(f"C18 selection counts differ: {counts}")
    if any(len(row["modules"]) != 20 or len(set(row["modules"])) != 20 for row in rows):
        raise ValueError("every C18 parameter set must contain 20 unique modules")
    identities = []
    for row in rows:
        draw = "top" if row["set_name"] == "top_k" else f"norm_{int(row['draw_id']):02d}"
        identities.append({**row, "set_id": draw})
    if len({row["set_id"] for row in identities}) != 26:
        raise ValueError("selection set IDs are not unique")
    return identities


def _expect(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"frozen {label} differs: expected {expected!r}, got {actual!r}")


def validate_manifest_contract(manifest: Mapping[str, Any]) -> None:
    _expect(manifest.get("schema_version"), 1, "schema version")
    _expect(manifest.get("experiment_id"), EXPERIMENT_ID, "experiment ID")
    design = manifest["design"]
    _expect(design["seed"], 2, "seed")
    _expect(design["k"], 20, "k")
    _expect(design["mode"], "necessity", "mode")
    _expect(design["cells"], ["Y00", "Y01", "Y10", "Y11"], "cell inventory")
    _expect(design["teacher_slots"], list(range(1, 28)), "teacher slots")
    _expect(design["hook_blocks"], list(range(27)), "hook blocks")
    _expect(design["epsilon"], 0.04405641704135471, "epsilon")
    _expect(design["bootstrap"], {
        "prng": "PCG64", "seed": 20260804, "draws": 20000, "stratified_family_size": 24
    }, "bootstrap")
    plan = design["scientific_batch_plan"] if "scientific_batch_plan" in design else manifest["execution"]["scientific_batch_plan"]
    _expect(plan["batch_size"], 6, "scientific batch size")
    _expect(manifest["execution"]["attention_backend"], "sdpa", "attention backend")
    _expect(manifest["execution"]["residual_dtype"], "float16", "residual dtype")
    if "@sha256:" not in manifest["execution"]["container_image"]:
        raise ValueError("container image is not digest-pinned")
    if manifest["frozen_inputs"]["orthogonal_directions"]["sha256"].startswith("TO_BE_"):
        raise ValueError("orthogonal direction hash is not frozen")


def validate_public_inputs(
    manifest: Mapping[str, Any], repo_root: str | Path, *, require_runtime_inputs: bool = False
) -> dict[str, object]:
    """Validate all locally available frozen inputs without reading outcomes."""
    root = Path(repo_root)
    frozen = manifest["frozen_inputs"]
    checked: dict[str, object] = {}
    for label, path_key, hash_key in (
        ("preregistration", "document", "document_sha256"),
        ("decision_matrix", "decision_matrix", "decision_matrix_sha256"),
    ):
        item = manifest["preregistration"]
        path = root / item[path_key]
        _expect(sha256_file(path), item[hash_key], f"{label} SHA-256")
        checked[label] = "PASS"
    for label in ("model_config", "prompt_file", "token_inventory", "selection_plan", "teacher_tensor", "orthogonal_directions"):
        item = frozen[label]
        path = root / item["path"]
        if not path.is_file():
            if require_runtime_inputs:
                raise FileNotFoundError(path)
            checked[label] = "MISSING"
            continue
        digest = sha256_file(path)
        if digest != item["sha256"] and label == "prompt_file" and not require_runtime_inputs:
            canonical = ("\n".join(path.read_text(encoding="utf-8").splitlines()) + "\n").encode("utf-8")
            if hashlib.sha256(canonical).hexdigest() == item["sha256"]:
                checked[label] = "PASS_CANONICAL_LF_WORKTREE_REQUIRES_STAGING"
                continue
        _expect(digest, item["sha256"], f"{label} SHA-256")
        checked[label] = "PASS"
    if str(checked.get("prompt_file", "")).startswith("PASS"):
        load_prompt_records(root / frozen["prompt_file"]["path"])
    if checked.get("selection_plan") == "PASS":
        selection_inventory(root / frozen["selection_plan"]["path"])
    if checked.get("orthogonal_directions") == "PASS":
        with np.load(root / frozen["orthogonal_directions"]["path"], allow_pickle=False) as data:
            if data["directions"].shape != (5, 27, 3584) or data["directions"].dtype != np.float64:
                raise ValueError("orthogonal direction array contract differs")
            directions = data["directions"]
        import torch
        teacher_artifact = torch.load(root / frozen["teacher_tensor"]["path"], map_location="cpu", weights_only=True)
        teacher = teacher_artifact[frozen["teacher_tensor"]["key"]]
        if tuple(teacher.shape) != tuple(frozen["teacher_tensor"]["shape"]):
            raise ValueError("teacher tensor shape differs")
        teacher = teacher[1:28].double()
        norms = torch.linalg.vector_norm(teacher, dim=-1)
        if not torch.isfinite(teacher).all() or torch.any(norms <= 1e-12):
            raise ValueError("teacher slots 1--27 are nonfinite or degenerate")
        teacher = (teacher / norms[:, None]).numpy()
        if np.max(np.abs(np.einsum("fsh,sh->fs", directions, teacher))) > 1e-12:
            raise ValueError("materialized controls are not teacher-orthogonal")
        if not np.allclose(np.linalg.norm(directions, axis=-1), 1.0, rtol=0, atol=1e-12):
            raise ValueError("materialized controls are not unit normalized")
    for condition, item in frozen["adapters"].items():
        path = root / item["path"]
        has_adapter = path.is_dir() and (path / "adapter_config.json").is_file() and any(
            (path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")
        )
        digest = tree_digest(path) if has_adapter else None
        if digest is None:
            if require_runtime_inputs:
                raise FileNotFoundError(path)
            checked[f"adapter_{condition}"] = "MISSING"
        else:
            _expect(digest, item["tree_sha256"], f"{condition} adapter tree")
            checked[f"adapter_{condition}"] = "PASS"
    for label in ("subliminal_states", "neutral_states", "aggregate"):
        item = frozen["fsd"][label]
        path = root / item["path"]
        if not path.is_file():
            if require_runtime_inputs:
                raise FileNotFoundError(path)
            checked[f"fsd_{label}"] = "MISSING"
        else:
            _expect(sha256_file(path), item["sha256"], f"FSD {label}")
            checked[f"fsd_{label}"] = "PASS"
    fsd_manifest = frozen["fsd"]
    fsd_path = root / fsd_manifest["manifest"]
    _expect(sha256_file(fsd_path), fsd_manifest["manifest_sha256"], "FSD manifest SHA-256")
    checked["fsd_manifest"] = "PASS"
    requirements = manifest["execution"]["requirements_lock"]
    _expect(sha256_file(root / requirements["path"]), requirements["sha256"], "requirements lock SHA-256")
    checked["requirements_lock"] = "PASS"

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
        _expect(ref.read_text(encoding="utf-8"), model["revision"], "model main revision")
        for name, expected in model["snapshot_files_sha256"].items():
            _expect(sha256_file(snapshot / name), expected, f"model snapshot {name}")
        checked["model_snapshot"] = "PASS"
    return checked


def expected_raw_ids(manifest: Mapping[str, Any], repo_root: str | Path) -> dict[str, set[str]]:
    prompts = load_prompt_records(Path(repo_root) / manifest["frozen_inputs"]["prompt_file"]["path"])
    sets = selection_inventory(Path(repo_root) / manifest["frozen_inputs"]["selection_plan"]["path"])
    base = {
        f"{condition}|{item['set_id']}|{prompt['prompt_id']}"
        for condition in ("subliminal", "neutral") for item in sets for prompt in prompts
    }
    return {
        "Y": {f"{prefix}|Y{a}{b}" for prefix in base for a in (0, 1) for b in (0, 1)},
        "Z": {f"{prefix}|Z{a}{b}" for prefix in base for a in (0, 1) for b in (0, 1)},
        "W": {f"{prefix}|W{a}{b}{r}" for prefix in base for a in (0, 1) for b in (0, 1) for r in (0, 1)},
        "orthogonal": {
            f"{prefix}|O{k}|Y{a}{b}" for prefix in base for k in range(5) for a, b in ((0, 1), (1, 0))
        },
    }

"""Verified access to the CTS Stage-0 v2 scientific contract (``research/cts_stage0_v2``).

The contract is the decision spec plus the condition registry generated from it. Both are pinned by SHA-256
in the execution manifest. The v1 frozen package supplies the reused inputs; every reused file must match
the hash the v2 spec records (``frozen_v1_inputs_by_hash``) before it is used.
"""

from __future__ import annotations

from .errors import FinalFailure

import importlib.util
import json
from functools import cached_property
from pathlib import Path
from typing import Any, Mapping

from .package import FrozenPackage, sha256_path

CONTRACT_RELATIVE_PATH = "research/cts_stage0_v2"
SPEC_NAME = "cts_stage0_v2_decision_spec.json"
REGISTRY_NAME = "cts_stage0_v2_condition_registry.jsonl"
GENERATOR_NAME = "tools/generate_registry.py"
SPEC_VERSION = "cts-stage0-v2"


class ContractError(RuntimeError, FinalFailure):
    """The v2 contract, or a v1 input it reuses, differs from its pinned hash or from itself."""

    event = "refusal"


class V2Contract:
    """The v2 spec and registry at ``root``; construction verifies pins and every reused v1 input."""

    def __init__(self, root: str | Path, package: FrozenPackage, pins: Mapping[str, str] | None = None):
        self.root = Path(root).resolve()
        self.package = package
        spec_path, registry_path = self.root / SPEC_NAME, self.root / REGISTRY_NAME
        for path in (spec_path, registry_path):
            if not path.is_file():
                raise ContractError(f"Contract file missing: {path.name}")
        self.spec_sha256 = sha256_path(spec_path)
        self.registry_sha256 = sha256_path(registry_path)
        if pins is not None:
            if pins.get("spec_sha256") != self.spec_sha256:
                raise ContractError("Decision spec differs from the pinned hash")
            if pins.get("registry_sha256") != self.registry_sha256:
                raise ContractError("Condition registry differs from the pinned hash")
        self.spec: dict[str, Any] = json.loads(spec_path.read_text(encoding="utf-8"))
        if self.spec.get("spec_version") != SPEC_VERSION:
            raise ContractError(f"Unexpected spec version {self.spec.get('spec_version')!r}")
        self._verify_v1_inputs()
        self.rows: list[dict[str, Any]] = [
            json.loads(line) for line in registry_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        cids = [row["cid"] for row in self.rows]
        if len(cids) != len(set(cids)):
            raise ContractError("Duplicate condition ids in the registry")

    @classmethod
    def from_repo(cls, repo_root: str | Path, package: FrozenPackage, pins: Mapping[str, str] | None = None) -> "V2Contract":
        return cls(Path(repo_root) / CONTRACT_RELATIVE_PATH, package, pins)

    def _verify_v1_inputs(self) -> None:
        reused = self.spec["frozen_v1_inputs_by_hash"]
        for name, expected in reused["files"].items():
            recorded = self.package.file_hashes.get(name)
            if recorded != expected:
                raise ContractError(f"v1 manifest hash of {name} differs from the v2 spec")
            if sha256_path(self.package.root / name) != expected:
                raise ContractError(f"Reused v1 input {name} differs from its hash")
        external = reused["external_inputs"]
        v1 = self.package.external_inputs
        pairs = [
            (external["model_revision"], v1["model_revision"]),
            (external["extraction_file_sha256"], v1["extraction_prompts"]["file_sha256"]),
            (external["extraction_prompts_json_sha256"], v1["extraction_prompts"]["prompts_json_sha256"]),
            (external["tokenizer_files_sha256"], v1["tokenizer_files_sha256"]),
        ]
        if any(a != b for a, b in pairs):
            raise ContractError("External inputs of the v2 spec differ from the v1 manifest")

    @cached_property
    def null_words(self) -> list[str]:
        singular = {entry["plural"]: entry["singular"] for entry in self.package.null_pool["log"]}
        return [singular[plural] for plural in self.package.null_pool["selected_16"]]

    def regenerated_rows(self) -> list[dict[str, Any]]:
        """Rows produced by the frozen generator from this spec (static reproducibility check)."""
        spec = importlib.util.spec_from_file_location("cts_v2_generate_registry", self.root / GENERATOR_NAME)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.build(self.spec, self.null_words)

    def verify_regeneration(self) -> None:
        if self.regenerated_rows() != self.rows:
            raise ContractError("The condition registry is not reproduced by the generator from the spec")

    @property
    def personas(self) -> list[str]:
        return list(self.spec["personas"]["extracted"])

    @property
    def planning_seconds(self) -> dict[str, float]:
        return dict(self.spec["condition_registry"]["planning_seconds_placeholder"])

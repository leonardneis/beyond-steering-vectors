"""Verified, read-only access to the frozen CTS Stage-0 package.

Every accessor fails closed. The prompt pool file contains the S0, D and C partitions inline; only
``s0_prompts`` reads prompt text from it, and it drops every non-S0 record before returning. The
analysis side needs only ids (``partition_ids``), which come from the partition file (no text).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

PACKAGE_RELATIVE_PATH = "research/cts_stage0_v1"
PREREG_TAG = "prereg/cts-stage0-v1"
PREREG_COMMIT = "43dd95d3d476a9db9ebb9f8d0aa5639c65a1c1ab"
MANIFEST_SHA256 = "6685d45685f3834b08d11641bbf058296ecfc34af48739a112f155bc27916ca0"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
ANIMAL_FAMILIES = ("direct", "identity", "hypothetical")
DEFAULT_PERSONA_ID = "P_default"
DEFAULT_SYSTEM_TEXT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
LAST_THREE_PROMPT_IDS = (151644, 77091, 198)


class FrozenPackageError(RuntimeError):
    """Raised when the frozen package, or an input it pins, does not match its recorded hash."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_hash(hashes: list[str]) -> str:
    """Hash-of-hashes convention of the frozen partition file: sha256(json.dumps(sorted(list)))."""
    return sha256_text(json.dumps(sorted(hashes)))


@dataclass(frozen=True)
class EvalPrompt:
    prompt_id: str
    subfamily: str
    prompt: str
    sha256: str

    @property
    def is_animal_family(self) -> bool:
        return self.subfamily in ANIMAL_FAMILIES


class FrozenPackage:
    """The frozen package at ``root``; construction verifies MANIFEST and every listed file."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        manifest_path = self.root / "MANIFEST.json"
        if not manifest_path.is_file():
            raise FrozenPackageError(f"MANIFEST.json missing under {self.root}")
        if sha256_path(manifest_path) != MANIFEST_SHA256:
            raise FrozenPackageError("MANIFEST.json does not match the frozen manifest hash")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        listed = self.manifest["files"]
        present = {
            path.relative_to(self.root).as_posix()
            for path in self.root.rglob("*")
            if path.is_file() and path.name != "MANIFEST.json"
        }
        extra = sorted(present - set(listed))
        missing = sorted(set(listed) - present)
        if extra or missing:
            raise FrozenPackageError(f"Frozen package file set differs: extra={extra} missing={missing}")
        for relative, expected in listed.items():
            if sha256_path(self.root / relative) != expected:
                raise FrozenPackageError(f"Frozen file hash mismatch: {relative}")
        self.file_hashes = dict(listed)

    @classmethod
    def from_repo(cls, repo_root: str | Path) -> "FrozenPackage":
        return cls(Path(repo_root) / PACKAGE_RELATIVE_PATH)

    def _json(self, name: str) -> Any:
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    @cached_property
    def spec(self) -> dict[str, Any]:
        return self._json("cts_stage0_decision_spec.json")

    @cached_property
    def personas_file(self) -> dict[str, Any]:
        return self._json("cts_stage0_personas.json")

    @cached_property
    def personas(self) -> dict[str, dict[str, Any]]:
        entries = self.personas_file["personas"]
        by_id = {entry["id"]: entry for entry in entries}
        if len(by_id) != len(entries):
            raise FrozenPackageError("Duplicate persona ids")
        return by_id

    def persona(self, persona_id: str) -> dict[str, Any]:
        try:
            return self.personas[persona_id]
        except KeyError as exc:
            raise FrozenPackageError(f"Unknown persona id: {persona_id!r}") from exc

    @cached_property
    def endpoint(self) -> dict[str, Any]:
        return self._json("cts_stage0_endpoint_tokens.json")

    @cached_property
    def null_pool(self) -> dict[str, Any]:
        return self._json("cts_stage0_null_pool.json")

    @cached_property
    def partition(self) -> dict[str, Any]:
        return self._json("cts_stage0_partition.json")

    @cached_property
    def parser_lexicon(self) -> dict[str, Any]:
        return self._json("cts_stage0_parser_lexicon.json")

    @property
    def external_inputs(self) -> dict[str, Any]:
        return self.manifest["external_inputs"]

    def partition_ids(self) -> dict[str, tuple[str, ...]]:
        """S0/D/C prompt ids from the partition file (ids only; no prompt text is read)."""
        sets = self.partition["sets"]
        out = {name: tuple(sets[name]["prompt_ids"]) for name in ("S0", "D", "C")}
        if len(set(out["S0"]) | set(out["D"]) | set(out["C"])) != sum(len(v) for v in out.values()):
            raise FrozenPackageError("Partition sets overlap")
        return out

    def s0_prompts(self) -> tuple[EvalPrompt, ...]:
        """The S0 evaluation prompts, sorted by prompt_id; non-S0 records are dropped unread.

        A non-S0 line is parsed only to read its ``set`` and ``prompt_id`` (for the consistency
        check); its text is discarded immediately and never returned, logged or tokenized.
        """
        partition = self.partition_ids()
        s0_ids = set(partition["S0"])
        other_ids: set[str] = set()
        records: list[EvalPrompt] = []
        with (self.root / "cts_stage0_prompts.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["set"] != "S0":
                    other_ids.add(row["prompt_id"])
                    del row
                    continue
                if row["prompt_id"] not in s0_ids:
                    raise FrozenPackageError(f"S0 row not in the S0 partition list: {row['prompt_id']}")
                if sha256_text(row["prompt"]) != row["sha256"]:
                    raise FrozenPackageError(f"Prompt hash mismatch: {row['prompt_id']}")
                records.append(EvalPrompt(row["prompt_id"], row["subfamily"], row["prompt"], row["sha256"]))
        ids = [record.prompt_id for record in records]
        if sorted(ids) != sorted(s0_ids) or len(ids) != len(set(ids)):
            raise FrozenPackageError("S0 prompt ids do not equal the partition S0 list")
        if other_ids != set(partition["D"]) | set(partition["C"]):
            raise FrozenPackageError("Non-S0 rows do not equal the D and C partition lists")
        expected = self.partition["sets"]["S0"]["prompt_sha256_list_sha256"]
        if list_hash([record.sha256 for record in records]) != expected:
            raise FrozenPackageError("S0 hash-of-hashes mismatch")
        return tuple(sorted(records, key=lambda record: record.prompt_id))

    def validation_prompts(self) -> tuple[EvalPrompt, ...]:
        """The 40 technical-validation prompts V (never used by the scientific run)."""
        records = []
        for line in (self.root / "cts_stage0_validation_prompts.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if sha256_text(row["prompt"]) != row["sha256"]:
                raise FrozenPackageError(f"V prompt hash mismatch: {row['prompt_id']}")
            records.append(EvalPrompt(row["prompt_id"], row["subfamily"], row["prompt"], row["sha256"]))
        if list_hash([record.sha256 for record in records]) != self.partition["validation"]["sha256_list_sha256"]:
            raise FrozenPackageError("V hash-of-hashes mismatch")
        return tuple(sorted(records, key=lambda record: record.prompt_id))

    def dc_fingerprints(self) -> dict[str, set[str]]:
        """D/C ids, per-record hashes and prompt texts, for the output leak scan only (never forwarded)."""
        out: dict[str, set[str]] = {"ids": set(), "sha256": set(), "texts": set()}
        with (self.root / "cts_stage0_prompts.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["set"] in {"D", "C"}:
                    out["ids"].add(row["prompt_id"])
                    out["sha256"].add(row["sha256"])
                    out["texts"].add(row["prompt"])
                    out["texts"].add(row["stem"])
        return out


def load_extraction_prompts(package: FrozenPackage, extraction_file: str | Path) -> tuple[str, ...]:
    """Rows 0-1023, field ``prompt`` only, verified against the frozen file and list hashes."""
    spec = package.external_inputs["extraction_prompts"]
    path = Path(extraction_file)
    if sha256_path(path) != spec["file_sha256"]:
        raise FrozenPackageError("Extraction file hash mismatch")
    prompts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(prompts) == 1024:
                break
            prompts.append(json.loads(line)["prompt"])
    if len(prompts) != 1024:
        raise FrozenPackageError("Extraction file has fewer than 1024 rows")
    if sha256_text(json.dumps(prompts, ensure_ascii=False)) != spec["prompts_json_sha256"]:
        raise FrozenPackageError("Extraction prompts list hash mismatch")
    return tuple(prompts)

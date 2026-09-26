"""Deterministic job plan: a pure function of the v2 contract (spec + registry) and the execution manifest.

Serialized as canonical JSON; its SHA-256 is carried by every shard marker. Shards are sized from the contract's
planning seconds so that one shard's planned compute times the overhead factor stays within
``shard_fraction_of_retirement`` of the retirement time (engineering requirement E3).
"""

from __future__ import annotations

from .errors import FinalFailure

import math
from dataclasses import dataclass
from typing import Any

from .artifacts import canonical_json, sha256_bytes
from .budget import N_EXTRACTION_ROWS, PROMPTS, projection_a100_h
from .conditions import L2, OWN, PERSONA, STEER, UNSTEERED, Condition, registry_conditions
from .contract import V2Contract

SCORING_STAGES = ("score", "rescore")
GPU_STAGES = ("extract", "baseline", "score", "rescore")


class PlanError(RuntimeError, FinalFailure):
    pass


def null_names(contract: V2Contract) -> list[str]:
    words = contract.null_words
    return [f"null:{a}>{b}" for a in words for b in words if a != b]


@dataclass(frozen=True)
class ShardSpec:
    shard_id: str
    stage: str
    gpu: bool
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"shard_id": self.shard_id, "stage": self.stage, "gpu": self.gpu, "payload": self.payload}

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json(self.as_dict()))


def per_shard(seconds_per_unit: float, units_per_item: int, overhead: float, fraction: float, retirement: float) -> int:
    """Items per shard so that items x units x seconds x overhead <= fraction x retirement (at least 1)."""
    if seconds_per_unit <= 0 or overhead < 1 or not 0 < fraction <= 1:
        raise PlanError("Invalid shard-sizing parameters")
    return max(1, math.floor(fraction * retirement / (units_per_item * seconds_per_unit * overhead)))


def _chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def build_plan(contract: V2Contract, manifest: dict) -> dict[str, Any]:
    conditions = registry_conditions(contract.rows)
    seconds = contract.planning_seconds
    overhead = float(seconds["overhead_factor"])
    fraction = float(manifest["plan"]["shard_fraction_of_retirement"])
    retirement = float(manifest["hpc"]["retirement_seconds"])

    shards: list[ShardSpec] = [ShardSpec("preflight", "preflight", False, {})]
    personas = contract.personas
    size = per_shard(float(seconds["extraction_forward"]), N_EXTRACTION_ROWS, overhead, fraction, retirement)
    for index, group in enumerate(_chunks(personas, size)):
        shards.append(ShardSpec(f"extract_{index:02d}", "extract", True, {"personas": group}))
    shards.append(ShardSpec("directions", "directions", False, {}))
    for rep in (1, 2):
        shards.append(ShardSpec(f"baseline_rep{rep}", "baseline", True, {"rep": rep}))

    by_key: dict[tuple[str, str, str], list[str]] = {}
    for condition in conditions:
        if condition.kind == UNSTEERED:
            continue  # scored by the baseline shards
        by_key.setdefault((condition.group, condition.cost_class, condition.prompt_set), []).append(condition.cid)
    for (group, cost_class, prompt_set), ids in sorted(by_key.items()):
        size = per_shard(float(seconds[cost_class]), PROMPTS[prompt_set], overhead, fraction, retirement)
        safe = group.replace(":", "-")
        tag = "l2" if cost_class == L2 else "own"
        for index, chunk in enumerate(_chunks(ids, size)):
            shards.append(ShardSpec(f"score_{safe}_{tag}_{index:03d}", "score", True,
                                    {"conditions": chunk, "cost_class": cost_class, "prompt_set": prompt_set}))

    rescore: dict[str, list[str]] = {}
    for condition in conditions:
        if condition.reference_rescore:
            rescore.setdefault(condition.prompt_set, []).append(condition.cid)
    for prompt_set, ids in sorted(rescore.items()):
        size = per_shard(float(seconds["L1_reference"]), PROMPTS[prompt_set], overhead, fraction, retirement)
        for index, chunk in enumerate(_chunks(ids, size)):
            shards.append(ShardSpec(f"rescore_{prompt_set}_{index:03d}", "rescore", True,
                                    {"conditions": chunk, "prompt_set": prompt_set}))

    shards.append(ShardSpec("fragility", "fragility", False, {}))
    shards.append(ShardSpec("integrity", "integrity", False, {}))
    shards.append(ShardSpec("analysis", "analysis", False, {}))

    ids = [shard.shard_id for shard in shards]
    if len(ids) != len(set(ids)):
        raise PlanError("Duplicate shard ids")
    scored = sorted(cid for shard in shards if shard.stage == "score" for cid in shard.payload["conditions"])
    expected = sorted(c.cid for c in conditions if c.kind != UNSTEERED)
    if scored != expected:
        raise PlanError("Scoring shards do not cover the condition registry exactly")
    rescored = sorted(cid for shard in shards if shard.stage == "rescore" for cid in shard.payload["conditions"])
    if rescored != sorted(c.cid for c in conditions if c.reference_rescore):
        raise PlanError("Re-score shards do not cover the flagged conditions exactly")
    for condition in conditions:
        if condition.kind == PERSONA and condition.cost_class != OWN:
            raise PlanError("Persona conditions must be own-prefix")
        if condition.kind == STEER and condition.mode == "all" and condition.cost_class != OWN:
            raise PlanError("All-position conditions must be own-prefix")
    return {
        "experiment_id": manifest["experiment_id"],
        "contract": {"spec_sha256": contract.spec_sha256, "registry_sha256": contract.registry_sha256},
        "n_conditions": len(conditions),
        "n_gating_conditions": sum(c.gating for c in conditions),
        "conditions": [c.as_dict() for c in conditions],
        "shards": [dict(shard.as_dict(), spec_sha256=shard.sha256) for shard in shards],
        "sizing": {"seconds": seconds, "overhead_factor": overhead, "fraction": fraction, "retirement_seconds": retirement},
    }


def plan_sha256(plan: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(plan))


def projection(plan: dict[str, Any], *, seconds: dict[str, float] | None = None, overhead: float | None = None) -> float:
    """Planned A100-h of the whole plan (warm seconds x overhead factor), optionally with measured seconds."""
    return projection_a100_h(plan, seconds or plan["sizing"]["seconds"], overhead if overhead is not None else plan["sizing"]["overhead_factor"])


def condition_objects(plan: dict[str, Any]) -> dict[str, Condition]:
    return {entry["cid"]: Condition(**{k: v for k, v in entry.items() if k != "cid"}) for entry in plan["conditions"]}

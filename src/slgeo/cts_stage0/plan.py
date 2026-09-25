"""Deterministic job plan: a pure function of the frozen package, the implementation choices and the
execution manifest. Serialized as canonical JSON; its SHA-256 is carried by every shard marker.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import canonical_json, sha256_bytes
from .conditions import Condition, all_conditions
from .package import FrozenPackage

EXTRACTION_GROUPS = (
    ("extract_default", lambda pid: pid == "P_default"),
    ("extract_catdogwolf", lambda pid: any(pid.startswith(f"{p}_") for p in ("P_cat", "P_dog", "P_wolf", "M"))),
    ("extract_panel", lambda pid: pid.startswith(("P_lion", "P_horse", "P_rabbit", "P_elephant", "P_fox", "P_owl"))),
    ("extract_null", lambda pid: pid.startswith("N_")),
    ("extract_controls", lambda pid: pid in ("P_chess", "P_blue", "P_qwencat", "P_helpful", "P_id")),
)


class PlanError(RuntimeError):
    pass


def null_words(package: FrozenPackage) -> list[str]:
    singular = {entry["plural"]: entry["singular"] for entry in package.null_pool["log"]}
    return [singular[plural] for plural in package.null_pool["selected_16"]]


def null_names(package: FrozenPackage) -> list[str]:
    words = null_words(package)
    return [f"null:{a}>{b}" for a in words for b in words if a != b]


def load_choices(repo_root: Path, manifest: dict) -> dict:
    return json.loads((repo_root / manifest["implementation_choices"]).read_text(encoding="utf-8"))


def conditions_for(package: FrozenPackage, choices: dict) -> list[Condition]:
    return all_conditions(null_names(package), sorted(package.personas), choices["descriptive"])


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


def build_plan(package: FrozenPackage, choices: dict, manifest: dict, *, conditions_per_shard: int, sampling_shards: int = 2) -> dict[str, Any]:
    if conditions_per_shard < 1:
        raise PlanError("conditions_per_shard must be positive")
    shards: list[ShardSpec] = [ShardSpec("preflight", "preflight", False, {})]
    personas = sorted(package.personas)
    assigned: set[str] = set()
    for shard_id, member in EXTRACTION_GROUPS:
        group = [pid for pid in personas if member(pid)]
        assigned.update(group)
        shards.append(ShardSpec(shard_id, "extract", True, {"personas": group}))
    if assigned != set(personas):
        raise PlanError(f"Personas without an extraction shard: {sorted(set(personas) - assigned)}")
    shards.append(ShardSpec("directions", "directions", False, {}))
    for rep in (1, 2):
        shards.append(ShardSpec(f"baseline_rep{rep}", "baseline", True, {"rep": rep}))
    conditions = conditions_for(package, choices)
    persona_conditions = [c.cid for c in conditions if c.kind == "persona"]
    shards.append(ShardSpec("score_persona", "score_persona", True, {"conditions": persona_conditions}))
    steered = [c for c in conditions if c.kind == "steer"]
    by_group: dict[str, list[str]] = {}
    for condition in steered:
        by_group.setdefault(condition.group, []).append(condition.cid)
    for group in sorted(by_group):
        ids = by_group[group]
        for index in range(0, len(ids), conditions_per_shard):
            chunk = ids[index : index + conditions_per_shard]
            safe = group.replace(":", "-")
            shards.append(ShardSpec(f"score_{safe}_{index // conditions_per_shard:03d}", "score", True, {"conditions": chunk}))
    sampled = sampling_condition_ids(conditions)
    for index in range(sampling_shards):
        shards.append(ShardSpec(f"sample_{index:02d}", "sample", True, {"conditions": sampled[index::sampling_shards]}))
    shards.append(ShardSpec("integrity", "integrity", False, {}))
    shards.append(ShardSpec("analysis", "analysis", False, {}))
    ids = [shard.shard_id for shard in shards]
    if len(ids) != len(set(ids)):
        raise PlanError("Duplicate shard ids")
    scored = sorted(cid for shard in shards if shard.stage in ("score", "score_persona") for cid in shard.payload["conditions"])
    expected = sorted(c.cid for c in conditions if c.kind != "unsteered")
    if scored != expected:
        raise PlanError("Scoring shards do not cover the condition registry exactly")
    from .criteria import required_condition_ids

    missing = required_condition_ids(null_names(package)) - {c.cid for c in conditions}
    if missing:
        raise PlanError(f"Gating conditions missing from the plan: {sorted(missing)[:5]}")
    pinned = choices["condition_counts"]
    if len(conditions) != pinned["total"] or sum(c.gating for c in conditions) != pinned["gating"]:
        raise PlanError("Condition counts differ from the pinned counts in IMPLEMENTATION_CHOICES.json")
    return {
        "experiment_id": manifest["experiment_id"],
        "n_conditions": len(conditions),
        "n_gating_conditions": sum(c.gating for c in conditions),
        "conditions": [c.as_dict() for c in conditions],
        "shards": [dict(shard.as_dict(), spec_sha256=shard.sha256) for shard in shards],
        "conditions_per_shard": conditions_per_shard,
    }


def sampling_condition_ids(conditions: list[Condition]) -> list[str]:
    wanted = []
    for condition in conditions:
        if condition.kind == "persona" and condition.persona in ("P_cat_T1", "P_dog_T1", "P_wolf_T1"):
            wanted.append(condition.cid)
        elif (
            condition.kind == "steer"
            and condition.slot == 14
            and condition.mode == "last"
            and condition.kappa == 1.0
            and (
                (condition.direction in ("c_cat_dog", "c_cat_wolf", "c_cat_anim") and condition.magnitude == f"tau:{condition.direction}")
                or (condition.direction == "t_cat" and condition.scale == "raw")
            )
        ):
            wanted.append(condition.cid)
    return ["unsteered"] + wanted


def plan_sha256(plan: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(plan))

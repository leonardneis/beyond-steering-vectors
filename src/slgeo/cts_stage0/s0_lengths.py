"""The S0 length profile consumed by TV-v2 (clarification of 2026-09-26, README "S0 length profile").

The profile is generated before TV-v2 by ``scripts/cts_stage0_s0_length_profile.py`` (tokenizer only: no model
weights, no forward). It holds, for every GPU prompt evaluation of an S0 prompt in the scientific plan, the rendered
prompt length under the rendering the execution code uses, aggregated per (cost class, prompt set, context) as
counts per length. It contains no prompt text, no prompt id and no persona id.

This module only reads and checks the committed profile; it never reads S0, D or C. TV-v2 uses the profile to build
length-matched stand-in prompts from V text and to weight the measured per-length seconds of each cost class by
the planned workload.
"""

from __future__ import annotations

from typing import Any, Mapping

from .budget import PROMPTS
from .errors import FinalFailure

KIND = "cts_stage0_v2_s0_length_profile"
SCHEMA_VERSION = 1
PROFILE_PATH = "research/cts_stage0_v2_execution/s0_length_profile.json"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
TOKENIZER_PACKAGES = ("transformers", "tokenizers")
S0_CLASSES = ("L2_shared_prefix", "own_prefix", "L1_reference")
CONTEXTS = ("default", "persona")
DRY_SHARD_GROUP = ("L2_shared_prefix", "S0_animal", "default")


class LengthProfileError(RuntimeError, FinalFailure):
    """The committed S0 length profile does not match the contract, the plan or the execution manifest."""


def planned_evaluations(plan: Mapping[str, Any]) -> dict[tuple[str, str], int]:
    """S0 prompt evaluations per (cost class, prompt set) of the plan's GPU shards (terms of the projection)."""
    totals: dict[tuple[str, str], int] = {}

    def add(key: tuple[str, str], count: int) -> None:
        totals[key] = totals.get(key, 0) + count

    for shard in plan["shards"]:
        stage, payload = shard["stage"], shard["payload"]
        if stage == "baseline":
            add(("L2_shared_prefix", "S0_all"), PROMPTS["S0_all"])
        elif stage == "score":
            add((payload["cost_class"], payload["prompt_set"]), len(payload["conditions"]) * PROMPTS[payload["prompt_set"]])
        elif stage == "rescore":
            add(("L1_reference", payload["prompt_set"]), len(payload["conditions"]) * PROMPTS[payload["prompt_set"]])
    return totals


def verify(profile: Mapping[str, Any], *, spec_sha256: str, registry_sha256: str, package_manifest_sha256: str,
           manifest: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the profile belongs to this contract, tokenizer, package versions and plan."""
    if profile.get("kind") != KIND or profile.get("schema_version") != SCHEMA_VERSION:
        raise LengthProfileError("Unexpected S0 length profile kind or schema")
    source = profile["provenance"]
    expected = {
        "spec_sha256": spec_sha256,
        "registry_sha256": registry_sha256,
        "frozen_package_manifest_sha256": package_manifest_sha256,
        "model_revision": manifest["model"]["revision"],
        "tokenizer_files_sha256": {name: manifest["model"]["snapshot_sha256"][name] for name in TOKENIZER_FILES},
        "packages": {name: manifest["execution"]["packages"][name] for name in TOKENIZER_PACKAGES},
    }
    for key, value in expected.items():
        if source.get(key) != value:
            raise LengthProfileError(f"S0 length profile provenance differs: {key}")
    observed: dict[tuple[str, str], int] = {}
    for group in profile["groups"]:
        if group["cost_class"] not in S0_CLASSES or group["prompt_set"] not in PROMPTS or group["context"] not in CONTEXTS:
            raise LengthProfileError("Unknown S0 length profile group")
        pairs = group["evaluations_by_length"]
        if any(not (isinstance(n, int) and isinstance(c, int) and n > 0 and c > 0) for n, c in pairs):
            raise LengthProfileError("S0 length profile entries must be positive integers")
        total = sum(c for _, c in pairs)
        if total % PROMPTS[group["prompt_set"]] or total // PROMPTS[group["prompt_set"]] != group["conditions"]:
            raise LengthProfileError("S0 length profile counts differ from conditions x prompt-set size")
        key = (group["cost_class"], group["prompt_set"])
        observed[key] = observed.get(key, 0) + total
    if observed != planned_evaluations(plan):
        raise LengthProfileError("S0 length profile workload differs from the plan")
    return {"pass": True, "groups": len(profile["groups"]), "evaluations": sum(observed.values()),
            "distinct_lengths": len(distinct_lengths(profile))}


def class_weights(profile: Mapping[str, Any]) -> dict[str, dict[int, int]]:
    """Planned evaluations per rendered length for each S0 cost class."""
    weights: dict[str, dict[int, int]] = {name: {} for name in S0_CLASSES}
    for group in profile["groups"]:
        target = weights[group["cost_class"]]
        for length, count in group["evaluations_by_length"]:
            target[length] = target.get(length, 0) + count
    return weights


def distinct_lengths(profile: Mapping[str, Any]) -> list[int]:
    return sorted({length for group in profile["groups"] for length, _ in group["evaluations_by_length"]})


def dry_shard_lengths(profile: Mapping[str, Any]) -> list[int]:
    """Rendered lengths of the prompts of one planned L2 S0_animal score shard, ascending (one per prompt)."""
    for group in profile["groups"]:
        if (group["cost_class"], group["prompt_set"], group["context"]) == DRY_SHARD_GROUP:
            lengths = []
            for length, count in group["evaluations_by_length"]:
                if count % group["conditions"]:
                    raise LengthProfileError("Dry-shard group is not a single prompt multiset")
                lengths += [length] * (count // group["conditions"])
            if len(lengths) != PROMPTS["S0_animal"]:
                raise LengthProfileError("Dry-shard group does not cover the S0_animal prompt set")
            return lengths
    raise LengthProfileError("S0 length profile has no L2 S0_animal default group")


def weighted_mean(values: Mapping[int, float], weights: Mapping[int, int]) -> float:
    """Workload-weighted mean of per-length measurements (every weighted length must be measured)."""
    total = sum(weights.values())
    return sum(values[length] * count for length, count in weights.items()) / total

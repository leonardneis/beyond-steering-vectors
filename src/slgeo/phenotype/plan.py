"""Deterministic execution plan of the Phenotype Anchor: arms x cells x stems -> contexts -> shards, per stage.

One plan per stage (``stages.PLAN``): the stage's arms come from ``stages`` in the execution manifest (p1: base,
teachers, library personas, N/S 1-3 and the number capture; p2: D1-D3; p1-seeds45: N4, S4, N5, S5). Contexts are
ordered (arm, stem order of the prompt manifest, cell order), so every arm sees the same stems in the same order; the
sampler's batch composition therefore does not depend on the arm (common random numbers). Shards group one arm's
work of one kind (``score``, ``sample``, ``numcap``) and are split to stay below the configured time per shard with
the seconds per unit measured in TV (placeholders before TV).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import stages as stage_layout
from .runner import Context

PLACEHOLDER_SECONDS = {"score": 0.4, "capture": 0.2, "sample": 0.12, "numcap": 0.2, "load": 120.0}
KINDS = ("score", "sample", "numcap")


class PlanError(ValueError):
    pass


def load_prompt_manifest(path: str | Path, expected_sha256: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != expected_sha256:
        raise PlanError("Prompt manifest hash mismatch")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def contexts_for_arm(arm: str, context: str, entries: Sequence[Mapping[str, Any]], cells: Mapping[str, Any]) -> list[Context]:
    """``context == "rendering"``: the 2 x 2 cells (Q/H x prefix none/r0/r1/r2); otherwise one persona crossed with
    the prefixes. Cell labels: "Q+r0", "Q+none", "H+r1", "persona+r0", ..."""
    out = []
    prefixes = list(cells["prefixes"])
    for e in entries:
        if context == "rendering":
            grid = [(f"{name}+{p}", persona, p) for name, persona in cells["rendering_personas"].items() for p in prefixes]
        else:
            grid = [(f"persona+{p}", context, p) for p in prefixes]
        for cell, persona, p in grid:
            out.append(Context(f"{arm}|{e['stem_id']}|{cell}", e["stem_id"], cell, p, persona, e["prompts"][p]))
    return out


def check_stages(config: Mapping[str, Any]) -> None:
    """The manifest's stage assignment: every arm in exactly one stage, every stage arm defined, every adapter an arm
    uses declared, the number capture and the sampled arms defined (number capture in p1 only)."""
    assignment = config.get("stages") or {}
    if set(assignment) != set(stage_layout.STAGES):
        raise PlanError(f"stages must name exactly {', '.join(stage_layout.STAGES)}")
    seen: dict[str, str] = {}
    for stage, arms in assignment.items():
        for arm in arms:
            if arm in seen:
                raise PlanError(f"Arm {arm} is in stages {seen[arm]} and {stage}")
            if arm not in config["arms"]:
                raise PlanError(f"Stage {stage} names the undefined arm {arm}")
            seen[arm] = stage
    unassigned = sorted(set(config["arms"]) - set(seen))
    if unassigned:
        raise PlanError(f"Arms without a stage: {', '.join(unassigned)}")
    for arm, (adapter, _context) in config["arms"].items():
        if adapter is not None and adapter not in config["adapters"]:
            raise PlanError(f"Arm {arm} uses the undeclared adapter {adapter}")
    for arm in config["number_capture"]["arms"]:
        if seen.get(arm) != "p1":
            raise PlanError(f"Number-capture arm {arm} is not a p1 arm")
    undefined = sorted(set(config["sampling"]["arms"]) - set(config["arms"]))
    if undefined:
        raise PlanError(f"Sampled arms without a definition: {', '.join(undefined)}")


def stage_of_arm(config: Mapping[str, Any], arm: str) -> str:
    for stage, arms in config["stages"].items():
        if arm in arms:
            return stage
    raise PlanError(f"Arm {arm} has no stage")


def stage_adapters(config: Mapping[str, Any], stage: str) -> list[str]:
    """Adapters used by the arms of one stage, in arm order (what ``pin-adapters --stage`` pins)."""
    names = [config["arms"][a][0] for a in config["stages"][stage_layout.check(stage)]]
    return list(dict.fromkeys(n for n in names if n is not None))


@dataclass(frozen=True)
class Shard:
    shard_id: str
    arm: str
    kind: str
    adapter: str | None
    context_ids: tuple[str, ...]
    projected_seconds: float


def build_plan(config: Mapping[str, Any], entries: Sequence[Mapping[str, Any]], *, stage: str = "p1",
               seconds: Mapping[str, float] = PLACEHOLDER_SECONDS, arm_factor: Mapping[str, float] | None = None) -> dict:
    """Plan dict of one stage: contexts (by id), shards, projected seconds, the stage, its arms and adapters and the
    prompt manifest hash. ``arm_factor``: per-arm slowdown (LoRA overhead) measured in TV; 1.0 for base-model arms."""
    check_stages(config)
    stage_arms = list(config["stages"][stage_layout.check(stage)])
    arm_factor = dict(arm_factor or {})
    cells = config["cells"]
    max_seconds = float(config["shards"]["max_seconds"])
    contexts: dict[str, Context] = {}
    shards: list[Shard] = []
    sample_sets = set(config["sampling"]["sets"])
    for arm in stage_arms:
        adapter, context = config["arms"][arm]
        factor = float(arm_factor.get(arm, 1.0))
        arm_contexts = contexts_for_arm(arm, context, entries, cells)
        for c in arm_contexts:
            contexts[c.context_id] = c
        capture = set(cells["capture"])
        unit = [(c, seconds["score"] + (seconds["capture"] if c.cell in capture else 0.0)) for c in arm_contexts]
        shards += _split(arm, "score", adapter, unit, factor, max_seconds, seconds["load"])
        if arm in config["sampling"]["arms"]:
            stems = {e["stem_id"] for e in entries if e["set"] in sample_sets}
            k = int(config["sampling"]["k"])
            unit = [(c, k * seconds["sample"]) for c in arm_contexts if c.cell in set(cells["sample"]) and c.stem_id in stems]
            shards += _split(arm, "sample", adapter, unit, factor, max_seconds, seconds["load"])
    lo, hi = config["number_capture"]["rows"]
    for arm in (a for a in config["number_capture"]["arms"] if a in stage_arms):
        adapter = config["arms"][arm][0]
        unit = [(Context(f"{arm}|num{row:05d}|Q+none", f"num{row:05d}", "Q+none", "none", "P_default", ""),
                 seconds["numcap"]) for row in range(lo, hi)]
        for c, _ in unit:
            contexts[c.context_id] = c
        shards += _split(arm, "numcap", adapter, unit, float(arm_factor.get(arm, 1.0)), max_seconds, seconds["load"])
    total = sum(s.projected_seconds for s in shards)
    return {
        "stage": stage,
        "arms": {arm: config["arms"][arm][0] for arm in stage_arms},
        "prompt_manifest_sha256": config["contract"]["prompt_manifest_sha256"],
        "adapter_lock": stage_layout.LOCK[stage],
        "contexts": {cid: asdict(c) for cid, c in contexts.items()},
        "shards": [asdict(s) for s in shards],
        "projected_gpu_seconds": total,
        "seconds_per_unit": dict(seconds),
        "arm_factor": arm_factor,
    }


def _split(arm, kind, adapter, unit, factor, max_seconds, load_seconds) -> list[Shard]:
    if not unit:
        return []
    shards, current, used = [], [], load_seconds
    for c, sec in unit:
        cost = sec * factor
        if current and used + cost > max_seconds:
            shards.append(Shard(f"{arm}.{kind}.{len(shards):03d}", arm, kind, adapter, tuple(current), used))
            current, used = [], load_seconds
        current.append(c.context_id)
        used += cost
    shards.append(Shard(f"{arm}.{kind}.{len(shards):03d}", arm, kind, adapter, tuple(current), used))
    if any(s.projected_seconds > max_seconds for s in shards):
        raise PlanError(f"A single unit of {arm}/{kind} exceeds the shard time limit")
    return shards


def projection_a100_h(plan: Mapping[str, Any], overhead: float) -> float:
    return plan["projected_gpu_seconds"] * overhead / 3600.0


def cap_a100_h(projection: float, throughput_cv: float) -> float:
    """Resource cap (decision D5): projection x (1 + max(0.15, CV)) x 1.20 retry allowance, rounded up to 1.5 x
    projection when that is larger. A planning rule, not a scientific threshold."""
    return max(projection * (1.0 + max(0.15, throughput_cv)) * 1.20, 1.5 * projection)

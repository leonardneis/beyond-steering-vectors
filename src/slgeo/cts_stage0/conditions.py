"""Scientific conditions, read from the v2 condition registry (the registry is authoritative).

The registry ``research/cts_stage0_v2/cts_stage0_v2_condition_registry.jsonl`` is generated from the decision
spec and pinned by hash; this module never adds, drops or edits a condition. A condition is symbolic: its
magnitude is resolved from the hashed direction bundle inside the scoring node.
"""

from __future__ import annotations

from .errors import FinalFailure

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .directions import SITE_SLOT, DirectionBundle, DirectionError
from .steering import ALL, LAST, RowSteer

UNSTEERED = "unsteered"
STEER = "steer"
PERSONA = "persona"
TESTED_CONTRASTS = ("c_cat_dog", "c_cat_wolf", "c_cat_anim")
PARAPHRASES = {"c_cat_dog": ("T2", "T3"), "c_cat_wolf": ("T2", "T3"), "c_cat_anim": ("T3",)}
SHARED_LABEL_DIRECTIONS = ("g_anim", "g_tmpl", "g_id")
N_RCOV = 199
N_RCOV_PC = 99
L2 = "L2_shared_prefix"
OWN = "own_prefix"
COST_CLASSES = (L2, OWN)
PROMPT_SETS = ("S0_all", "S0_animal")
L2_BASELINE = "unsteered"
OWN_BASELINE = "persona:P_default"


class ConditionError(RuntimeError, FinalFailure):
    """A condition is malformed or its vector cannot be resolved from the bundle."""


@dataclass(frozen=True)
class Condition:
    """One scoring condition, exactly as listed in the registry.

    ``direction``: bundle name, or ``rcov:<i>`` (None for unsteered/persona conditions).
    ``scale``: ``unit`` (vector = sign * kappa * magnitude * unit(D)) or ``raw`` (vector = raw D).
    ``magnitude``: ``tau:<name>`` (|tau(name)|) or ``norm:<name>`` (||raw name||); unused for ``raw``.
    """

    kind: str
    group: str
    gating: bool
    direction: str | None = None
    scale: str = "unit"
    magnitude: str | None = None
    kappa: float = 1.0
    sign: int = 1
    slot: int = SITE_SLOT
    mode: str = LAST
    persona: str = "P_default"
    prompt_set: str = "S0_all"
    cost_class: str = L2
    purpose: str = ""
    reference_rescore: bool = False

    def __post_init__(self) -> None:
        if self.kind not in (UNSTEERED, STEER, PERSONA):
            raise ConditionError(f"Unknown condition kind {self.kind!r}")
        if self.cost_class not in COST_CLASSES or self.prompt_set not in PROMPT_SETS:
            raise ConditionError(f"Unknown cost class or prompt set in {self.cid}")
        own = self.kind == PERSONA or self.mode == ALL
        if own != (self.cost_class == OWN):
            raise ConditionError(f"{self.cid}: cost class does not follow from kind and mode")
        if self.reference_rescore and self.cost_class != L2:
            raise ConditionError(f"{self.cid}: only L2 conditions are re-scored in the reference layout")

    @property
    def block(self) -> int:
        return self.slot - 1

    @property
    def cid(self) -> str:
        if self.kind == UNSTEERED:
            return "unsteered"
        if self.kind == PERSONA:
            return f"persona:{self.persona}"
        mag = self.magnitude if self.scale == "unit" else "self"
        return f"{self.direction}|{self.scale}:{mag}|k={self.kappa:g}|s={self.sign:+d}|slot={self.slot}|{self.mode}"

    @property
    def baseline(self) -> str:
        """The baseline of the same cost class (spec ``execution_layout.baseline_rule``)."""
        return OWN_BASELINE if self.cost_class == OWN else L2_BASELINE

    def as_dict(self) -> dict:
        out = asdict(self)
        out["cid"] = self.cid
        return out


def condition_from_row(row: Mapping[str, Any]) -> Condition:
    fields = {key: value for key, value in row.items() if key != "cid"}
    condition = Condition(**fields)
    if condition.cid != row["cid"]:
        raise ConditionError(f"Registry cid {row['cid']!r} does not match its fields ({condition.cid!r})")
    return condition


def registry_conditions(rows: Iterable[Mapping[str, Any]]) -> list[Condition]:
    conditions = [condition_from_row(row) for row in rows]
    ids = [condition.cid for condition in conditions]
    if len(ids) != len(set(ids)):
        raise ConditionError("Duplicate condition ids")
    return conditions


def cid(direction, *, magnitude=None, scale="unit", kappa=1.0, sign=1, slot=SITE_SLOT, mode=LAST) -> str:
    """Condition id of a steered condition (same format as the registry generator)."""
    if scale == "unit" and magnitude is None:
        magnitude = f"tau:{direction}"
    mag = magnitude if scale == "unit" else "self"
    return f"{direction}|{scale}:{mag}|k={kappa:g}|s={sign:+d}|slot={slot}|{mode}"


def _random(bundle: DirectionBundle, name: str) -> np.ndarray:
    family, index = name.split(":")
    if family != "rcov":
        raise ConditionError(f"Unknown random family {family!r}")
    return bundle.r_cov[int(index)]


def magnitude_of(bundle: DirectionBundle, reference: str) -> float:
    kind, name = reference.split(":", 1)
    direction = bundle.get(name)
    if kind == "tau":
        return abs(direction.tau)
    if kind == "norm":
        return direction.norm
    raise ConditionError(f"Unknown magnitude reference {reference!r}")


def resolve_vector(condition: Condition, bundle: DirectionBundle) -> np.ndarray:
    """float64 steering vector of a steered condition (cast to float32 at the hook boundary)."""
    if condition.kind != STEER:
        raise ConditionError(f"{condition.cid} has no steering vector")
    if condition.sign not in (1, -1) or condition.kappa <= 0:
        raise ConditionError(f"{condition.cid}: invalid sign or kappa")
    if condition.scale == "raw":
        if condition.sign != 1 or condition.kappa != 1.0:
            raise ConditionError("Raw-axis conditions are added with sign +1 and kappa 1 only")
        direction = bundle.get(condition.direction)
        if direction.slot != condition.slot:
            raise ConditionError(f"{condition.cid}: direction slot {direction.slot} != condition slot")
        return direction.raw.copy()
    if condition.direction.startswith("rcov:"):
        unit = _random(bundle, condition.direction)
        if condition.slot != SITE_SLOT:
            raise ConditionError("Random directions are defined at slot 14 only")
    else:
        direction = bundle.get(condition.direction)
        if direction.slot != condition.slot:
            raise ConditionError(f"{condition.cid}: direction slot {direction.slot} != condition slot")
        unit = direction.unit
    magnitude = magnitude_of(bundle, condition.magnitude)
    if not np.isfinite(magnitude) or magnitude <= 0:
        raise DirectionError(f"{condition.cid}: non-positive magnitude")
    return condition.sign * condition.kappa * magnitude * unit


def row_steer(condition: Condition, bundle: DirectionBundle) -> RowSteer:
    if condition.kind != STEER:
        return RowSteer(vectors={})
    vector = torch.from_numpy(resolve_vector(condition, bundle)).to(torch.float32)
    return RowSteer(vectors={condition.block: vector}, mode=condition.mode)


def vector_sha256(condition: Condition, bundle: DirectionBundle) -> str | None:
    if condition.kind != STEER:
        return None
    import hashlib

    vector = resolve_vector(condition, bundle).astype(np.float32)
    return hashlib.sha256(vector.tobytes()).hexdigest()

"""The complete, deterministic condition registry of the scientific run.

Gating conditions follow the frozen spec exactly (pre-implementation audit 1, §4). Descriptive
conditions are fixed by ``IMPLEMENTATION_CHOICES.json`` before any scientific output exists. A condition
is symbolic: its magnitude is resolved from the hashed direction bundle inside the scoring node.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from .directions import SITE_SLOT, DirectionBundle, DirectionError
from .steering import ALL, LAST, RowSteer

UNSTEERED = "unsteered"
STEER = "steer"
PERSONA = "persona"
TESTED_CONTRASTS = ("c_cat_dog", "c_cat_wolf", "c_cat_anim")
PARAPHRASES = {"c_cat_dog": ("T2", "T3"), "c_cat_wolf": ("T2", "T3"), "c_cat_anim": ("T3",)}
N_RCOV = 1000
N_RCOV_PC = 200
N_RISO = 1000


class ConditionError(RuntimeError):
    """A condition is malformed or its vector cannot be resolved from the bundle."""


@dataclass(frozen=True)
class Condition:
    """One scoring condition.

    ``direction``: bundle name, ``rcov:<i>`` or ``riso:<i>`` (None for unsteered/persona conditions).
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

    def as_dict(self) -> dict:
        out = asdict(self)
        out["cid"] = self.cid
        return out


def _steer(direction, *, gating, group, magnitude=None, scale="unit", kappa=1.0, sign=1, slot=SITE_SLOT, mode=LAST):
    if scale == "unit" and magnitude is None:
        magnitude = f"tau:{direction}"
    return Condition(STEER, group, gating, direction, scale, magnitude, kappa, sign, slot, mode)


def gating_conditions(null_names: list[str]) -> list[Condition]:
    out = [Condition(UNSTEERED, "baseline", True)]
    # Persona contexts are scored in the batch-1 layout; A6 pairs P_X against the batch-1 default context.
    out.append(Condition(PERSONA, "persona", True, persona="P_default"))
    for x in ("dog", "wolf"):
        out.append(Condition(PERSONA, "persona", True, persona=f"P_{x}_T1"))
    for contrast in TESTED_CONTRASTS:
        for kappa in (1.0, 0.5):
            for sign in (1, -1):
                out.append(_steer(contrast, gating=True, group="named", kappa=kappa, sign=sign))
        for template in PARAPHRASES[contrast]:
            for sign in (1, -1):
                out.append(_steer(f"{contrast}_{template}", gating=True, group="named", sign=sign))
    out.append(_steer("t_cat", gating=True, group="named", scale="raw"))
    out.append(_steer("t_cat", gating=True, group="named", scale="raw", mode=ALL))
    for contrast in TESTED_CONTRASTS:
        out.append(_steer("t_cat", gating=True, group="named", magnitude=f"tau:{contrast}"))
    for x in ("dog", "wolf"):
        out.append(_steer(f"t_{x}", gating=True, group="named", scale="raw"))
    out.append(_steer("m_cat_dog", gating=True, group="named", magnitude="tau:c_cat_dog"))
    out.append(_steer("m_cat_wolf", gating=True, group="named", magnitude="tau:c_cat_wolf"))
    out.append(_steer("m_cat_dog", gating=True, group="named", magnitude="tau:c_cat_anim"))
    out.append(_steer("m_cat_wolf", gating=True, group="named", magnitude="tau:c_cat_anim"))
    for contrast in TESTED_CONTRASTS:
        for name in null_names:
            out.append(_steer(name, gating=True, group=f"null:{contrast}", magnitude=f"tau:{contrast}"))
        for index in range(N_RCOV):
            out.append(_steer(f"rcov:{index}", gating=True, group=f"rcov:{contrast}", magnitude=f"tau:{contrast}"))
    for mode in (LAST, ALL):
        for index in range(N_RCOV_PC):
            out.append(_steer(f"rcov:{index}", gating=True, group=f"rcov_pc:{mode}", magnitude="norm:t_cat", mode=mode))
    return out


def descriptive_conditions(personas: list[str], choices: dict) -> list[Condition]:
    out: list[Condition] = []
    for contrast in TESTED_CONTRASTS:
        for kappa in choices["kappa_descriptive"]:
            for sign in (1, -1):
                out.append(_steer(contrast, gating=False, group="named", kappa=kappa, sign=sign))
    for name in choices["descriptive_directions"]:
        for sign in (1, -1):
            out.append(_steer(name, gating=False, group="named", sign=sign))
    for index in range(N_RISO):
        out.append(_steer(f"riso:{index}", gating=False, group="riso", magnitude=choices["riso_magnitude"]))
    for slot in choices["secondary_slots"]:
        for contrast in TESTED_CONTRASTS:
            for sign in (1, -1):
                out.append(_steer(f"{contrast}@{slot}", gating=False, group="secondary", sign=sign, slot=slot))
        out.append(_steer(f"t_cat@{slot}", gating=False, group="secondary", scale="raw", slot=slot))
    for contrast in TESTED_CONTRASTS:
        for sign in (1, -1):
            out.append(_steer(contrast, gating=False, group="named", sign=sign, mode=ALL))
    for name in choices["panel33_contrasts"]:
        for sign in (1, -1):
            out.append(_steer(name, gating=False, group="named", sign=sign))
    gating_personas = {"P_default", "P_dog_T1", "P_wolf_T1"}
    for persona in personas:
        if persona not in gating_personas:
            out.append(Condition(PERSONA, "persona", False, persona=persona))
    return out


def all_conditions(null_names: list[str], personas: list[str], choices: dict) -> list[Condition]:
    conditions = gating_conditions(null_names) + descriptive_conditions(personas, choices)
    ids = [condition.cid for condition in conditions]
    if len(ids) != len(set(ids)):
        duplicates = sorted({cid for cid in ids if ids.count(cid) > 1})
        raise ConditionError(f"Duplicate condition ids: {duplicates[:5]}")
    return conditions


def _random(bundle: DirectionBundle, name: str) -> np.ndarray:
    family, index = name.split(":")
    matrix = {"rcov": bundle.r_cov, "riso": bundle.r_iso}[family]
    return matrix[int(index)]


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
    if condition.direction.startswith(("rcov:", "riso:")):
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

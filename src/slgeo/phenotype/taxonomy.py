"""Outcome taxonomy of the Phenotype Anchor (preregistration draft ``research/phenotype_anchor_v1/PREREGISTRATION.md`` §7 and §9).

The driver maps per-seed test results to exactly one summary class plus modifiers, by the first matching rank.
It never changes a statistic; it only reads booleans and p-values that ``stats`` produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .stats import ALPHA, seed_passes

P1_HYPOTHESES = ("C1", "C2", "C3", "C4", "C5")
P1_GATED = ("C1", "C2", "C3")
P2_HYPOTHESES = ("K1", "K2", "K3", "K4")
P2_GATED = ("K1", "K2", "K3")


@dataclass(frozen=True)
class SeedResult:
    """One confirmatory seed. ``p``: one-sided p-values; ``gates``: run-level gates of the gated hypotheses;
    ``c3_equivalent``: 90 % CI of the C3 residual inside the pre-declared margin; ``label``: cat-dominance label;
    ``adequate``: tempering-model adequacy."""

    p: Mapping[str, float]
    gates: Mapping[str, bool]
    c3_equivalent: bool
    label: bool
    adequate: bool


@dataclass(frozen=True)
class Outcome:
    cls: str
    modifiers: tuple[str, ...]
    confirmed: dict[str, bool]
    per_seed: dict[str, dict[str, bool]]
    notes: tuple[str, ...] = field(default=())


def _passes(results: Mapping[str, SeedResult], hyps, gated, alpha):
    return {s: seed_passes({h: r.p[h] for h in hyps}, r.gates, gated=gated, alpha=alpha) for s, r in results.items()}


def classify_p1(
    results: Mapping[str, SeedResult], *, integrity_ok: bool, instrument_ok: bool,
    secondary_class: str | None = None, alpha: float = ALPHA,
) -> Outcome:
    """P1 summary class (first match) and modifiers. ``secondary_class``: the class computed the same way in the
    Q-no-prefix cell (for the RENDERING_DEPENDENT modifier)."""
    if not integrity_ok:
        return Outcome("TECHNICAL_FAIL", (), {}, {})
    if not instrument_ok:
        return Outcome("INSTRUMENT_FAIL", (), {}, {})
    per_seed = _passes(results, P1_HYPOTHESES, P1_GATED, alpha)
    both = {h: all(per_seed[s][h] for s in per_seed) for h in P1_HYPOTHESES}
    one = {h: sum(per_seed[s][h] for s in per_seed) == 1 for h in P1_HYPOTHESES}
    adequate = all(r.adequate for r in results.values())
    equiv = all(r.c3_equivalent for r in results.values())
    label = all(r.label for r in results.values())
    notes = []

    if not adequate:
        cls = "FLATTENING_MODEL_INADEQUATE"
        notes.append("C3-C5 not interpretable; only C1/C2 directions are reported")
        confirmed = {h: both[h] for h in ("C1", "C2")}
        return Outcome(cls, (), confirmed, per_seed, tuple(notes))

    if both["C3"] and label:
        cls = "CAT_DOMINANT"
    elif both["C3"]:
        cls = "CAT_RESIDUAL_NOT_DOMINANT"
        if equiv:
            notes.append("positive but inside the equivalence margin")
    elif both["C2"] and equiv:
        cls = "FLATTENING_NO_CAT"
    elif both["C2"]:
        cls = "FLATTENING_CAT_UNRESOLVED"
    elif both["C1"]:
        cls = "REDISTRIBUTION_UNSTRUCTURED"
    elif any(one[h] for h in P1_GATED):
        cls = "SEED_HETEROGENEOUS"
    else:
        cls = "NULL"

    modifiers = []
    if both["C4"]:
        modifiers.append("TEACHER_SHADOW")
    if both["C5"]:
        modifiers.append("REPLICATES_DEV_PROFILE")
    if secondary_class is not None and secondary_class != cls:
        modifiers.append("RENDERING_DEPENDENT")
    return Outcome(cls, tuple(modifiers), both, per_seed, tuple(notes))


def classify_p2(
    results: Mapping[str, Mapping[str, float]], gates: Mapping[str, Mapping[str, bool]], *,
    dog_students_null: bool, integrity_ok: bool, alpha: float = ALPHA,
) -> Outcome:
    """P2 summary class. ``results[seed]``: one-sided p-values of K1-K4; ``gates[seed]``: run-level gates of K1-K3;
    ``dog_students_null``: D vs N fails C1, C2 and C3 in both confirmatory seeds."""
    if not integrity_ok:
        return Outcome("TECHNICAL_FAIL", (), {}, {})
    per_seed = {s: seed_passes(dict(p), gates.get(s, {}), gated=P2_GATED, alpha=alpha) for s, p in results.items()}
    both = {h: all(per_seed[s][h] for s in per_seed) for h in P2_HYPOTHESES}
    if both["K1"] and both["K2"]:
        cls = "DOUBLE_DISSOCIATION"
    elif both["K1"]:
        cls = "CAT_ONLY_SPECIFIC"
    elif both["K2"]:
        cls = "DOG_ONLY_SPECIFIC"
    elif both["K3"] and both["K4"]:
        cls = "GENERIC_PERSONA_TEACHER"
    elif dog_students_null:
        cls = "NO_DOG_TRANSFER"
    else:
        cls = "P2_NULL_OR_MIXED"
    return Outcome(cls, (), both, per_seed)

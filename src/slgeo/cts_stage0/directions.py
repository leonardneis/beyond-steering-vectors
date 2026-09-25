"""Direction construction from extraction statistics (PREREGISTRATION §4.1, §6.2; spec ``directions``).

Every persona direction is a linear combination of raw axes t_P = mu(P) - mu(P_default); half versions use
same-half means of every persona in the formula (spec ``criteria.R.half_axes``). All arithmetic is float64.
The teacher coordinate tau(D) = <t_ref, unit(D)> uses t_cat (P_cat_T1) at the same slot, except for the
paraphrase contrasts, which use t_cat of their own template (spec ``paraphrase_dose``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .statistics import random_cov_directions, random_iso_directions

SITE_SLOT = 14
SECONDARY_SLOTS = (8, 21, 27)
A_LEN = ("dog", "wolf", "lion", "horse", "rabbit", "elephant")
CANDIDATES = ("dog", "wolf")
MIN_RAW_NORM = 1e-6


class DirectionError(RuntimeError):
    """A direction is undefined (zero norm, missing persona) or inconsistent with the frozen formula."""


@dataclass(frozen=True)
class Formula:
    """D = sum_P coef[P] * t_P at ``slot``; ``tau_ref`` is the persona whose axis defines tau(D)."""

    name: str
    coefficients: Mapping[str, float]
    slot: int = SITE_SLOT
    tau_ref: str = "P_cat_T1"
    gating_reliability: bool = False
    note: str = ""


def _contrast(name, a, others, **kwargs) -> Formula:
    coefficients: dict[str, float] = {a: 1.0}
    for other in others:
        coefficients[other] = coefficients.get(other, 0.0) - 1.0 / len(others)
    return Formula(name, coefficients, **kwargs)


def persona_formulas() -> list[Formula]:
    """All named persona-axis directions (gating and descriptive), in a fixed order."""
    formulas: list[Formula] = []
    for x in CANDIDATES:
        formulas.append(_contrast(f"c_cat_{x}", "P_cat_T1", [f"P_{x}_T1"], gating_reliability=True))
    formulas.append(_contrast("c_cat_anim", "P_cat_T1", [f"P_{x}_T1" for x in A_LEN], gating_reliability=True))
    for x in CANDIDATES:
        for template in ("T2", "T3"):
            formulas.append(
                _contrast(f"c_cat_{x}_{template}", f"P_cat_{template}", [f"P_{x}_{template}"], tau_ref=f"P_cat_{template}")
            )
    formulas.append(_contrast("c_cat_anim_T3", "P_cat_T3", [f"P_{x}_T3" for x in A_LEN], tau_ref="P_cat_T3"))
    for x in CANDIDATES:
        formulas.append(_contrast(f"m_cat_{x}", "M_cat", [f"M_{x}"]))
    formulas.append(Formula("t_cat", {"P_cat_T1": 1.0}, gating_reliability=True))
    for x in CANDIDATES:
        formulas.append(Formula(f"t_{x}", {f"P_{x}_T1": 1.0}))
    # Descriptive shared-component descriptors and 33-token panel contrasts.
    formulas.append(Formula("g_id", {"P_id": 1.0}, note="descriptive"))
    formulas.append(Formula("h", {"P_cat_T1": 1.0, "P_qwencat": -1.0}, note="descriptive"))
    formulas.append(Formula("g_tmpl", {"P_chess": 0.5, "P_blue": 0.5}, note="descriptive"))
    anim = {f"P_{x}_T1": 1.0 / 7.0 for x in ("cat",) + A_LEN}
    anim["P_chess"] = -0.5
    anim["P_blue"] = -0.5
    formulas.append(Formula("g_anim", anim, note="descriptive"))
    for x in ("fox", "owl"):
        formulas.append(_contrast(f"c_cat_{x}", "P_cat_T1", [f"P_{x}_T1"], note="descriptive; 33-token persona"))
    for slot in SECONDARY_SLOTS:
        for x in CANDIDATES:
            formulas.append(_contrast(f"c_cat_{x}@{slot}", "P_cat_T1", [f"P_{x}_T1"], slot=slot, note="secondary site"))
        formulas.append(
            _contrast(f"c_cat_anim@{slot}", "P_cat_T1", [f"P_{x}_T1" for x in A_LEN], slot=slot, note="secondary site")
        )
        formulas.append(Formula(f"t_cat@{slot}", {"P_cat_T1": 1.0}, slot=slot, note="secondary site"))
    return formulas


def null_formulas(null_words: list[str]) -> list[Formula]:
    """The 240 ordered structured-null contrasts c_{a,b} = unit(t_a - t_b), T1 null-pool personas."""
    return [
        _contrast(f"null:{a}>{b}", f"N_{a}_T1", [f"N_{b}_T1"], note="structured null")
        for a in null_words
        for b in null_words
        if a != b
    ]


@dataclass
class AxisStatistics:
    """Per-persona float64 means: ``full[P]`` [29, H]; ``half[P]`` [2, 29, H]."""

    full: dict[str, np.ndarray]
    half: dict[str, np.ndarray]

    @classmethod
    def from_half_sums(cls, half_sums: Mapping[str, np.ndarray], half_size: int = 512) -> "AxisStatistics":
        full, half = {}, {}
        for persona, sums in half_sums.items():
            sums = np.asarray(sums, dtype=np.float64)
            if sums.shape[0] != 2 or sums.ndim != 3:
                raise DirectionError(f"Half sums of {persona} must be [2, slots, H]")
            half[persona] = sums / half_size
            full[persona] = (sums[0] + sums[1]) / (2 * half_size)
        return cls(full, half)

    def axis(self, persona: str, slot: int) -> np.ndarray:
        return self.full[persona][slot] - self.full["P_default"][slot]

    def half_axis(self, persona: str, slot: int, h: int) -> np.ndarray:
        return self.half[persona][h, slot] - self.half["P_default"][h, slot]


def _unit(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= MIN_RAW_NORM:
        raise DirectionError(f"Direction {label} has zero or non-finite norm")
    return vector / norm


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))


@dataclass
class Direction:
    name: str
    slot: int
    raw: np.ndarray  # pre-normalization combination of axes (float64)
    unit: np.ndarray
    tau: float
    reliability: float | None  # split-half cosine (None where undefined)
    gating_reliability: bool
    tau_ref: str
    coefficients: dict[str, float] = field(default_factory=dict)

    @property
    def norm(self) -> float:
        return float(np.linalg.norm(self.raw))


def build_direction(stats: AxisStatistics, formula: Formula) -> Direction:
    missing = [persona for persona in [*formula.coefficients, formula.tau_ref] if persona not in stats.full]
    if missing:
        raise DirectionError(f"{formula.name}: missing personas {missing}")
    raw = sum(coef * stats.axis(persona, formula.slot) for persona, coef in formula.coefficients.items())
    halves = [
        sum(coef * stats.half_axis(persona, formula.slot, h) for persona, coef in formula.coefficients.items())
        for h in (0, 1)
    ]
    unit = _unit(raw, formula.name)
    tau = float(stats.axis(formula.tau_ref, formula.slot) @ unit)
    reliability = cosine(halves[0], halves[1])
    return Direction(
        formula.name, formula.slot, raw, unit, tau, reliability, formula.gating_reliability, formula.tau_ref,
        dict(formula.coefficients),
    )


def embedding_direction(name: str, row_a: np.ndarray, row_b: np.ndarray, t_cat14: np.ndarray) -> Direction:
    """e_{A,B} = unit(W_E[plural_A] - W_E[plural_B]) (descriptive lexical control)."""
    raw = np.asarray(row_a, dtype=np.float64) - np.asarray(row_b, dtype=np.float64)
    unit = _unit(raw, name)
    return Direction(name, SITE_SLOT, raw, unit, float(t_cat14 @ unit), None, False, "P_cat_T1")


@dataclass
class DirectionBundle:
    directions: dict[str, Direction]
    r_cov: np.ndarray  # [1000, H]
    r_iso: np.ndarray  # [1000, H]
    null_names: list[str]

    def get(self, name: str) -> Direction:
        try:
            return self.directions[name]
        except KeyError as exc:
            raise DirectionError(f"Unknown direction {name!r}") from exc


def build_bundle(
    stats: AxisStatistics,
    default_states14: np.ndarray,
    null_words: list[str],
    embedding_rows: Mapping[str, np.ndarray],
) -> DirectionBundle:
    directions: dict[str, Direction] = {}
    for formula in persona_formulas() + null_formulas(null_words):
        if formula.name in directions:
            raise DirectionError(f"Duplicate direction {formula.name}")
        directions[formula.name] = build_direction(stats, formula)
    t_cat14 = stats.axis("P_cat_T1", SITE_SLOT)
    for x in CANDIDATES:
        name = f"e_cat_{x}"
        directions[name] = embedding_direction(name, embedding_rows["cat"], embedding_rows[x], t_cat14)
    hidden = t_cat14.shape[0]
    return DirectionBundle(
        directions=directions,
        r_cov=random_cov_directions(default_states14),
        r_iso=random_iso_directions(hidden),
        null_names=[formula.name for formula in null_formulas(null_words)],
    )

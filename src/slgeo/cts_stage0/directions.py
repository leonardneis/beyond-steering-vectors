"""Direction construction from extraction statistics (v2 spec ``directions``, ``intervention.dose``).

Every persona direction is a linear combination of raw axes t_P = mu(P) - mu(P_default); half versions use
same-half means of every persona in the formula (spec ``criteria.R.half_axes``). All arithmetic is float64.
The teacher coordinate tau(D) = <t_ref, unit(D)> uses t_cat (P_cat_T1) at the same slot, except for the
paraphrase contrasts, which use t_cat of their own template (spec ``paraphrase_dose``).

v2 additions: the held-out animal-generic component g_anim (16 null-pool axes minus the non-animal template
axes), the span G of the shared components, and the leakage-free contrasts c_perpG = unit(c - P_G c).
"""

from __future__ import annotations

from .errors import FinalFailure

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .statistics import random_cov_directions

SITE_SLOT = 14
SECONDARY_SLOT = 27
A_LEN = ("dog", "wolf", "lion", "horse", "rabbit", "elephant")
CANDIDATES = ("dog", "wolf")
TESTED = ("c_cat_dog", "c_cat_wolf", "c_cat_anim")
SHARED_COMPONENTS = ("g_anim", "g_tmpl", "g_id", "h")  # columns of G; the projector does not depend on order
G_RANK_TOLERANCE = 1e-6  # singular values <= 1e-6 x the largest are dropped (spec shared_components.span_G)
MIN_RAW_NORM = 1e-6


class DirectionError(RuntimeError, FinalFailure):
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


def _mean_minus(name: str, positive: Sequence[str], negative: Sequence[str], **kwargs) -> Formula:
    coefficients: dict[str, float] = {}
    for persona in positive:
        coefficients[persona] = coefficients.get(persona, 0.0) + 1.0 / len(positive)
    for persona in negative:
        coefficients[persona] = coefficients.get(persona, 0.0) - 1.0 / len(negative)
    return Formula(name, coefficients, **kwargs)


def persona_formulas(null_words: Sequence[str]) -> list[Formula]:
    """All named persona-axis directions of v2 (gating and descriptive), in a fixed order."""
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
    # Shared components (G). g_anim is held out: built from the 16 null-pool axes only (v2 spec, change #20).
    formulas.append(Formula("g_id", {"P_id": 1.0}, note="shared component"))
    formulas.append(Formula("h", {"P_cat_T1": 1.0, "P_qwencat": -1.0}, note="shared component"))
    formulas.append(Formula("g_tmpl", {"P_chess": 0.5, "P_blue": 0.5}, note="shared component"))
    formulas.append(
        _mean_minus("g_anim", [f"N_{word}_T1" for word in null_words], ["P_chess", "P_blue"], note="shared component (held out)")
    )
    formulas.append(
        _mean_minus("g_anim_v1", [f"P_{x}_T1" for x in ("cat",) + A_LEN], ["P_chess", "P_blue"], note="v1 definition; descriptive only")
    )
    for contrast in TESTED:
        others = [f"P_{x}_T1" for x in (A_LEN if contrast == "c_cat_anim" else (contrast.removeprefix("c_cat_"),))]
        formulas.append(_contrast(f"{contrast}@{SECONDARY_SLOT}", "P_cat_T1", others, slot=SECONDARY_SLOT, note="descriptive secondary site"))
    return formulas


def null_formulas(null_words: Sequence[str]) -> list[Formula]:
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
    raw: np.ndarray  # pre-normalization vector (float64)
    unit: np.ndarray
    tau: float
    reliability: float | None  # split-half cosine (None where undefined)
    gating_reliability: bool
    tau_ref: str
    coefficients: dict[str, float] = field(default_factory=dict)
    stored_norm: float | None = None

    @property
    def norm(self) -> float:
        """||raw||, computed once when the bundle is built and then read from the bundle (so every node uses
        bit-identical magnitudes, independent of BLAS reduction order)."""
        return float(self.stored_norm) if self.stored_norm is not None else float(np.linalg.norm(self.raw))


def _combination(stats: AxisStatistics, formula: Formula, h: int | None) -> np.ndarray:
    if h is None:
        return sum(coef * stats.axis(persona, formula.slot) for persona, coef in formula.coefficients.items())
    return sum(coef * stats.half_axis(persona, formula.slot, h) for persona, coef in formula.coefficients.items())


def build_direction(stats: AxisStatistics, formula: Formula) -> Direction:
    missing = [persona for persona in [*formula.coefficients, formula.tau_ref] if persona not in stats.full]
    if missing:
        raise DirectionError(f"{formula.name}: missing personas {missing}")
    raw = _combination(stats, formula, None)
    halves = [_combination(stats, formula, h) for h in (0, 1)]
    unit = _unit(raw, formula.name)
    tau = float(stats.axis(formula.tau_ref, formula.slot) @ unit)
    reliability = cosine(halves[0], halves[1])
    return Direction(
        formula.name, formula.slot, raw, unit, tau, reliability, formula.gating_reliability, formula.tau_ref,
        dict(formula.coefficients), float(np.linalg.norm(raw)),
    )


def span_basis(columns: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal basis of span(columns) from the thin SVD; singular values <= tol x max are dropped.

    Returns (basis [H, r], singular values of all columns). The projector basis @ basis.T does not depend on the
    column order or scaling of the inputs."""
    matrix = np.stack([_unit(np.asarray(c, dtype=np.float64), "G column") for c in columns], axis=1)
    u, s, _vt = np.linalg.svd(matrix, full_matrices=False)
    keep = s > G_RANK_TOLERANCE * s.max()
    return u[:, keep], s


def project_out(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return vector - basis @ (basis.T @ vector)


def leakage_free_direction(stats: AxisStatistics, contrast: Formula, shared: Mapping[str, Formula]) -> tuple[Direction, dict]:
    """c_perpG = unit(c - P_G c) at slot 14 (spec ``directions.leakage_free_contrasts``).

    Halves: c and G are rebuilt from the same half means; R = cos(c_perpG half 1, c_perpG half 2).
    The report holds the geometry only (never a behavioural quantity)."""
    columns = [shared[name] for name in SHARED_COMPONENTS]
    c_unit = _unit(_combination(stats, contrast, None), contrast.name)
    basis, singular = span_basis([_combination(stats, formula, None) for formula in columns])
    projection = basis @ (basis.T @ c_unit)
    residual = c_unit - projection
    unit = _unit(residual, f"{contrast.name}_perpG")
    half_residuals = []
    for h in (0, 1):
        half_c = _unit(_combination(stats, contrast, h), f"{contrast.name} half {h}")
        half_basis, _ = span_basis([_combination(stats, formula, h) for formula in columns])
        half_residuals.append(project_out(half_c, half_basis))
    reliability = cosine(half_residuals[0], half_residuals[1])
    tau = float(stats.axis("P_cat_T1", SITE_SLOT) @ unit)
    g_matrix = np.stack([_unit(_combination(stats, formula, None), formula.name) for formula in columns], axis=1)
    coefficients, *_ = np.linalg.lstsq(g_matrix, projection, rcond=None)
    report = {
        "projection_norm": float(np.linalg.norm(projection)),
        "cos_c_projection": float(np.linalg.norm(projection)),
        "retained_rank": int(basis.shape[1]),
        "singular_values": [float(v) for v in singular],
        "projection_share_per_svd_direction": [float(v) ** 2 for v in (basis.T @ c_unit)],
        "projection_coefficients_on_unit_components": dict(zip(SHARED_COMPONENTS, (float(v) for v in coefficients))),
    }
    direction = Direction(
        f"{contrast.name}_perpG", SITE_SLOT, residual, unit, tau, reliability, True, "P_cat_T1", {}, float(np.linalg.norm(residual)),
    )
    return direction, report


@dataclass
class DirectionBundle:
    directions: dict[str, Direction]
    r_cov: np.ndarray  # [199, H]
    null_names: list[str]
    perp_reports: dict[str, dict] = field(default_factory=dict)

    def get(self, name: str) -> Direction:
        try:
            return self.directions[name]
        except KeyError as exc:
            raise DirectionError(f"Unknown direction {name!r}") from exc


def build_bundle(stats: AxisStatistics, default_states14: np.ndarray, null_words: Sequence[str]) -> DirectionBundle:
    directions: dict[str, Direction] = {}
    formulas = persona_formulas(null_words)
    for formula in formulas + null_formulas(null_words):
        if formula.name in directions:
            raise DirectionError(f"Duplicate direction {formula.name}")
        directions[formula.name] = build_direction(stats, formula)
    by_name = {formula.name: formula for formula in formulas}
    reports = {}
    for contrast in TESTED:
        direction, report = leakage_free_direction(stats, by_name[contrast], by_name)
        directions[direction.name] = direction
        reports[direction.name] = report
    return DirectionBundle(
        directions=directions,
        r_cov=random_cov_directions(default_states14),
        null_names=[formula.name for formula in null_formulas(null_words)],
        perp_reports=reports,
    )

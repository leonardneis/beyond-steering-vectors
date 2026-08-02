"""Exact directional decomposition at a model's final linear readout."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np


def unit_direction(vector: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """Return a finite unit vector or fail closed for a degenerate direction."""
    value = np.asarray(vector, dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("Direction must be a finite rank-one vector")
    norm = float(np.linalg.norm(value))
    if norm <= eps:
        raise ValueError("Direction norm is zero or numerically degenerate")
    return value / norm


def decompose_delta(delta: np.ndarray, teacher: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project final-state displacements onto a fixed teacher axis and complement."""
    values = np.asarray(delta, dtype=np.float64)
    direction = unit_direction(teacher)
    if values.shape[-1] != direction.shape[0]:
        raise ValueError("State displacement and teacher dimensions differ")
    coefficient = np.einsum("...d,d->...", values, direction)
    parallel = coefficient[..., None] * direction
    residual = values - parallel
    return parallel, residual, coefficient


def margin_contributions(
    delta: np.ndarray,
    teacher: np.ndarray,
    readout_direction: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute exact full, teacher-axis, and complementary readout effects."""
    readout = np.asarray(readout_direction, dtype=np.float64)
    parallel, residual, _ = decompose_delta(delta, teacher)
    if readout.ndim != 1 or readout.shape[0] != delta.shape[-1]:
        raise ValueError("Readout direction must match the final-state dimension")
    full = np.einsum("...d,d->...", delta, readout)
    teacher_effect = np.einsum("...d,d->...", parallel, readout)
    residual_effect = np.einsum("...d,d->...", residual, readout)
    return {
        "full": full,
        "teacher": teacher_effect,
        "residual": residual_effect,
    }


def prompt_level_estimands(
    subliminal: dict[str, np.ndarray],
    neutral: dict[str, np.ndarray],
    set_names: Iterable[str],
) -> dict[str, np.ndarray]:
    """Form paired top-minus-mean-control values for every prompt."""
    names = np.asarray(list(set_names), dtype=str)
    counts = Counter(names.tolist())
    if counts != {"top_k": 1, "norm_matched_control": 25}:
        raise ValueError(f"Expected one top set and 25 norm controls, got {dict(counts)}")
    top = np.flatnonzero(names == "top_k")
    controls = np.flatnonzero(names == "norm_matched_control")
    result = {}
    for key in ("full", "teacher", "residual"):
        sub = np.asarray(subliminal[key], dtype=np.float64)
        neu = np.asarray(neutral[key], dtype=np.float64)
        if sub.shape != neu.shape or sub.ndim != 2 or sub.shape[0] != len(names):
            raise ValueError(f"Invalid condition arrays for {key}: {sub.shape}, {neu.shape}")
        paired = sub - neu
        result[key] = paired[top[0]] - paired[controls].mean(axis=0)
    return result


def stratified_bootstrap(
    prompt_values: dict[str, np.ndarray],
    families: Iterable[str],
    *,
    samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Bootstrap prompts within each fixed family and return replicate means."""
    family_values = np.asarray(list(families), dtype=str)
    if samples < 1 or family_values.ndim != 1:
        raise ValueError("Bootstrap samples and family labels are invalid")
    if set(prompt_values) != {"full", "teacher", "residual"}:
        raise ValueError("Bootstrap requires full, teacher, and residual values")
    length = len(family_values)
    if any(np.asarray(value).shape != (length,) for value in prompt_values.values()):
        raise ValueError("Every prompt value must align with the family labels")
    unique = sorted(set(family_values.tolist()))
    indices = [np.flatnonzero(family_values == family) for family in unique]
    if any(len(index) == 0 for index in indices):
        raise ValueError("Every bootstrap family must be non-empty")
    rng = np.random.default_rng(seed)
    output = {key: np.empty(samples, dtype=np.float64) for key in prompt_values}
    for draw in range(samples):
        sampled = np.concatenate([rng.choice(index, size=len(index), replace=True) for index in indices])
        for key, values in prompt_values.items():
            output[key][draw] = float(np.asarray(values, dtype=np.float64)[sampled].mean())
    return output


def percentile_interval(values: np.ndarray, level: float) -> list[float]:
    """Return a two-sided percentile interval at ``level``."""
    if not 0 < level < 1:
        raise ValueError("Interval level must lie strictly between zero and one")
    tail = 100.0 * (1.0 - level) / 2.0
    return [float(value) for value in np.percentile(np.asarray(values), [tail, 100.0 - tail])]


def decision_gate(
    summaries: dict[str, dict], *, parent_margin: float,
    equivalence_margin: float, parent_reconstruction_atol: float,
) -> dict:
    """Apply the mutually exclusive preregistered decision hierarchy."""
    full, teacher, residual = (summaries[key] for key in ("full", "teacher", "residual"))
    effect_pass = (
        full["mean"] < 0
        and full["ci95"][1] < 0
        and abs(full["mean"] - parent_margin) <= parent_reconstruction_atol
    )
    residual_equivalent = (
        residual["ci90"][0] > -equivalence_margin
        and residual["ci90"][1] < equivalence_margin
    )
    teacher_equivalent = (
        teacher["ci90"][0] > -equivalence_margin
        and teacher["ci90"][1] < equivalence_margin
    )
    residual_substantial = residual["ci95"][1] < -equivalence_margin
    teacher_substantial = teacher["ci95"][1] < -equivalence_margin
    residual_opposing = residual["ci95"][0] > equivalence_margin
    residual_underresolved = (
        not residual_equivalent
        and residual["ci95"][1] < 0
        and not residual_substantial
    )
    if not effect_pass:
        classification = "effect_reconstruction_failure"
    elif residual_equivalent:
        classification = "teacher_axis_sufficient"
    elif residual_substantial and teacher_equivalent:
        classification = "residual_dominant"
    elif residual_substantial and teacher_substantial:
        classification = "mixed_teacher_and_residual"
    elif residual_opposing and teacher_substantial:
        classification = "opposing_component_cancellation"
    elif residual_underresolved:
        classification = "residual_directional_underresolved"
    else:
        classification = "non_equivalent_inconclusive"
    return {
        "effect_reconstruction_pass": effect_pass,
        "residual_equivalent": residual_equivalent,
        "teacher_equivalent": teacher_equivalent,
        "residual_substantial_in_inherited_direction": residual_substantial,
        "teacher_substantial_in_inherited_direction": teacher_substantial,
        "residual_substantial_in_opposing_direction": residual_opposing,
        "residual_directional_underresolved": residual_underresolved,
        "classification": classification,
    }

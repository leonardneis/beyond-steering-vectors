"""Independent C18 aggregation, bootstrap, cancellation, and classification."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import numpy as np

from .teacher_coordinate_interchange import (
    cancellation_flags,
    four_cell_quantities,
    iut_pass,
    percentile_interval,
    stratified_bootstrap_indices,
)


QUANTITIES = ("E", "L", "H", "B0", "B1", "I")


def _strict_equivalent(interval: Sequence[float], epsilon: float) -> bool:
    return float(interval[0]) > -epsilon and float(interval[1]) < epsilon


def _material(interval: Sequence[float], epsilon: float) -> bool:
    return float(interval[1]) < -epsilon or float(interval[0]) > epsilon


def _same_direction(interval: Sequence[float], direction: float, epsilon: float) -> bool:
    return float(interval[1]) < -epsilon if direction < 0 else float(interval[0]) > epsilon


def classify_outcome(
    *,
    integrity_pass: bool,
    reconstruction_pass: bool,
    intervals90: Mapping[str, Sequence[float]],
    intervals99: Mapping[str, Sequence[float]],
    cancellation: bool,
    heterogeneous: bool,
    effect_direction: float,
    epsilon: float,
) -> str:
    """Apply the preregistered top-to-bottom disjoint outcome matrix."""
    if not integrity_pass:
        return "technical_integrity_failure"
    if not reconstruction_pass:
        return "effect_reconstruction_failure"
    if iut_pass(intervals90["B0"], intervals90["B1"], epsilon):
        return "cancellation_only_apparent_success" if cancellation else "symmetric_transport"
    if _strict_equivalent(intervals99["B1"], epsilon) and _material(intervals99["B0"], epsilon):
        return "rescue_only"
    if _strict_equivalent(intervals99["B0"], epsilon) and _material(intervals99["B1"], epsilon):
        return "reverse_only"
    if _material(intervals99["I"], epsilon):
        return "interaction_background_dependence"
    if (
        _material(intervals99["B0"], epsilon)
        and _material(intervals99["B1"], epsilon)
        and _same_direction(intervals99["B0"], effect_direction, epsilon)
        and _same_direction(intervals99["B1"], effect_direction, epsilon)
    ):
        return "persistent_parameter_residual"
    if heterogeneous:
        return "heterogeneous_result"
    return "insufficient_precision"


def aggregate_y_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    bootstrap_draws: int = 20000,
    bootstrap_seed: int = 20260804,
    epsilon: float = 0.04405641704135471,
) -> dict[str, object]:
    """Aggregate identified Y rows; no array-position identity is accepted."""
    records = list(rows)
    key_fields = ("condition", "set_id", "prompt_id", "cell")
    keys = [tuple(str(row[field]) for field in key_fields) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate raw Y IDs")
    if any(not np.isfinite(float(row["margin"])) for row in records):
        raise ValueError("nonfinite raw Y value")
    prompts = list(dict.fromkeys(str(row["prompt_id"]) for row in records))
    families_by_prompt = {}
    for row in records:
        prompt = str(row["prompt_id"])
        family = str(row["family"])
        if prompt in families_by_prompt and families_by_prompt[prompt] != family:
            raise ValueError("prompt family changed across rows")
        families_by_prompt[prompt] = family
    if len(prompts) != 72:
        raise ValueError("raw Y inventory does not contain 72 prompts")
    sets = list(dict.fromkeys(str(row["set_id"]) for row in records))
    if "top" not in sets or len(sets) != 26:
        raise ValueError("raw Y inventory does not contain top plus 25 norm sets")
    norms = [value for value in sets if value != "top"]
    lookup = {key: float(row["margin"]) for key, row in zip(keys, records)}
    pooled_locations = sorted({(condition, set_id, prompt) for condition, set_id, prompt, _cell in keys})
    pooled = {
        cell: np.asarray([lookup[(*location, cell)] for location in pooled_locations], dtype=np.float64)
        for cell in ("Y00", "Y01", "Y10", "Y11")
    }
    natural_sd = min(float(np.std(pooled["Y00"])), float(np.std(pooled["Y11"])))
    collapse = False
    if natural_sd > 1e-6:
        collapse = any(
            float(np.std(pooled[cross])) <= 0.1 * natural_sd
            and float(np.mean(np.abs(pooled[cross] - pooled[donor]))) >= epsilon
            for cross, donor in (("Y01", "Y11"), ("Y10", "Y00"))
        )

    q_values: dict[str, dict[tuple[str, str, str], float]] = {q: {} for q in QUANTITIES}
    for condition in ("subliminal", "neutral"):
        for set_id in sets:
            for prompt in prompts:
                cell_values = {
                    cell: np.asarray(lookup[(condition, set_id, prompt, cell)], dtype=np.float64)
                    for cell in ("Y00", "Y01", "Y10", "Y11")
                }
                for q, value in four_cell_quantities(cell_values).items():
                    q_values[q][(condition, set_id, prompt)] = float(value)

    g: dict[str, np.ndarray] = {}
    condition_per_prompt: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
    condition_components: dict[str, dict[str, float]] = defaultdict(dict)
    family_components: dict[str, dict[str, float]] = defaultdict(dict)
    for q in QUANTITIES:
        per_prompt = []
        for prompt in prompts:
            component = {}
            for condition in ("subliminal", "neutral"):
                top = q_values[q][(condition, "top", prompt)]
                control = np.mean([q_values[q][(condition, set_id, prompt)] for set_id in norms])
                component[condition] = top - control
            per_prompt.append(component["subliminal"] - component["neutral"])
        g[q] = np.asarray(per_prompt, dtype=np.float64)
        for condition in ("subliminal", "neutral"):
            values = np.asarray([
                q_values[q][(condition, "top", prompt)]
                - np.mean([q_values[q][(condition, set_id, prompt)] for set_id in norms])
                for prompt in prompts
            ], dtype=np.float64)
            condition_per_prompt[q][condition] = values
            condition_components[q][condition] = float(np.mean(values))
        for family in dict.fromkeys(families_by_prompt.values()):
            indices = [index for index, prompt in enumerate(prompts) if families_by_prompt[prompt] == family]
            family_components[q][family] = float(np.mean(g[q][indices]))

    family_order = [families_by_prompt[prompt] for prompt in prompts]
    boot_indices = stratified_bootstrap_indices(family_order, draws=bootstrap_draws, seed=bootstrap_seed)
    bootstrap = {q: np.mean(g[q][boot_indices], axis=1) for q in QUANTITIES}
    means = {q: float(np.mean(g[q])) for q in QUANTITIES}
    intervals = {
        q: {
            "90": percentile_interval(bootstrap[q], 0.90),
            "95": percentile_interval(bootstrap[q], 0.95),
            "99": percentile_interval(bootstrap[q], 0.99),
        }
        for q in QUANTITIES
    }
    unique_families = list(dict.fromkeys(family_order))
    family_intervals99: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for q in QUANTITIES:
        for family_index, family in enumerate(unique_families):
            sampled = boot_indices[:, family_index * 24 : (family_index + 1) * 24]
            family_intervals99[q][family] = percentile_interval(np.mean(g[q][sampled], axis=1), 0.99)
    condition_intervals99: dict[str, dict[str, tuple[float, float]]] = defaultdict(dict)
    for q in QUANTITIES:
        for condition in ("subliminal", "neutral"):
            condition_intervals99[q][condition] = percentile_interval(
                np.mean(condition_per_prompt[q][condition][boot_indices], axis=1), 0.99
            )
    def opposing(items: Sequence[Sequence[float]]) -> bool:
        return any(
            (left[1] < -epsilon and right[0] > epsilon) or (right[1] < -epsilon and left[0] > epsilon)
            for index, left in enumerate(items) for right in items[index + 1 :]
        )
    heterogeneous = any(
        opposing(list(family_intervals99[q].values())) or opposing(list(condition_intervals99[q].values()))
        for q in ("B0", "B1")
    )
    flags = cancellation_flags(
        {"B0": g["B0"], "B1": g["B1"]}, condition_components, family_components,
        epsilon=epsilon, collapse=collapse,
    )
    reconstruction = (
        means["E"] < 0
        and intervals["E"]["95"][1] < 0
        and abs(means["E"] - (-0.22028208520677353)) <= epsilon
    )
    classification = classify_outcome(
        integrity_pass=True,
        reconstruction_pass=reconstruction,
        intervals90={q: intervals[q]["90"] for q in QUANTITIES},
        intervals99={q: intervals[q]["99"] for q in QUANTITIES},
        cancellation=any(flags.values()),
        heterogeneous=heterogeneous,
        effect_direction=means["E"],
        epsilon=epsilon,
    )
    return {
        "means": means,
        "intervals": intervals,
        "per_prompt": {q: g[q].tolist() for q in QUANTITIES},
        "condition_components": dict(condition_components),
        "family_components": dict(family_components),
        "family_intervals99": dict(family_intervals99),
        "condition_intervals99": dict(condition_intervals99),
        "heterogeneous": heterogeneous,
        "cancellation_flags": flags,
        "output_collapse": collapse,
        "effect_reconstruction_point_interval_pass": reconstruction,
        "classification": classification,
        "bootstrap": {"draws": bootstrap_draws, "seed": bootstrap_seed, "prng": "PCG64"},
    }

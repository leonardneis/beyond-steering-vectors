"""Frozen C18-v2 aggregation, cancellation, Z/W allocation, and classification."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

import numpy as np


EPSILON = 0.04405641704135471
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260804
QUANTITIES = ("E", "L", "H", "B0", "B1", "I")


def percentile_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = 50.0 * (1.0 - confidence)
    low, high = np.percentile(np.asarray(values, dtype=np.float64), [tail, 100.0 - tail])
    return float(low), float(high)


@lru_cache(maxsize=8)
def _stratified_indices_cached(families: tuple[str, ...]) -> np.ndarray:
    labels = np.asarray(families)
    unique = list(dict.fromkeys(labels.tolist()))
    if len(labels) != 72 or len(unique) != 3 or any(np.sum(labels == item) != 24 for item in unique):
        raise ValueError("bootstrap requires three ordered families of 24 prompts")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    result = np.empty((BOOTSTRAP_DRAWS, 72), dtype=np.int64)
    for draw in range(BOOTSTRAP_DRAWS):
        offset = 0
        for family in unique:
            pool = np.flatnonzero(labels == family)
            result[draw, offset : offset + 24] = rng.choice(pool, 24, replace=True)
            offset += 24
    return result


def stratified_indices(families: Sequence[str]) -> np.ndarray:
    return _stratified_indices_cached(tuple(families)).copy()


def eq90(interval: Sequence[float], epsilon: float = EPSILON) -> bool:
    return float(interval[0]) > -epsilon and float(interval[1]) < epsilon


def mat99(interval: Sequence[float], epsilon: float = EPSILON) -> bool:
    return float(interval[1]) < -epsilon or float(interval[0]) > epsilon


def four_cell(cells: Mapping[str, float]) -> dict[str, float]:
    if set(cells) != {"Y00", "Y01", "Y10", "Y11"}:
        raise ValueError("four-cell inventory differs")
    y00, y01, y10, y11 = (float(cells[key]) for key in ("Y00", "Y01", "Y10", "Y11"))
    values = {
        "E": y11 - y00,
        "L": y01 - y00,
        "H": y11 - y10,
        "B0": y10 - y00,
        "B1": y11 - y01,
        "I": y11 - y10 - y01 + y00,
    }
    scale = max(1.0, sum(abs(value) for value in cells.values()))
    tolerance = 64 * np.finfo(np.float64).eps * scale
    checks = (
        values["E"] - values["L"] - values["B1"],
        values["E"] - values["H"] - values["B0"],
        values["I"] - values["H"] + values["L"],
        values["I"] - values["B1"] + values["B0"],
    )
    if max(map(abs, checks)) > tolerance:
        raise ArithmeticError("four-cell algebra failed")
    return values


def _identified(rows: Iterable[Mapping[str, object]], prefix: str, cells: set[str]) -> tuple[list[dict], dict]:
    records = [dict(row) for row in rows]
    keys = [(str(r["condition"]), str(r["set_id"]), str(r["prompt_id"]), str(r["cell"])) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate {prefix} raw ID")
    if any(key[3] not in cells for key in keys):
        raise ValueError(f"unexpected {prefix} cell")
    if any(not np.isfinite(float(row["margin"])) for row in records):
        raise ValueError(f"nonfinite {prefix} margin")
    return records, {key: float(row["margin"]) for key, row in zip(keys, records)}


def _inventory(records: Sequence[Mapping[str, object]]) -> tuple[list[str], list[str], dict[str, str]]:
    prompts = list(dict.fromkeys(str(row["prompt_id"]) for row in records))
    sets = list(dict.fromkeys(str(row["set_id"]) for row in records))
    families = {str(row["prompt_id"]): str(row["family"]) for row in records}
    if len(prompts) != 72 or len(sets) != 26 or "top" not in sets:
        raise ValueError("C18-v2 prompt/set inventory differs")
    norms = [item for item in sets if item != "top"]
    if len(norms) != 25:
        raise ValueError("C18-v2 norm inventory differs")
    return prompts, norms, families


def aggregate_scalar(
    values: Mapping[tuple[str, str, str], float],
    prompts: Sequence[str], norms: Sequence[str], families: Mapping[str, str],
) -> dict[str, object]:
    g = []
    condition = {"subliminal": [], "neutral": []}
    atoms = []
    set_means = {}
    for prompt in prompts:
        components = {}
        for c in ("subliminal", "neutral"):
            per_norm = [values[(c, "top", prompt)] - values[(c, norm, prompt)] for norm in norms]
            components[c] = float(np.mean(per_norm))
            condition[c].append(components[c])
        g.append(components["subliminal"] - components["neutral"])
        atoms.append([
            (values[("subliminal", "top", prompt)] - values[("subliminal", norm, prompt)])
            - (values[("neutral", "top", prompt)] - values[("neutral", norm, prompt)])
            for norm in norms
        ])
    g_array = np.asarray(g, dtype=np.float64)
    atom_array = np.asarray(atoms, dtype=np.float64).T  # norm x prompt
    for index, norm in enumerate(norms):
        set_means[norm] = float(np.mean(atom_array[index]))
    family_order = [families[prompt] for prompt in prompts]
    indices = stratified_indices(family_order)
    draws = np.mean(g_array[indices], axis=1)
    family_means = {
        family: float(np.mean(atom_array[:, [i for i, label in enumerate(family_order) if label == family]]))
        for family in dict.fromkeys(family_order)
    }
    return {
        "G": float(np.mean(atom_array)),
        "interval90": percentile_interval(draws, 0.90),
        "interval95": percentile_interval(draws, 0.95),
        "interval99": percentile_interval(draws, 0.99),
        "per_prompt": g_array.tolist(),
        "atoms": atom_array.tolist(),
        "mean_absolute_atoms": float(np.mean(np.abs(atom_array))),
        "rms_atoms": float(np.sqrt(np.mean(atom_array**2))),
        "atom_quantiles_abs": {
            str(q): float(np.quantile(np.abs(atom_array), q)) for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "positive_fraction": float(np.mean(atom_array > 0)),
        "negative_fraction": float(np.mean(atom_array < 0)),
        "set_means": set_means,
        "family_means": family_means,
        "condition_means": {key: float(np.mean(value)) for key, value in condition.items()},
    }


def _opposing(values: Sequence[float], epsilon: float) -> bool:
    return any(np.sign(left) != np.sign(right) and abs(left - right) >= 2 * epsilon
               for index, left in enumerate(values) for right in values[index + 1 :])


def cancellation_reasons(summary: Mapping[str, object], *, collapse: bool, epsilon: float = EPSILON) -> list[str]:
    reasons = []
    if float(summary["mean_absolute_atoms"]) >= epsilon:
        reasons.append("mean_absolute_25x72")
    if float(summary["rms_atoms"]) >= 2 * epsilon:
        reasons.append("rms_25x72")
    set_means = list(summary["set_means"].values())
    family_means = list(summary["family_means"].values())
    condition_means = list(summary["condition_means"].values())
    if max(map(abs, set_means)) >= epsilon:
        reasons.append("norm_set_mean")
    if max(map(abs, family_means)) >= epsilon:
        reasons.append("family_mean")
    if max(map(abs, condition_means)) >= epsilon:
        reasons.append("condition_mean")
    if _opposing(set_means, epsilon) or _opposing(family_means, epsilon):
        reasons.append("opposing_fixed_groups")
    if (np.sign(condition_means[0]) == np.sign(condition_means[1])
            and min(map(abs, condition_means)) >= epsilon
            and abs(condition_means[0] - condition_means[1]) < epsilon):
        reasons.append("condition_cancellation")
    if collapse:
        reasons.append("cross_cell_output_collapse")
    return reasons


def aggregate_y(rows: Iterable[Mapping[str, object]], epsilon: float = EPSILON) -> dict[str, object]:
    records, lookup = _identified(rows, "Y", {"Y00", "Y01", "Y10", "Y11"})
    prompts, norms, families = _inventory(records)
    expected = 2 * 26 * 72 * 4
    if len(records) != expected:
        raise ValueError(f"Y inventory count differs: {len(records)} != {expected}")
    raw_q = {name: {} for name in QUANTITIES}
    for condition in ("subliminal", "neutral"):
        for set_id in ["top", *norms]:
            for prompt in prompts:
                cells = {cell: lookup[(condition, set_id, prompt, cell)] for cell in ("Y00", "Y01", "Y10", "Y11")}
                for name, value in four_cell(cells).items():
                    raw_q[name][(condition, set_id, prompt)] = value
    summaries = {name: aggregate_scalar(raw_q[name], prompts, norms, families) for name in QUANTITIES}

    locations = sorted({(c, s, p) for c, s, p, _ in lookup})
    natural_sd = min(
        float(np.std([lookup[(*loc, "Y00")] for loc in locations])),
        float(np.std([lookup[(*loc, "Y11")] for loc in locations])),
    )
    collapse = False
    if natural_sd > 1e-6:
        collapse = any(
            float(np.std([lookup[(*loc, cross)] for loc in locations])) <= 0.1 * natural_sd
            and float(np.mean([abs(lookup[(*loc, cross)] - lookup[(*loc, donor)]) for loc in locations])) >= epsilon
            for cross, donor in (("Y01", "Y11"), ("Y10", "Y00"))
        )
    cancellations = {name: cancellation_reasons(summaries[name], collapse=collapse, epsilon=epsilon)
                     for name in ("B0", "B1")}
    reconstruction = (
        summaries["E"]["G"] < 0
        and summaries["E"]["interval95"][1] < 0
        and abs(summaries["E"]["G"] - (-0.22028208520677353)) <= epsilon
    )
    return {
        "quantities": summaries,
        "cancellation": cancellations,
        "output_collapse": collapse,
        "natural_effect_reconstruction": reconstruction,
        "prompt_order": prompts,
        "families": [families[prompt] for prompt in prompts],
    }


def aggregate_factorial(rows: Iterable[Mapping[str, object]], prefix: str) -> dict[str, object]:
    cells = {f"{prefix}{a}{b}" for a in (0, 1) for b in (0, 1)}
    records, lookup = _identified(rows, prefix, cells)
    prompts, norms, families = _inventory(records)
    if len(records) != 2 * 26 * 72 * 4:
        raise ValueError(f"{prefix} inventory count differs")
    raw = {name: {} for name in QUANTITIES}
    for c in ("subliminal", "neutral"):
        for s in ["top", *norms]:
            for p in prompts:
                values = {f"Y{a}{b}": lookup[(c, s, p, f"{prefix}{a}{b}")] for a in (0, 1) for b in (0, 1)}
                for name, value in four_cell(values).items():
                    raw[name][(c, s, p)] = value
    return {name: aggregate_scalar(raw[name], prompts, norms, families) for name in QUANTITIES}


def aggregate_w(rows: Iterable[Mapping[str, object]]) -> dict[str, object]:
    cells = {f"W{a}{b}{r}" for a in (0, 1) for b in (0, 1) for r in (0, 1)}
    records, lookup = _identified(rows, "W", cells)
    prompts, norms, families = _inventory(records)
    if len(records) != 2 * 26 * 72 * 8:
        raise ValueError("W inventory count differs")
    output = {}
    for donor in (0, 1):
        raw = {name: {} for name in (
            "total", "U_suffix0", "S_state1", "U_suffix1", "S_state0", "J"
        )}
        for c in ("subliminal", "neutral"):
            for s in ["top", *norms]:
                for p in prompts:
                    get = lambda a, r: lookup[(c, s, p, f"W{a}{donor}{r}")]
                    values = {
                        "total": get(1, 1) - get(0, 0),
                        "U_suffix0": get(1, 0) - get(0, 0),
                        "S_state1": get(1, 1) - get(1, 0),
                        "U_suffix1": get(1, 1) - get(0, 1),
                        "S_state0": get(0, 1) - get(0, 0),
                        "J": get(1, 1) - get(1, 0) - get(0, 1) + get(0, 0),
                    }
                    scale = max(1.0, sum(abs(get(a, r)) for a in (0, 1) for r in (0, 1)))
                    tol = 64 * np.finfo(np.float64).eps * scale
                    if abs(values["total"] - values["U_suffix0"] - values["S_state1"]) > tol:
                        raise ArithmeticError("W suffix-0 allocation failed")
                    if abs(values["total"] - values["U_suffix1"] - values["S_state0"]) > tol:
                        raise ArithmeticError("W suffix-1 allocation failed")
                    for name, value in values.items():
                        raw[name][(c, s, p)] = value
        output[f"donor_{donor}"] = {
            name: aggregate_scalar(values, prompts, norms, families) for name, values in raw.items()
        }
    return output


def suffix_conflict(z: Mapping[str, object], w: Mapping[str, object], epsilon: float = EPSILON) -> tuple[bool, list[str]]:
    reasons = []
    if mat99(z["I"]["interval99"], epsilon):
        reasons.append("Z_state_suffix_interaction")
    for donor, values in w.items():
        if mat99(values["J"]["interval99"], epsilon):
            reasons.append(f"{donor}_W_J")
        for left, right, label in (
            ("U_suffix0", "S_state1", "allocation_suffix0"),
            ("U_suffix1", "S_state0", "allocation_suffix1"),
        ):
            lval, rval = values[left], values[right]
            if mat99(lval["interval99"], epsilon) and mat99(rval["interval99"], epsilon) \
                    and np.sign(lval["G"]) != np.sign(rval["G"]):
                reasons.append(f"{donor}_{label}_opposing_material")
    return bool(reasons), reasons


def classify(
    *, integrity: bool, reconstruction: bool, y: Mapping[str, object],
    suffix_conflict_value: bool, epsilon: float = EPSILON,
) -> str:
    if not integrity or not reconstruction:
        return "F_TECHNICAL_IDENTIFICATION_FAILURE"
    q = y["quantities"]
    e0, e1 = eq90(q["B0"]["interval90"], epsilon), eq90(q["B1"]["interval90"], epsilon)
    m0, m1 = mat99(q["B0"]["interval99"], epsilon), mat99(q["B1"]["interval99"], epsilon)
    if (e0 and m1) or (e1 and m0):
        return "B_ASYMMETRIC_TRANSPORT"
    if mat99(q["I"]["interval99"], epsilon) or suffix_conflict_value:
        return "C_STATE_PARAMETER_INTERACTION_COADAPTATION"
    cancel = bool(y["cancellation"]["B0"] or y["cancellation"]["B1"])
    if e0 and e1 and cancel:
        return "E_CANCELLATION_HETEROGENEITY_LIMITED"
    if e0 and e1:
        return "A_BIDIRECTIONAL_FUNCTIONAL_TRANSPORT"
    if not e0 and not e1 and (m0 or m1):
        return "D_TEACHER_COORDINATE_INSUFFICIENCY"
    return "G_INSUFFICIENT_PRECISION_UNRESOLVED"

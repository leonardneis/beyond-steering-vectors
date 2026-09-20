"""Independent C18-v2 scientific recomputation used only by the auditor.

This module intentionally does not import the production aggregation module.
"""

from __future__ import annotations

from typing import Mapping, Sequence
from functools import lru_cache

import numpy as np


EPS = 0.04405641704135471


def interval(draws: np.ndarray, level: float) -> list[float]:
    tail = 50 * (1 - level)
    return [float(x) for x in np.percentile(draws, [tail, 100 - tail])]


@lru_cache(maxsize=8)
def _bootstrap_cached(families: tuple[str, ...]) -> np.ndarray:
    labels = np.asarray(families)
    ordered = list(dict.fromkeys(labels.tolist()))
    if len(ordered) != 3 or any(np.sum(labels == label) != 24 for label in ordered):
        raise ValueError("independent audit family inventory differs")
    generator = np.random.Generator(np.random.PCG64(20260804))
    result = np.empty((20000, 72), dtype=np.int64)
    for draw in range(20000):
        cursor = 0
        for label in ordered:
            source = np.flatnonzero(labels == label)
            result[draw, cursor : cursor + 24] = generator.choice(source, 24, replace=True)
            cursor += 24
    return result


def bootstrap(families: Sequence[str]) -> np.ndarray:
    return _bootstrap_cached(tuple(families)).copy()


def cube(rows: Sequence[Mapping[str, object]], cell_names: Sequence[str]) -> tuple[np.ndarray, list[str], list[str]]:
    prompts = list(dict.fromkeys(str(row["prompt_id"]) for row in rows))
    sets_seen = list(dict.fromkeys(str(row["set_id"]) for row in rows))
    sets = ["top", *[item for item in sets_seen if item != "top"]]
    families_by_prompt = {}
    lookup = {}
    for row in rows:
        key = (str(row["condition"]), str(row["set_id"]), str(row["prompt_id"]), str(row["cell"]))
        if key in lookup:
            raise ValueError("independent audit duplicate raw ID")
        lookup[key] = float(row["margin"])
        families_by_prompt[str(row["prompt_id"])] = str(row["family"])
    if len(prompts) != 72 or len(sets) != 26:
        raise ValueError("independent audit prompt/set inventory differs")
    array = np.empty((2, 26, 72, len(cell_names)), dtype=np.float64)
    for ci, condition in enumerate(("subliminal", "neutral")):
        for si, set_id in enumerate(sets):
            for pi, prompt in enumerate(prompts):
                for xi, cell in enumerate(cell_names):
                    array[ci, si, pi, xi] = lookup[(condition, set_id, prompt, cell)]
    return array, prompts, [families_by_prompt[prompt] for prompt in prompts]


def summarize(values: np.ndarray, families: Sequence[str]) -> dict[str, object]:
    # values: condition x set x prompt; set 0 top, sets 1: norms.
    per_condition_atoms = values[:, :1, :] - values[:, 1:, :]
    atoms = per_condition_atoms[0] - per_condition_atoms[1]  # norm x prompt
    per_prompt = np.mean(atoms, axis=0)
    indices = bootstrap(families)
    draws = np.mean(per_prompt[indices], axis=1)
    family_means = {}
    for family in dict.fromkeys(families):
        chosen = [index for index, label in enumerate(families) if label == family]
        family_means[family] = float(np.mean(atoms[:, chosen]))
    return {
        "G": float(np.mean(atoms)),
        "interval90": interval(draws, .90),
        "interval95": interval(draws, .95),
        "interval99": interval(draws, .99),
        "per_prompt": per_prompt.tolist(),
        "atoms": atoms.tolist(),
        "mean_absolute_atoms": float(np.mean(np.abs(atoms))),
        "rms_atoms": float(np.sqrt(np.mean(np.square(atoms)))),
        "set_means": {f"norm_{index:02d}": float(np.mean(atoms[index])) for index in range(25)},
        "family_means": family_means,
        "condition_means": {
            "subliminal": float(np.mean(per_condition_atoms[0])),
            "neutral": float(np.mean(per_condition_atoms[1])),
        },
    }


def equivalent(bounds: Sequence[float]) -> bool:
    return bounds[0] > -EPS and bounds[1] < EPS


def material(bounds: Sequence[float]) -> bool:
    return bounds[1] < -EPS or bounds[0] > EPS


def cancellation(summary: Mapping[str, object], collapse: bool) -> list[str]:
    reasons = []
    if summary["mean_absolute_atoms"] >= EPS: reasons.append("mean_absolute_25x72")
    if summary["rms_atoms"] >= 2 * EPS: reasons.append("rms_25x72")
    set_values = list(summary["set_means"].values())
    family_values = list(summary["family_means"].values())
    condition_values = list(summary["condition_means"].values())
    if max(map(abs, set_values)) >= EPS: reasons.append("norm_set_mean")
    if max(map(abs, family_values)) >= EPS: reasons.append("family_mean")
    if max(map(abs, condition_values)) >= EPS: reasons.append("condition_mean")
    def opposing(items):
        return any(np.sign(x) != np.sign(y) and abs(x-y) >= 2*EPS
                   for i, x in enumerate(items) for y in items[i+1:])
    if opposing(set_values) or opposing(family_values): reasons.append("opposing_fixed_groups")
    if (np.sign(condition_values[0]) == np.sign(condition_values[1])
            and min(map(abs, condition_values)) >= EPS
            and abs(condition_values[0]-condition_values[1]) < EPS):
        reasons.append("condition_cancellation")
    if collapse: reasons.append("cross_cell_output_collapse")
    return reasons


def recompute_y(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    data, _prompts, families = cube(rows, ("Y00", "Y01", "Y10", "Y11"))
    y00, y01, y10, y11 = (data[..., index] for index in range(4))
    raw = {"E": y11-y00, "L": y01-y00, "H": y11-y10,
           "B0": y10-y00, "B1": y11-y01, "I": y11-y10-y01+y00}
    scale = np.maximum(1.0, np.abs(y00)+np.abs(y01)+np.abs(y10)+np.abs(y11))
    tol = 64*np.finfo(np.float64).eps*scale
    if np.any(np.abs(raw["E"]-raw["L"]-raw["B1"]) > tol) or np.any(np.abs(raw["E"]-raw["H"]-raw["B0"]) > tol):
        raise ArithmeticError("independent Y algebra failed")
    summaries = {name: summarize(values, families) for name, values in raw.items()}
    natural_sd = min(float(np.std(y00)), float(np.std(y11)))
    collapse = False
    if natural_sd > 1e-6:
        collapse = ((float(np.std(y01)) <= .1*natural_sd and float(np.mean(np.abs(y01-y11))) >= EPS)
                    or (float(np.std(y10)) <= .1*natural_sd and float(np.mean(np.abs(y10-y00))) >= EPS))
    cancels = {name: cancellation(summaries[name], collapse) for name in ("B0", "B1")}
    reconstruction = summaries["E"]["G"] < 0 and summaries["E"]["interval95"][1] < 0 \
        and abs(summaries["E"]["G"] + .22028208520677353) <= EPS
    return {"quantities": summaries, "cancellation": cancels, "output_collapse": collapse,
            "natural_effect_reconstruction": reconstruction, "families": families}


def recompute_z(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    data, _prompts, families = cube(rows, ("Z00", "Z01", "Z10", "Z11"))
    z00, z01, z10, z11 = (data[..., index] for index in range(4))
    raw = {"E": z11-z00, "L": z01-z00, "H": z11-z10,
           "B0": z10-z00, "B1": z11-z01, "I": z11-z10-z01+z00}
    return {name: summarize(value, families) for name, value in raw.items()}


def recompute_w(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    names = tuple(f"W{a}{b}{r}" for a in (0,1) for b in (0,1) for r in (0,1))
    data, _prompts, families = cube(rows, names)
    position = {name: index for index, name in enumerate(names)}
    output = {}
    for donor in (0,1):
        get = lambda a, r: data[..., position[f"W{a}{donor}{r}"]]
        raw = {
            "total": get(1,1)-get(0,0), "U_suffix0": get(1,0)-get(0,0),
            "S_state1": get(1,1)-get(1,0), "U_suffix1": get(1,1)-get(0,1),
            "S_state0": get(0,1)-get(0,0), "J": get(1,1)-get(1,0)-get(0,1)+get(0,0),
        }
        if not np.allclose(raw["total"], raw["U_suffix0"]+raw["S_state1"], rtol=0, atol=1e-12):
            raise ArithmeticError("independent W allocation failed")
        output[f"donor_{donor}"] = {name: summarize(value, families) for name, value in raw.items()}
    return output


def conflict(z: Mapping[str, object], w: Mapping[str, object]) -> tuple[bool, list[str]]:
    reasons=[]
    if material(z["I"]["interval99"]): reasons.append("Z_state_suffix_interaction")
    for donor, values in w.items():
        if material(values["J"]["interval99"]): reasons.append(f"{donor}_W_J")
        for left,right,label in (("U_suffix0","S_state1","allocation_suffix0"),
                                 ("U_suffix1","S_state0","allocation_suffix1")):
            if material(values[left]["interval99"]) and material(values[right]["interval99"]) \
                    and np.sign(values[left]["G"]) != np.sign(values[right]["G"]):
                reasons.append(f"{donor}_{label}_opposing_material")
    return bool(reasons), reasons


def decision(y: Mapping[str, object], suffix: bool) -> str:
    if not y["natural_effect_reconstruction"]: return "F_TECHNICAL_IDENTIFICATION_FAILURE"
    q=y["quantities"]; e0=equivalent(q["B0"]["interval90"]); e1=equivalent(q["B1"]["interval90"])
    m0=material(q["B0"]["interval99"]); m1=material(q["B1"]["interval99"])
    if (e0 and m1) or (e1 and m0): return "B_ASYMMETRIC_TRANSPORT"
    if material(q["I"]["interval99"]) or suffix: return "C_STATE_PARAMETER_INTERACTION_COADAPTATION"
    cancel=bool(y["cancellation"]["B0"] or y["cancellation"]["B1"])
    if e0 and e1 and cancel: return "E_CANCELLATION_HETEROGENEITY_LIMITED"
    if e0 and e1: return "A_BIDIRECTIONAL_FUNCTIONAL_TRANSPORT"
    if not e0 and not e1 and (m0 or m1): return "D_TEACHER_COORDINATE_INSUFFICIENCY"
    return "G_INSUFFICIENT_PRECISION_UNRESOLVED"

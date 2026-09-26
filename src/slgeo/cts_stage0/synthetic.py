"""Synthetic, planted-truth data for the decision engine, labels, fragility and artifact protocol.

No model and no real data: word scores are generated for the registry's condition ids under a scenario of
planted effects. Used by the unit tests and by the CPU technical-validation node, so the complete criteria,
decision, label and fragility paths run in the execution environment before any scientific output exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import artifacts as art
from .conditions import Condition, registry_conditions
from .criteria import ANIMAL_WORDS, WordScores, all_statistics, evaluate
from .fragility import fragility_report
from .statistics import FamilyIndex

WORDS = ("cat", "dog", "wolf", "lion", "horse", "rabbit", "elephant", "fox", "owl", "turtle", "spider", "ant", "chess", "blue")
SYNTHETIC_SEED = 1000011
CONTRASTS = ("c_cat_dog", "c_cat_wolf", "c_cat_anim")


def synthetic_prompt_ids() -> dict[str, list[str]]:
    # Prefixed so synthetic ids can never coincide with real S0/D/C prompt ids.
    return {family: [f"synthetic-{family}-{i:03d}" for i in range(1, 101)] for family in ("direct", "identity", "hypothetical")}


def _contrast_of(direction: str) -> tuple[str | None, str]:
    """(tested contrast, variant) of a named direction: variant in main, T2, T3, perpG, slot27."""
    for contrast in CONTRASTS:
        if direction == contrast:
            return contrast, "main"
        if direction.startswith(contrast + "_"):
            return contrast, direction[len(contrast) + 1 :]
        if direction == f"{contrast}@27":
            return contrast, "slot27"
    return None, ""


def _effect(condition: Condition, scenario: Mapping[str, Any]) -> dict[str, float]:
    d = condition.direction or ""
    if condition.kind == "persona":
        if condition.persona in ("P_dog_T1", "P_wolf_T1"):
            x = condition.persona.split("_")[1]
            return {x: scenario["a6"][x]}
        return {}
    if condition.kind != "steer":
        return {}
    contrast, variant = _contrast_of(d)
    if contrast is not None and condition.magnitude == f"tau:{d}":
        strength = (scenario["perp"][contrast] if variant == "perpG" else scenario["ts"][contrast]) * condition.kappa
        x = contrast.removeprefix("c_cat_")
        if condition.sign > 0:
            return {"cat": strength}
        shift = {"cat": -strength}
        if x != "anim":
            shift[x] = strength
        return shift
    if d == "t_cat" and condition.scale == "raw":
        return {"cat": scenario["pc_pos"] if condition.mode == "all" else scenario["pc"]}
    if d == "t_cat":
        return {"cat": scenario["pc_star"][condition.magnitude.removeprefix("tau:")]}
    if d in ("t_dog", "t_wolf"):
        return {d.removeprefix("t_"): scenario["a7"][d.removeprefix("t_")]}
    if d.startswith("m_cat_"):
        return {"cat": scenario["mention"]}
    if d in ("g_anim", "g_tmpl", "g_id"):
        shared = scenario["shared"][d]
        shift = {word: shared["mass"] for word in ANIMAL_WORDS}
        shift["cat"] = shift["cat"] + shared["cat_selective"]
        return shift
    return {}


def synthetic_scores(rows: Sequence[Mapping[str, Any]], scenario: Mapping[str, Any], seed: int = SYNTHETIC_SEED,
                     *, only: set[str] | None = None, noise: float | None = None) -> WordScores:
    rng = np.random.default_rng(seed)
    ids = [pid for family in synthetic_prompt_ids().values() for pid in family]
    base_L = {word: np.log(scenario.get("base_rate", {}).get(word, 0.05)) for word in WORDS}
    baseline = {pid: np.array([base_L[w] for w in WORDS]) + rng.normal(0, 0.05, len(WORDS)) for pid in ids}
    sd = scenario.get("noise", 0.05) if noise is None else noise
    values: dict[str, dict[str, np.ndarray]] = {}
    for condition in registry_conditions(rows):
        if only is not None and condition.cid not in only:
            continue
        if condition.kind == "unsteered" or condition.cid == "persona:P_default":
            values[condition.cid] = {pid: row + rng.normal(0, sd * 0.01, len(WORDS)) for pid, row in baseline.items()}
            continue
        shift = _effect(condition, scenario)
        table = {}
        for pid in ids:
            row = baseline[pid] + rng.normal(0, sd, len(WORDS))
            for word, value in shift.items():
                row[WORDS.index(word)] += value
            table[pid] = row
        values[condition.cid] = table
    return WordScores(WORDS, values)


def default_scenario() -> dict:
    return {
        "ts": {c: 1.0 for c in CONTRASTS},
        "perp": {c: 1.0 for c in CONTRASTS},
        "pc": 2.0, "pc_pos": 2.0,
        "pc_star": {c: 1.0 for c in CONTRASTS},
        "a7": {"dog": 1.0, "wolf": 1.0},
        "a6": {"dog": 1.0, "wolf": 1.0},
        "mention": 0.2,
        "shared": {g: {"mass": 0.5, "cat_selective": 0.0} for g in ("g_anim", "g_tmpl", "g_id")},
        "base_rate": {"dog": 0.1, "wolf": 0.1},
    }


def scenarios() -> dict[str, tuple[dict, str, str | None]]:
    """name -> (scenario, expected class, expected X or Stage-2a primary)."""
    out = {}
    out["go_dog"] = (default_scenario(), "GO_X", "dog")
    s = default_scenario(); s["base_rate"]["dog"] = 0.9
    out["go_wolf_a4"] = (s, "GO_X", "wolf")
    s = default_scenario(); s["a6"] = {"dog": 0.0, "wolf": 0.0}
    out["go_cat_only"] = (s, "GO_CAT_ONLY", "c_cat_anim")
    s = default_scenario(); s["pc"] = 0.0; s["pc_pos"] = 0.0
    out["stop_instrument"] = (s, "STOP_INSTRUMENT", None)
    s = default_scenario(); s["pc"] = 0.0
    out["inconclusive_position"] = (s, "INCONCLUSIVE_POSITION", None)
    s = default_scenario(); s["ts"] = {k: 0.0 for k in s["ts"]}; s["pc_star"] = {k: 0.0 for k in s["pc_star"]}
    out["inconclusive_dose"] = (s, "INCONCLUSIVE_DOSE", None)
    s = default_scenario(); s["perp"] = {k: 0.0 for k in s["perp"]}
    out["pivot_shared_leakage"] = (s, "PIVOT_SHARED_LEAKAGE", None)
    s = default_scenario(); s["ts"] = {k: 0.0 for k in s["ts"]}
    out["pivot"] = (s, "PIVOT_NO_BASE_VALIDATED_CONTRAST", None)
    s = default_scenario(); s["mention"] = 1.5
    out["lexical_label"] = (s, "GO_X", "dog")
    s = default_scenario(); s["shared"]["g_anim"]["cat_selective"] = 0.6
    out["shared_label_selective"] = (s, "GO_X", "dog")
    return out


def run_scenario(rows, null_names, scenario: dict, reliabilities: dict | None = None) -> dict:
    family = FamilyIndex.from_ids(synthetic_prompt_ids())
    reliabilities = reliabilities or {**{c: 0.99 for c in CONTRASTS}, **{f"{c}_perpG": 0.99 for c in CONTRASTS}, "t_cat": 0.999}
    conditions = registry_conditions(rows)
    baselines = {c.cid: c.baseline for c in conditions}
    result, _ctx = evaluate(synthetic_scores(rows, scenario), family, reliabilities, null_names, True, baselines)
    return result


def synthetic_decision_suite(rows, null_names) -> dict:
    results = {}
    for name, (scenario, expected, detail) in scenarios().items():
        outcome = run_scenario(rows, null_names, scenario)
        decision = outcome["decision"]
        ok = decision["class"] == expected
        if expected == "GO_X":
            ok = ok and decision.get("X") == detail
        if expected == "GO_CAT_ONLY":
            ok = ok and decision.get("stage2a_primary") == detail
        if name == "lexical_label":
            ok = ok and outcome["labels"]["c_cat_dog"]["mention"]["label"] == "LEXICAL_NOT_EXCLUDED"
        if name == "go_dog":
            ok = ok and all(entry["mention"]["label"] == "PREFERENCE_CONTRAST" for entry in outcome["labels"].values())
            ok = ok and all(v["label"] == "BASE_SHARED_DIRECTION" for entry in outcome["labels"].values() for v in entry["shared"].values())
        if name == "shared_label_selective":
            ok = ok and outcome["labels"]["c_cat_dog"]["shared"]["g_anim"]["label"] == "NOT_BASE_SHARED"
            ok = ok and outcome["labels"]["c_cat_dog"]["shared"]["g_tmpl"]["label"] == "BASE_SHARED_DIRECTION"
        if name == "pivot":
            ok = ok and outcome["labels"] == {"BASE_SHARED_DIRECTION": "NOT_ASSESSABLE"}
        results[name] = bool(ok)
    return {"scenarios": results, "pass": all(results.values())}


def synthetic_fragility_suite(rows, null_names) -> dict:
    """The fragility check passes when the reference layout differs by noise far below epsilon x SE and fails
    when it differs by a shift comparable to the SE."""
    family = FamilyIndex.from_ids(synthetic_prompt_ids())
    conditions = registry_conditions(rows)
    baselines = {c.cid: c.baseline for c in conditions}
    rescored = {c.cid for c in conditions if c.reference_rescore}
    reliabilities = {**{c: 0.99 for c in CONTRASTS}, **{f"{c}_perpG": 0.99 for c in CONTRASTS}, "t_cat": 0.999}
    scenario = default_scenario()
    l2_scores = synthetic_scores(rows, scenario)
    l2 = all_statistics(l2_scores, family, reliabilities, null_names, baselines)
    out = {}
    for name, jitter in (("tiny_layout_noise", 1e-5), ("se_sized_layout_noise", 0.2)):
        rng = np.random.default_rng(SYNTHETIC_SEED + 7)
        values = {cid: {pid: row + rng.normal(0, jitter, row.shape) for pid, row in l2_scores.values[cid].items()} for cid in rescored}
        report = fragility_report(l2, WordScores(l2_scores.words, values), rescored)
        out[name] = report["pass"]
    checks = {"tiny_noise_passes": bool(out["tiny_layout_noise"]), "se_sized_noise_fails": not out["se_sized_layout_noise"]}
    return {"checks": checks, "pass": all(checks.values())}


def artifact_drill(root: Path) -> dict:
    """Atomic publication, partial-shard quarantine and hash-verified resume on the shared filesystem."""
    root = Path(root) / art.utc_now().replace(":", "").replace(".", "")
    identity = {"execution_commit": "drill"}
    shard = art.Shard(root, "drill", "spec", identity)
    checks = {}
    shard.directory.mkdir(parents=True)
    (shard.directory / "partial.npz").write_bytes(b"truncated")
    checks["partial_not_complete"] = not shard.is_complete()
    checks["quarantined"] = shard.quarantine() is not None and not shard.directory.exists()
    shard.publish({"data.npz": art.npz_bytes({"x": np.arange(3.0)})}, {"stage": "drill"})
    checks["published_complete"] = shard.is_complete()
    (shard.directory / "data.npz").write_bytes(b"corrupted")
    checks["corruption_detected"] = not shard.is_complete()
    other = art.Shard(root, "drill", "spec", {"execution_commit": "other"})
    checks["other_identity_not_complete"] = not other.is_complete()
    try:
        art.atomic_write_bytes(root / "once.bin", b"a", write_once=True)
        art.atomic_write_bytes(root / "once.bin", b"b", write_once=True)
        checks["write_once_enforced"] = False
    except art.ArtifactError:
        checks["write_once_enforced"] = True
    checks["no_incoming_left"] = not list(root.rglob("*.incoming"))
    return {"checks": checks, "pass": all(checks.values())}

"""Synthetic, planted-truth data for the decision engine and the artifact protocol (no model, no real data).

Used by the unit tests and by the CPU technical-validation node, so the complete criteria/decision path is
exercised in the execution environment before any scientific output exists.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import artifacts as art
from .conditions import gating_conditions
from .criteria import WordScores, evaluate
from .statistics import FamilyIndex

WORDS = ("cat", "dog", "wolf", "lion", "horse", "rabbit", "elephant", "fox", "owl", "turtle", "spider", "ant", "chess", "blue")
SYNTHETIC_SEED = 1000011


def synthetic_prompt_ids() -> dict[str, list[str]]:
    # Prefixed so synthetic ids can never coincide with real S0/D/C prompt ids.
    return {family: [f"synthetic-{family}-{i:03d}" for i in range(1, 101)] for family in ("direct", "identity", "hypothetical")}


def null_names_synthetic() -> list[str]:
    words = [f"n{i:02d}" for i in range(16)]
    return [f"null:{a}>{b}" for a in words for b in words if a != b]


def _effects(scenario: dict) -> dict[str, dict[str, float]]:
    """Mean shift of L_w per condition id under a scenario; unspecified conditions have no effect."""
    effects: dict[str, dict[str, float]] = {}
    for condition in gating_conditions(null_names_synthetic()):
        cid, shift = condition.cid, {}
        d = condition.direction or ""
        base = d.split("_T")[0] if d.startswith("c_cat_") else d
        if condition.kind == "steer" and base in ("c_cat_dog", "c_cat_wolf", "c_cat_anim") and condition.magnitude == f"tau:{d}":
            strength = scenario["ts"][base] * condition.kappa
            x = base.removeprefix("c_cat_")
            if condition.sign > 0:
                shift = {"cat": strength}
            else:
                shift = {"cat": -strength}
                if x != "anim":
                    shift[x] = strength
        elif condition.kind == "steer" and d == "t_cat" and condition.scale == "raw":
            shift = {"cat": scenario["pc_pos"] if condition.mode == "all" else scenario["pc"]}
        elif condition.kind == "steer" and d == "t_cat":
            shift = {"cat": scenario["pc_star"][condition.magnitude.removeprefix("tau:c_cat_")]}
        elif condition.kind == "steer" and d in ("t_dog", "t_wolf"):
            shift = {d.removeprefix("t_"): scenario["a7"][d.removeprefix("t_")]}
        elif condition.kind == "steer" and d.startswith("m_cat_"):
            shift = {"cat": scenario["mention"]}
        elif condition.kind == "persona" and condition.persona in ("P_dog_T1", "P_wolf_T1"):
            x = condition.persona.split("_")[1]
            shift = {x: scenario["a6"][x]}
        effects[cid] = shift
    return effects


def synthetic_scores(scenario: dict, seed: int = SYNTHETIC_SEED) -> WordScores:
    rng = np.random.default_rng(seed)
    ids = [pid for family in synthetic_prompt_ids().values() for pid in family]
    base_L = {word: np.log(scenario.get("base_rate", {}).get(word, 0.05)) for word in WORDS}
    baseline = {pid: np.array([base_L[w] for w in WORDS]) + rng.normal(0, 0.05, len(WORDS)) for pid in ids}
    values = {"unsteered": baseline, "persona:P_default": {pid: row.copy() for pid, row in baseline.items()}}
    for cid, shift in _effects(scenario).items():
        table = {}
        for pid in ids:
            row = baseline[pid] + rng.normal(0, scenario.get("noise", 0.05), len(WORDS))
            for word, value in shift.items():
                row[WORDS.index(word)] += value
            table[pid] = row
        values[cid] = table
    return WordScores(WORDS, values)


def default_scenario() -> dict:
    return {
        "ts": {"c_cat_dog": 1.0, "c_cat_wolf": 1.0, "c_cat_anim": 1.0},
        "pc": 2.0, "pc_pos": 2.0,
        "pc_star": {"dog": 1.0, "wolf": 1.0, "anim": 1.0},
        "a7": {"dog": 1.0, "wolf": 1.0},
        "a6": {"dog": 1.0, "wolf": 1.0},
        "mention": 0.2,
        "base_rate": {"dog": 0.1, "wolf": 0.1},
    }


def scenarios() -> dict[str, tuple[dict, str, str | None]]:
    """name -> (scenario, expected class, expected X or stage-2a primary)."""
    out = {}
    s = default_scenario()
    out["go_dog"] = (s, "GO_X", "dog")
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
    s = default_scenario(); s["ts"] = {k: 0.0 for k in s["ts"]}
    out["pivot"] = (s, "PIVOT_NO_BASE_VALIDATED_CONTRAST", None)
    s = default_scenario(); s["mention"] = 1.5
    out["lexical_label"] = (s, "GO_X", "dog")
    return out


def run_scenario(scenario: dict, reliabilities: dict | None = None) -> dict:
    family = FamilyIndex.from_ids(synthetic_prompt_ids())
    reliabilities = reliabilities or {"c_cat_dog": 0.99, "c_cat_wolf": 0.99, "c_cat_anim": 0.99, "t_cat": 0.999}
    return evaluate(synthetic_scores(scenario), family, reliabilities, null_names_synthetic(), integrity_ok=True)


def synthetic_decision_suite() -> dict:
    results = {}
    for name, (scenario, expected, detail) in scenarios().items():
        outcome = run_scenario(scenario)
        decision = outcome["decision"]
        ok = decision["class"] == expected
        if expected == "GO_X":
            ok = ok and decision.get("X") == detail
        if expected == "GO_CAT_ONLY":
            ok = ok and decision.get("stage2a_primary") == detail
        if name == "lexical_label":
            ok = ok and outcome["labels"]["c_cat_dog"]["label"] == "LEXICAL_NOT_EXCLUDED"
        if name == "go_dog":
            ok = ok and all(label["label"] == "PREFERENCE_CONTRAST" for label in outcome["labels"].values())
        results[name] = bool(ok)
    return {"scenarios": results, "pass": all(results.values())}


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

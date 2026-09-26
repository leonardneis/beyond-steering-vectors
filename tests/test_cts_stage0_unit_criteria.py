from __future__ import annotations

from pathlib import Path
import itertools
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.cts_stage0 import criteria as cr
from slgeo.cts_stage0 import statistics as st
from slgeo.cts_stage0 import synthetic
from slgeo.cts_stage0.criteria import CriteriaError, DecisionInputs, WordScores, decide, off_target
from slgeo.cts_stage0.package import FrozenPackage

CANDS = ("dog", "wolf")
CONTRASTS = ("c_cat_dog", "c_cat_wolf", "c_cat_anim")


def _inputs(**changes) -> DecisionInputs:
    """All-True defaults; nested keys given as e.g. TS_dog=False, A4_wolf=False, PC_star_anim=False."""
    values = {
        "integrity_ok": True, "PC": True, "PC_pos": True, "R_t_cat": True,
        "TS": {c: True for c in CONTRASTS},
        "A4": {x: True for x in CANDS}, "A6": {x: True for x in CANDS}, "A7": {x: True for x in CANDS},
        "PC_star": {k: True for k in ("dog", "wolf", "anim")},
    }
    for key, value in changes.items():
        if key.startswith("TS_"):
            values["TS"][f"c_cat_{key[3:]}"] = value
        elif key.startswith("PC_star_"):
            values["PC_star"][key[len("PC_star_"):]] = value
        elif key[:3] in ("A4_", "A6_", "A7_"):
            values[key[:2]][key[3:]] = value
        else:
            assert key in values, key
            values[key] = value
    return DecisionInputs(**values)


ALL_TS_F = {"TS_dog": False, "TS_wolf": False, "TS_anim": False}
ALL_PCS_F = {"PC_star_dog": False, "PC_star_wolf": False, "PC_star_anim": False}

# (case, changes, class, X, stage2a_primary). D4, D6, D8, D11 and D14 are not in the task table; they are
# additional cases derived from the spec ``decision.order``.
TRUTH_TABLE = [
    ("D0", {}, "GO_X", "dog", "c_cat_anim"),
    ("D1", {"integrity_ok": False}, "TECHNICAL_FAIL", None, None),
    ("D2", {"PC": False, "PC_pos": False}, "STOP_INSTRUMENT", None, None),
    ("D3", {"R_t_cat": False}, "STOP_INSTRUMENT", None, None),
    ("D4", {"R_t_cat": False, "PC": False}, "STOP_INSTRUMENT", None, None),
    ("D5", {"PC": False}, "INCONCLUSIVE_POSITION", None, None),
    ("D6", {"PC": False, **ALL_TS_F, **ALL_PCS_F}, "INCONCLUSIVE_POSITION", None, None),
    ("D7", {"PC_pos": False}, "GO_X", "dog", "c_cat_anim"),
    ("D8", {"A6_dog": False}, "GO_X", "wolf", "c_cat_anim"),
    ("D9", {"TS_anim": False}, "GO_X", "dog", "c_cat_dog"),
    ("D10", {"A4_dog": False}, "GO_X", "wolf", "c_cat_anim"),
    ("D11", {"A7_dog": False, "TS_anim": False}, "GO_X", "wolf", "c_cat_wolf"),
    ("D12", {"TS_dog": False}, "GO_X", "wolf", "c_cat_anim"),
    ("D13", {"A7_dog": False, "TS_wolf": False, "TS_anim": False}, "GO_CAT_ONLY", None, "c_cat_dog"),
    ("D14", {"A4_dog": False, "A4_wolf": False}, "GO_CAT_ONLY", None, "c_cat_anim"),
    ("D15", {"TS_dog": False, "A4_wolf": False, "TS_anim": False}, "GO_CAT_ONLY", None, "c_cat_wolf"),
    ("D16", {"TS_dog": False, "TS_wolf": False}, "GO_CAT_ONLY", None, "c_cat_anim"),
    ("D17", {**ALL_TS_F, **ALL_PCS_F}, "INCONCLUSIVE_DOSE", None, None),
    ("D18", {**ALL_TS_F, "PC_star_wolf": False, "PC_star_anim": False}, "PIVOT_NO_BASE_VALIDATED_CONTRAST", None, None),
    ("D19", {**ALL_TS_F, "PC_star_dog": False, "PC_star_anim": False}, "PIVOT_NO_BASE_VALIDATED_CONTRAST", None, None),
    ("D20", {**ALL_TS_F, "PC_star_dog": False, "PC_star_wolf": False}, "PIVOT_NO_BASE_VALIDATED_CONTRAST", None, None),
    ("D21", {**ALL_TS_F, **ALL_PCS_F, "integrity_ok": False}, "TECHNICAL_FAIL", None, None),
]


@pytest.mark.parametrize("case, changes, klass, x, primary", TRUTH_TABLE, ids=[row[0] for row in TRUTH_TABLE])
def test_decision_truth_table(case, changes, klass, x, primary):
    decision = decide(_inputs(**changes))
    assert decision["class"] == klass
    assert decision["rank"] == cr.DECISION_CLASSES.index(klass) + 1
    if klass == "GO_X":
        assert decision["X"] == x and decision["stage1_trait"] == x
    if primary is not None:
        assert decision["stage2a_primary"] == primary
    if klass == "GO_CAT_ONLY":
        assert decision["stage1"] == "not started"
        assert "X" not in decision


def _reference_decide(v: DecisionInputs) -> tuple[str, str | None, str | None]:
    """Independent restatement of spec decision.order."""
    if not v.integrity_ok:
        return "TECHNICAL_FAIL", None, None
    if (not v.PC and not v.PC_pos) or not v.R_t_cat:
        return "STOP_INSTRUMENT", None, None
    if not v.PC and v.PC_pos:
        return "INCONCLUSIVE_POSITION", None, None
    eligible = [x for x in ("dog", "wolf") if v.A4[x] and v.TS[f"c_cat_{x}"] and v.A6[x] and v.A7[x]]
    if eligible:
        x = eligible[0]
        return "GO_X", x, "c_cat_anim" if v.TS["c_cat_anim"] else f"c_cat_{x}"
    passing = [c for c in ("c_cat_dog", "c_cat_wolf") if v.TS[c]]
    if v.TS["c_cat_anim"] or passing:
        return "GO_CAT_ONLY", None, "c_cat_anim" if v.TS["c_cat_anim"] else passing[0]
    if not any(v.PC_star.values()):
        return "INCONCLUSIVE_DOSE", None, None
    return "PIVOT_NO_BASE_VALIDATED_CONTRAST", None, None


def test_decision_exhaustive_over_all_boolean_inputs():
    keys = (
        ["integrity_ok", "PC", "PC_pos", "R_t_cat"]
        + [f"TS_{k}" for k in ("dog", "wolf", "anim")]
        + [f"{a}_{x}" for a in ("A4", "A6", "A7") for x in CANDS]
        + [f"PC_star_{k}" for k in ("dog", "wolf", "anim")]
    )
    assert len(keys) == 16
    seen = set()
    for combo in itertools.product((False, True), repeat=len(keys)):
        inputs = _inputs(**dict(zip(keys, combo)))
        decision = decide(inputs)
        klass = decision["class"]
        assert klass in cr.DECISION_CLASSES
        assert decision["rank"] == cr.DECISION_CLASSES.index(klass) + 1
        seen.add(klass)
        ref = _reference_decide(inputs)
        assert (klass, decision.get("X"), decision.get("stage2a_primary")) == ref
        if klass == "GO_X":
            x = decision["X"]
            assert inputs.TS[f"c_cat_{x}"] and inputs.A4[x] and inputs.A6[x] and inputs.A7[x]
            assert inputs.integrity_ok and inputs.PC and inputs.R_t_cat
        if klass == "GO_CAT_ONLY":
            assert any(inputs.TS.values())
        if klass in ("INCONCLUSIVE_DOSE", "PIVOT_NO_BASE_VALIDATED_CONTRAST"):
            assert not any(inputs.TS.values())
    assert seen == set(cr.DECISION_CLASSES)


# --- off-target sets -------------------------------------------------------------------------------


def test_off_target_sets_match_spec():
    spec = FrozenPackage.from_repo(ROOT).spec["endpoint"]
    assert cr.O_PANEL == tuple(spec["O_panel"])
    assert cr.O_PRIME == tuple(spec["O_for_c_cat_anim"])
    assert off_target("dog") == tuple(w for w in spec["O_panel"] if w != "dog")
    assert off_target("c_cat_dog") == off_target("dog")
    assert off_target("c_cat_wolf") == tuple(w for w in spec["O_panel"] if w != "wolf")
    assert off_target("anim") == off_target("c_cat_anim") == tuple(spec["O_for_c_cat_anim"])
    assert off_target("PC") == cr.O_PANEL
    for key in ("dog", "wolf", "anim", "PC"):
        assert "cat" not in off_target(key)
    assert "dog" not in off_target("dog") and "wolf" not in off_target("c_cat_wolf")
    with pytest.raises(CriteriaError):
        off_target("lion")
    with pytest.raises(CriteriaError):
        off_target("c_cat_fox")


# --- WordScores ------------------------------------------------------------------------------------

WORDS = synthetic.WORDS


def _table(seed: int, shift: dict[str, float] | None = None) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    ids = [pid for fam in synthetic.synthetic_prompt_ids().values() for pid in fam]
    out = {}
    for pid in ids:
        row = np.log(np.full(len(WORDS), 0.05)) + rng.normal(0, 0.05, len(WORDS))
        for word, value in (shift or {}).items():
            row[WORDS.index(word)] += value
        out[pid] = row
    return out


def test_ell_per_prompt_and_target_guard():
    scores = WordScores(WORDS, {"x": _table(1)})
    others = off_target("dog")
    ell = scores.ell("x", "cat", others)
    pid = synthetic.synthetic_prompt_ids()["identity"][6]
    row = scores.values["x"][pid]
    expected = row[WORDS.index("cat")] - np.log(np.sum(np.exp([row[WORDS.index(o)] for o in others])))
    assert ell[pid] == pytest.approx(expected, abs=1e-12)
    with pytest.raises(CriteriaError, match="must not be in its off-target"):
        scores.ell("x", "dog", cr.O_PANEL)
    with pytest.raises(CriteriaError, match="Missing condition"):
        scores.ell("missing", "cat", others)


def test_wordscores_shape_guard():
    bad = {"direct_001": [0.0, 1.0]}
    with pytest.raises(CriteriaError, match="wrong shape"):
        WordScores(WORDS, {"x": bad}).L("x", "cat")


def _label_ctx(m_dog: float, m_wolf: float, plus: float = 1.0) -> cr.Context:
    plus_cid = cr.cid("c_cat_anim", sign=1)
    values = {
        "unsteered": _table(0),
        plus_cid: _table(1, {"cat": plus}),
        cr.cid("m_cat_dog", magnitude="tau:c_cat_anim"): _table(2, {"cat": m_dog}),
        cr.cid("m_cat_wolf", magnitude="tau:c_cat_anim"): _table(3, {"cat": m_wolf}),
    }
    family = st.FamilyIndex.from_ids(synthetic.synthetic_prompt_ids())
    return cr.Context(WordScores(WORDS, values), family, st.bootstrap_indices())


def test_label_c_cat_anim_requires_both_mentions():
    ctx = _label_ctx(0.2, 0.2)
    result = cr.label(ctx, "c_cat_anim")
    assert set(result["against"]) == {"m_cat_dog", "m_cat_wolf"}
    assert result["label"] == "PREFERENCE_CONTRAST"
    for m_dog, m_wolf in ((0.2, 1.5), (1.5, 0.2)):
        result = cr.label(_label_ctx(m_dog, m_wolf), "c_cat_anim")
        assert result["label"] == "LEXICAL_NOT_EXCLUDED"
        assert sorted(part["pass"] for part in result["against"].values()) == [False, True]


def test_label_single_mention_for_pair_contrast():
    plus_cid = cr.cid("c_cat_dog", sign=1)
    values = {"unsteered": _table(0), plus_cid: _table(1, {"cat": 1.0}),
              cr.cid("m_cat_dog", magnitude="tau:c_cat_dog"): _table(2, {"cat": 0.2})}
    family = st.FamilyIndex.from_ids(synthetic.synthetic_prompt_ids())
    ctx = cr.Context(WordScores(WORDS, values), family, st.bootstrap_indices())
    result = cr.label(ctx, "c_cat_dog")
    assert list(result["against"]) == ["m_cat_dog"]
    assert result["label"] == "PREFERENCE_CONTRAST"


def test_condition_id_helper():
    assert cr.cid("c_cat_dog") == "c_cat_dog|unit:tau:c_cat_dog|k=1|s=+1|slot=14|last"
    assert cr.cid("c_cat_dog", kappa=0.5, sign=-1) == "c_cat_dog|unit:tau:c_cat_dog|k=0.5|s=-1|slot=14|last"
    assert cr.cid("t_cat", scale="raw", mode="all") == "t_cat|raw:self|k=1|s=+1|slot=14|all"


def test_required_condition_ids_cover_gating_registry():
    names = synthetic.null_names_synthetic()
    required = cr.required_condition_ids(names)
    assert len(required) == 4157
    assert "unsteered" in required and "persona:P_default" in required


# --- end-to-end synthetic scenarios (each ~8 s) ----------------------------------------------------


@pytest.fixture(scope="module")
def scenario_table():
    return synthetic.scenarios()


def test_scenario_go_dog(scenario_table):
    scenario, expected, x = scenario_table["go_dog"]
    outcome = synthetic.run_scenario(scenario)
    assert outcome["decision"]["class"] == expected == "GO_X"
    assert outcome["decision"]["X"] == x == "dog"
    assert outcome["decision"]["stage2a_primary"] == "c_cat_anim"
    assert set(outcome["labels"]) == set(CONTRASTS)
    assert all(label["label"] == "PREFERENCE_CONTRAST" for label in outcome["labels"].values())
    crit = outcome["criteria"]
    # Every criterion is computed for every candidate, regardless of the decision path.
    assert set(crit["TS"]) == set(CONTRASTS) and set(crit["admissibility"]) == set(CANDS)
    for contrast in CONTRASTS:
        ts = crit["TS"][contrast]
        assert all(ts[part]["pass"] for part in "abcde")
        assert ts["d"]["rcov_n"] == 1000 and len(ts["d"]["null_statistics"]) == 240
    assert crit["PC"]["rcov_n"] == 200 and crit["PC_pos"]["rcov_n"] == 200
    assert outcome["alpha_TS"] == st.ALPHA_TS


def test_scenario_lexical_label(scenario_table):
    scenario, expected, x = scenario_table["lexical_label"]
    outcome = synthetic.run_scenario(scenario)
    assert outcome["decision"]["class"] == expected and outcome["decision"]["X"] == x
    assert outcome["labels"]["c_cat_dog"]["label"] == "LEXICAL_NOT_EXCLUDED"


def test_scenario_pivot_computes_all_criteria(scenario_table):
    scenario, expected, _ = scenario_table["pivot"]
    outcome = synthetic.run_scenario(scenario)
    assert outcome["decision"]["class"] == expected == "PIVOT_NO_BASE_VALIDATED_CONTRAST"
    assert outcome["labels"] == {}
    crit = outcome["criteria"]
    assert not any(crit["TS"][c]["pass"] for c in CONTRASTS)
    assert all(crit["PC_star"][k]["pass"] for k in ("dog", "wolf", "anim"))
    assert set(crit["admissibility"]["dog"]) == {"A1", "A2", "A3", "A4", "A5", "A6", "A7"}

"""v2 criteria, decision ladder, labels and fragility on synthetic planted-truth data (no model)."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.cts_stage0 import criteria as cr  # noqa: E402
from slgeo.cts_stage0 import fragility as fr  # noqa: E402
from slgeo.cts_stage0 import statistics as st  # noqa: E402
from slgeo.cts_stage0 import synthetic as sy  # noqa: E402
from slgeo.cts_stage0.conditions import registry_conditions  # noqa: E402
from slgeo.cts_stage0.contract import V2Contract  # noqa: E402
from slgeo.cts_stage0.package import FrozenPackage  # noqa: E402
from slgeo.cts_stage0.plan import null_names  # noqa: E402

RELIABLE = {**{c: 0.99 for c in sy.CONTRASTS}, **{f"{c}_perpG": 0.99 for c in sy.CONTRASTS}, "t_cat": 0.999}


@pytest.fixture(scope="module")
def contract() -> V2Contract:
    manifest = yaml.safe_load((ROOT / "configs" / "validation" / "cts_stage0_v2.yaml").read_text(encoding="utf-8"))
    return V2Contract.from_repo(ROOT, FrozenPackage.from_repo(ROOT), {k: manifest["contract"][k] for k in ("spec_sha256", "registry_sha256")})


@pytest.fixture(scope="module")
def rows(contract):
    return contract.rows


@pytest.fixture(scope="module")
def names(contract):
    return null_names(contract)


@pytest.fixture(scope="module")
def baselines(rows):
    return {c.cid: c.baseline for c in registry_conditions(rows)}


@pytest.fixture(scope="module")
def family():
    return st.FamilyIndex.from_ids(sy.synthetic_prompt_ids())


@pytest.fixture(scope="module")
def default_result(rows, names):
    return sy.run_scenario(rows, names, sy.default_scenario())


# --- decision ladder --------------------------------------------------------------------------------


def _inputs(**overrides):
    base = dict(
        integrity_ok=True, PC=True, PC_pos=True, R_t_cat=True,
        TS={c: True for c in sy.CONTRASTS}, TS_abcde={c: True for c in sy.CONTRASTS},
        A4={"dog": True, "wolf": True}, A6={"dog": True, "wolf": True}, A7={"dog": True, "wolf": True},
        PC_star={c: True for c in sy.CONTRASTS},
    )
    base.update(overrides)
    return cr.DecisionInputs(**base)


def test_ladder_is_ordered_disjoint_and_exhaustive():
    seen = set()
    for bits in itertools.product([False, True], repeat=10):
        ts = dict(zip(sy.CONTRASTS, bits[4:7]))
        abcde = {c: ts[c] or bits[7] for c in sy.CONTRASTS}
        inputs = _inputs(integrity_ok=bits[0], PC=bits[1], PC_pos=bits[2], R_t_cat=bits[3], TS=ts, TS_abcde=abcde,
                         A6={"dog": bits[8], "wolf": bits[8]}, PC_star={c: bits[9] for c in sy.CONTRASTS})
        decision = cr.decide(inputs)
        assert decision["class"] in cr.DECISION_CLASSES
        assert decision["rank"] == cr.DECISION_CLASSES.index(decision["class"]) + 1
        seen.add(decision["class"])
    assert seen == set(cr.DECISION_CLASSES)


def test_pivot_shared_leakage_only_after_dose_and_before_pivot():
    none = {c: False for c in sy.CONTRASTS}
    leak = dict(none, c_cat_dog=True)
    assert cr.decide(_inputs(TS=none, TS_abcde=leak))["class"] == "PIVOT_SHARED_LEAKAGE"
    assert cr.decide(_inputs(TS=none, TS_abcde=leak, PC_star=none))["class"] == "INCONCLUSIVE_DOSE"
    assert cr.decide(_inputs(TS=none, TS_abcde=none))["class"] == "PIVOT_NO_BASE_VALIDATED_CONTRAST"
    assert cr.decide(_inputs(TS=none, TS_abcde=leak))["contrasts_passing_a_to_e"] == ["c_cat_dog"]


def test_inconclusive_position_overrides_ts():
    assert cr.decide(_inputs(PC=False))["class"] == "INCONCLUSIVE_POSITION"
    assert cr.decide(_inputs(PC=False, PC_pos=False))["class"] == "STOP_INSTRUMENT"
    assert cr.decide(_inputs(R_t_cat=False))["class"] == "STOP_INSTRUMENT"
    assert cr.decide(_inputs(integrity_ok=False))["class"] == "TECHNICAL_FAIL"


# --- synthetic end-to-end ---------------------------------------------------------------------------


def test_synthetic_decision_suite(rows, names):
    result = sy.synthetic_decision_suite(rows, names)
    assert result["pass"], result["scenarios"]


def test_default_result_structure(default_result):
    ts = default_result["criteria"]["TS"]["c_cat_dog"]
    assert set(ts) >= {"a", "b", "c", "d1", "d2", "e", "f", "TS_abcde", "pass"}
    assert ts["d1"]["n"] == 240 and ts["d1"]["k_max"] == 3
    assert ts["d2"]["n"] == 199 and ts["d2"]["k_max"] == 2
    assert default_result["criteria"]["PC"]["reference_n"] == 99
    assert default_result["criteria"]["PC_pos"]["reference_n"] == 99
    assert set(default_result["criteria"]["PC_star"]) == set(sy.CONTRASTS)
    anim = default_result["criteria"]["TS"]["c_cat_anim"]
    assert anim["off_target"] == list(cr.O_PRIME) and set(anim["e"]) == {"kappa_0.5", "T3", "pass"}


def test_ts_f_failure_alone_blocks_ts(rows, names):
    scenario = sy.default_scenario()
    scenario["perp"]["c_cat_dog"] = 0.0
    result = sy.run_scenario(rows, names, scenario)
    ts = result["criteria"]["TS"]["c_cat_dog"]
    assert ts["TS_abcde"] and not ts["f"]["pass"] and not ts["pass"]
    assert result["decision"]["class"] == "GO_X" and result["decision"]["X"] == "wolf"


def test_perp_reliability_below_threshold_fails_f(rows, names):
    reliabilities = dict(RELIABLE, c_cat_dog_perpG=0.9)
    result = sy.run_scenario(rows, names, sy.default_scenario(), reliabilities)
    assert not result["criteria"]["TS"]["c_cat_dog"]["f"]["pass"]


def test_labels_only_for_passing_contrasts(rows, names):
    scenario = sy.default_scenario()
    scenario["ts"]["c_cat_wolf"] = 0.0
    result = sy.run_scenario(rows, names, scenario)
    assert "c_cat_wolf" not in result["labels"] and "c_cat_dog" in result["labels"]


def test_shared_label_margin_is_relative(rows, names):
    """r = 0.2: a g whose cat selectivity is a third of c's is not shared; a tenth is."""
    for selective, label in ((1.0 / 3.0, "NOT_BASE_SHARED"), (0.1, "BASE_SHARED_DIRECTION")):
        scenario = sy.default_scenario()
        scenario["shared"]["g_id"]["cat_selective"] = selective
        result = sy.run_scenario(rows, names, scenario)
        assert result["labels"]["c_cat_dog"]["shared"]["g_id"]["label"] == label


def test_shared_label_requires_mass_increase(rows, names):
    scenario = sy.default_scenario()
    scenario["shared"]["g_tmpl"]["mass"] = 0.0
    result = sy.run_scenario(rows, names, scenario)
    shared = result["labels"]["c_cat_dog"]["shared"]["g_tmpl"]
    assert not shared["i_shared_effect"]["pass"] and shared["label"] == "NOT_BASE_SHARED"


def test_prompt_order_does_not_change_the_evaluation(rows, names, baselines, family):
    scores = sy.synthetic_scores(rows, sy.default_scenario())
    shuffled = {cid: dict(sorted(table.items(), reverse=True)) for cid, table in scores.values.items()}
    a, _ = cr.evaluate(scores, family, RELIABLE, names, True, baselines)
    b, _ = cr.evaluate(cr.WordScores(scores.words, shuffled), family, RELIABLE, names, True, baselines)
    assert a == b


def test_missing_condition_raises(rows, names, baselines, family):
    scores = sy.synthetic_scores(rows, sy.default_scenario())
    values = dict(scores.values)
    values.pop(next(cid for cid in values if cid.startswith("rcov:5|")))
    with pytest.raises(cr.CriteriaError):
        cr.evaluate(cr.WordScores(scores.words, values), family, RELIABLE, names, True, baselines)


def test_nan_raises(rows, names, baselines, family):
    scores = sy.synthetic_scores(rows, sy.default_scenario())
    cid = "c_cat_dog|unit:tau:c_cat_dog|k=1|s=+1|slot=14|last"
    table = dict(scores.values[cid])
    first = next(iter(table))
    table[first] = table[first].copy()
    table[first][0] = np.nan
    values = dict(scores.values, **{cid: table})
    with pytest.raises(st.StatisticsError):
        cr.evaluate(cr.WordScores(scores.words, values), family, RELIABLE, names, True, baselines)


def test_baseline_mapping_is_strict(rows, names, family):
    scores = sy.synthetic_scores(rows, sy.default_scenario())
    with pytest.raises(cr.CriteriaError):
        cr.evaluate(scores, family, RELIABLE, names, True, {})


def test_own_prefix_conditions_use_own_baseline(baselines):
    assert baselines["t_cat|raw:self|k=1|s=+1|slot=14|all"] == "persona:P_default"
    assert baselines["rcov:0|unit:norm:t_cat|k=1|s=+1|slot=14|all"] == "persona:P_default"
    assert baselines["t_cat|raw:self|k=1|s=+1|slot=14|last"] == "unsteered"
    assert baselines["persona:P_dog_T1"] == "persona:P_default"


def test_off_target_sets_never_contain_the_target():
    for contrast in sy.CONTRASTS:
        assert "cat" not in cr.off_target(contrast)
    assert "dog" not in cr.off_target("c_cat_dog") and "wolf" not in cr.off_target("c_cat_wolf")
    assert cr.shared_label_pairs("c_cat_dog") == cr.O_PANEL and cr.shared_label_pairs("c_cat_anim") == cr.O_PRIME


# --- fragility ---------------------------------------------------------------------------------------


def test_fragility_suite(rows, names):
    assert sy.synthetic_fragility_suite(rows, names)["pass"]


def test_fragility_statistic_set_is_outcome_independent(rows, names, baselines, family):
    """The checked statistics (and m) are the same whether or not contrasts pass."""
    labels = []
    for scenario in (sy.default_scenario(), dict(sy.default_scenario(), ts={c: 0.0 for c in sy.CONTRASTS})):
        ctx = cr.all_statistics(sy.synthetic_scores(rows, scenario), family, RELIABLE, names, baselines)
        labels.append(set(ctx.recorded))
    assert labels[0] == labels[1]


def test_fragility_counts_and_guard(rows, names, baselines, family):
    conditions = registry_conditions(rows)
    rescored = {c.cid for c in conditions if c.reference_rescore}
    scores = sy.synthetic_scores(rows, sy.default_scenario())
    ctx = cr.all_statistics(scores, family, RELIABLE, names, baselines)
    l1 = cr.WordScores(scores.words, {cid: scores.values[cid] for cid in rescored})
    report = fr.fragility_report(ctx, l1, rescored)
    assert report["pass"] and report["rms_z"] == 0.0
    assert report["m_null_conditions"] == 70 and set(report["null_members_per_family"].values()) == {10}
    assert report["m"] == report["m_ci_and_sign_statistics"] + 70
    assert 250 < report["m"] < 450
    assert report["max_guard"] == pytest.approx(0.04 * 3.78, abs=0.01)


def test_fragility_refuses_a_missing_rescore(rows, names, baselines, family):
    conditions = registry_conditions(rows)
    rescored = {c.cid for c in conditions if c.reference_rescore}
    scores = sy.synthetic_scores(rows, sy.default_scenario())
    ctx = cr.all_statistics(scores, family, RELIABLE, names, baselines)
    fewer = set(rescored)
    fewer.remove(next(cid for cid in sorted(fewer) if cid.startswith("null:")))
    with pytest.raises(fr.FragilityError):
        fr.fragility_report(ctx, cr.WordScores(scores.words, {cid: scores.values[cid] for cid in fewer}), fewer)


def test_guard_formula():
    assert fr.guard(321) == pytest.approx(0.04 * 3.782, abs=1e-3)
    with pytest.raises(fr.FragilityError):
        fr.guard(0)

"""Red-team negative tests: statistics, criteria, decision engine, conditions, plan and batching gate.

Uses the planted-truth synthetic scores of ``synthetic.py`` only (no model, no real scores).
"""

from __future__ import annotations

import itertools
import json
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.cts_stage0 import analysis, artifacts as art, criteria, plan as plan_mod, statistics as st, synthetic  # noqa: E402
from slgeo.cts_stage0.conditions import Condition, ConditionError, resolve_vector, vector_sha256  # noqa: E402
from slgeo.cts_stage0.criteria import CriteriaError, DecisionInputs, WordScores, decide, off_target  # noqa: E402
from slgeo.cts_stage0.directions import (  # noqa: E402
    AxisStatistics,
    Direction,
    DirectionBundle,
    DirectionError,
    Formula,
    build_direction,
    persona_formulas,
)
from slgeo.cts_stage0.statistics import StatisticsError  # noqa: E402

_CACHE: dict = {}


def _family() -> st.FamilyIndex:
    return st.FamilyIndex.from_ids(synthetic.synthetic_prompt_ids())


def _reference(name: str = "go_dog") -> dict:
    if name not in _CACHE:
        scenario = synthetic.scenarios()[name][0]
        _CACHE[name] = synthetic.run_scenario(scenario)
    return _CACHE[name]


# ------------------------------------------------------------------------------------ 7. outcome canary


def test_synthetic_run_prints_nothing(capsys):
    result = synthetic.run_scenario(synthetic.default_scenario())
    captured = capsys.readouterr()
    assert captured.out == ""
    assert not re.search(r"\d", captured.err), "numeric output on stderr"
    _CACHE["go_dog"] = result
    assert result["decision"]["class"] == "GO_X"


def test_analysis_files_and_handoff_shape_are_class_independent():
    go, pivot = _reference("go_dog"), _reference("pivot")
    assert go["decision"]["class"] == "GO_X"
    assert pivot["decision"]["class"] == "PIVOT_NO_BASE_VALIDATED_CONTRAST"
    assert isinstance(analysis.OUTPUT_FILES, tuple) and len(analysis.OUTPUT_FILES) == 5

    class FakeBundle:
        def get(self, name):
            return SimpleNamespace(unit=np.zeros(4))

    metas = [analysis.stage2a_handoff(FakeBundle(), result)[0] for result in (go, pivot)]
    assert set(metas[0]) == set(metas[1])
    for result in (go, pivot):
        art.pretty_json(result)  # finite and serializable for every class (allow_nan=False)
        assert set(result) == set(go)
        assert set(result["criteria"]) == set(go["criteria"])  # every criterion computed on every path


def test_decide_is_ordered_disjoint_and_exhaustive():
    seen = set()
    for bits in itertools.product((False, True), repeat=16):
        inputs = DecisionInputs(
            integrity_ok=bits[0], PC=bits[1], PC_pos=bits[2], R_t_cat=bits[3],
            TS={"c_cat_dog": bits[4], "c_cat_wolf": bits[5], "c_cat_anim": bits[6]},
            A4={"dog": bits[7], "wolf": bits[8]}, A6={"dog": bits[9], "wolf": bits[10]}, A7={"dog": bits[11], "wolf": bits[12]},
            PC_star={"dog": bits[13], "wolf": bits[14], "anim": bits[15]},
        )
        decision = decide(inputs)
        assert decision["class"] in criteria.DECISION_CLASSES
        assert criteria.DECISION_CLASSES[decision["rank"] - 1] == decision["class"]
        if not bits[0]:
            assert decision["class"] == "TECHNICAL_FAIL"
        seen.add(decision["class"])
    assert seen == set(criteria.DECISION_CLASSES)


# ------------------------------------------------------------------------------------- 8. non-finite


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_statistics_raise_on_non_finite(bad):
    family = _family()
    good = {pid: 0.1 for ids in family.ids.values() for pid in ids}
    values = dict(good)
    values[family.ids["identity"][5]] = bad
    with pytest.raises(StatisticsError):
        family.align(values)
    y = {f: np.zeros(100) for f in st.FAMILIES}
    y["direct"][3] = bad
    index = st.bootstrap_indices()
    with pytest.raises(StatisticsError):
        st.replicates(y, index)
    with pytest.raises(StatisticsError):
        st.interval(y, index)
    with pytest.raises(StatisticsError):
        st.mc_p_value(bad, [0.0, 1.0])
    with pytest.raises(StatisticsError):
        st.mc_p_value(0.5, [0.0, bad])
    with pytest.raises(StatisticsError):
        st.null_threshold([0.0, bad, 1.0])


def test_statistics_refuse_malformed_inputs():
    family = _family()
    values = {pid: 0.1 for ids in family.ids.values() for pid in ids}
    del values[family.ids["direct"][0]]
    with pytest.raises(StatisticsError, match="Missing prompt"):
        family.align(values)
    ids = synthetic.synthetic_prompt_ids()
    ids["hypothetical"] = ids["hypothetical"][:99]
    with pytest.raises(StatisticsError):
        st.FamilyIndex.from_ids(ids)
    ids = synthetic.synthetic_prompt_ids()
    ids["direct"] = ids["direct"][:99] + ids["direct"][:1]
    with pytest.raises(StatisticsError):
        st.FamilyIndex.from_ids(ids)
    with pytest.raises(StatisticsError):
        st.mc_p_value(0.0, [])
    with pytest.raises(StatisticsError):
        st.random_cov_directions(np.zeros((1023, 8)))


def test_point_raises_on_nan():
    y = {f: np.zeros(100) for f in st.FAMILIES}
    y["direct"][0] = np.nan
    with pytest.raises(StatisticsError):
        st.point(y)


def test_random_cov_directions_and_covariance_raise_on_nan():
    states = np.random.default_rng(0).standard_normal((1024, 8))
    states[3, 2] = np.nan
    with pytest.raises(StatisticsError):
        st.random_cov_directions(states, n=4)
    with pytest.raises(StatisticsError):
        st.covariance(states)


def _poisoned(cid: str, word: str, prompt_index: int = 7, value=np.nan) -> WordScores:
    scores = synthetic.synthetic_scores(synthetic.default_scenario())
    values = {key: dict(table) for key, table in scores.values.items()}
    pid = sorted(values[cid])[prompt_index]
    row = np.array(values[cid][pid], dtype=np.float64)
    row[scores.index[word]] = value
    values[cid][pid] = row
    return WordScores(scores.words, values)


@pytest.mark.parametrize(
    "cid, word",
    [
        (criteria.cid("c_cat_dog", sign=1), "cat"),
        (criteria.cid("c_cat_dog", sign=1), "lion"),  # off-target word inside logsumexp
        ("unsteered", "cat"),  # baseline row
    ],
)
def test_nan_row_raises_through_context_delta(cid, word):
    scores = _poisoned(cid, word)
    ctx = criteria.Context(scores, _family(), st.bootstrap_indices())
    with pytest.raises(StatisticsError):
        ctx.delta(criteria.cid("c_cat_dog", sign=1), "cat", off_target("c_cat_dog"))


def test_nan_row_raises_through_evaluate():
    scores = _poisoned(criteria.cid("c_cat_dog", sign=1), "cat", value=-np.inf)
    reliabilities = {"c_cat_dog": 0.99, "c_cat_wolf": 0.99, "c_cat_anim": 0.99, "t_cat": 0.999}
    with pytest.raises(StatisticsError):
        criteria.evaluate(scores, _family(), reliabilities, synthetic.null_names_synthetic(), integrity_ok=True)


def test_missing_gating_condition_raises():
    scores = synthetic.synthetic_scores(synthetic.default_scenario())
    values = dict(scores.values)
    del values[criteria.cid("rcov:17", magnitude="tau:c_cat_wolf")]
    reliabilities = {"c_cat_dog": 0.99, "c_cat_wolf": 0.99, "c_cat_anim": 0.99, "t_cat": 0.999}
    with pytest.raises(CriteriaError, match="Missing condition"):
        criteria.evaluate(WordScores(scores.words, values), _family(), reliabilities, synthetic.null_names_synthetic(), True)


def test_off_target_sets_never_contain_the_target():
    for contrast in ("c_cat_dog", "c_cat_wolf", "c_cat_anim", "panel"):
        others = off_target(contrast)
        assert "cat" not in others
    assert "dog" not in off_target("c_cat_dog") and "wolf" not in off_target("c_cat_wolf")
    assert off_target("c_cat_anim") == criteria.O_PRIME
    with pytest.raises(CriteriaError):
        off_target("c_cat_lion")
    scores = synthetic.synthetic_scores(synthetic.default_scenario())
    with pytest.raises(CriteriaError, match="must not be in its off-target set"):
        scores.ell("unsteered", "cat", ("cat",) + criteria.O_PANEL)


# -------------------------------------------------------------------------------- 9. order invariance


def test_prompt_order_shuffle_gives_identical_evaluation():
    reference = _reference("go_dog")
    scores = synthetic.synthetic_scores(synthetic.default_scenario())
    rng = random.Random(4)
    shuffled = {}
    for cid in rng.sample(sorted(scores.values), len(scores.values)):
        items = list(scores.values[cid].items())
        rng.shuffle(items)
        shuffled[cid] = dict(items)
    ids = synthetic.synthetic_prompt_ids()
    family = st.FamilyIndex.from_ids({f: rng.sample(v, len(v)) for f, v in ids.items()})
    reliabilities = {"c_cat_dog": 0.99, "c_cat_wolf": 0.99, "c_cat_anim": 0.99, "t_cat": 0.999}
    result = criteria.evaluate(WordScores(scores.words, shuffled), family, reliabilities, synthetic.null_names_synthetic(), True)
    assert result["decision"] == reference["decision"]
    assert result["labels"] == reference["labels"]
    for contrast in ("c_cat_dog", "c_cat_anim"):
        assert result["criteria"]["TS"][contrast]["b"] == reference["criteria"]["TS"][contrast]["b"]
        assert result["criteria"]["TS"][contrast]["d"]["rcov_p"] == reference["criteria"]["TS"][contrast]["d"]["rcov_p"]
    assert result["criteria"]["PC"] == reference["criteria"]["PC"]
    assert art.canonical_json(result) == art.canonical_json(reference)


def test_bootstrap_streams_match_frozen_goldens():
    from slgeo.cts_stage0.preflight import rng_golden_check

    assert rng_golden_check()["pass"] is True


# ------------------------------------------------------------------------- directions and conditions


def _stats(axes: dict[str, np.ndarray], hidden: int = 6, drift: np.ndarray | None = None) -> AxisStatistics:
    """Personas with exact half means: default halves differ by ``drift``; persona halves share it."""
    base = np.zeros((29, hidden))
    drift = np.zeros(hidden) if drift is None else drift
    half_default = np.stack([base - drift, base + drift])
    full, half = {"P_default": half_default.mean(axis=0)}, {"P_default": half_default}
    for persona, axis in axes.items():
        h = np.stack([half_default[0] + axis, half_default[1] + axis])
        half[persona] = h
        full[persona] = h.mean(axis=0)
    return AxisStatistics(full, half)


def test_contrast_sign_and_same_half_default():
    e = np.eye(6)
    stats = _stats({"P_cat_T1": np.tile(e[0], (29, 1)), "P_dog_T1": np.tile(e[1], (29, 1))}, drift=np.tile(3 * e[2], (29, 1)))
    formula = next(f for f in persona_formulas() if f.name == "c_cat_dog")
    direction = build_direction(stats, formula)
    assert np.allclose(direction.raw, e[0] - e[1])  # t_cat - t_dog, never t_dog - t_cat
    assert direction.tau == pytest.approx(1 / np.sqrt(2))
    # Same-half default subtraction removes the planted half drift exactly (full-mean subtraction would not).
    assert direction.reliability == pytest.approx(1.0, abs=1e-12)


def test_zero_direction_is_refused():
    stats = _stats({"P_cat_T1": np.zeros((29, 6))})
    with pytest.raises(DirectionError, match="zero"):
        build_direction(stats, Formula("t_cat", {"P_cat_T1": 1.0}))
    with pytest.raises(DirectionError, match="missing"):
        build_direction(stats, Formula("t_dog", {"P_dog_T1": 1.0}))


def test_paraphrase_tau_uses_own_template():
    formulas = {f.name: f for f in persona_formulas()}
    assert formulas["c_cat_dog_T2"].tau_ref == "P_cat_T2"
    assert formulas["c_cat_anim_T3"].tau_ref == "P_cat_T3"
    assert formulas["c_cat_dog"].tau_ref == "P_cat_T1"
    assert set(formulas["c_cat_anim"].coefficients) == {"P_cat_T1"} | {f"P_{x}_T1" for x in ("dog", "wolf", "lion", "horse", "rabbit", "elephant")}


def _bundle() -> DirectionBundle:
    unit = np.array([0.6, 0.8, 0.0])
    directions = {
        "c_cat_dog": Direction("c_cat_dog", 14, unit * 5, unit, -2.0, 0.99, True, "P_cat_T1"),
        "t_cat": Direction("t_cat", 14, np.array([3.0, 0.0, 4.0]), np.array([0.6, 0.0, 0.8]), 5.0, 0.99, True, "P_cat_T1"),
        "null:a>b": Direction("null:a>b", 14, np.array([0.0, 0.1, 0.0]), np.array([0.0, 1.0, 0.0]), 0.01, 0.5, False, "P_cat_T1"),
        "c_cat_dog@8": Direction("c_cat_dog@8", 8, unit, unit, 1.0, 0.9, False, "P_cat_T1"),
    }
    r = np.eye(3)
    return DirectionBundle(directions, r_cov=r, r_iso=r, null_names=["null:a>b"])


def _steer(direction, **kwargs):
    defaults = dict(kind="steer", group="g", gating=True, direction=direction, magnitude=f"tau:{direction}")
    defaults.update(kwargs)
    return Condition(**defaults)


def test_vector_magnitude_uses_abs_tau_and_provenance_hash_tracks_sign():
    bundle = _bundle()
    plus, minus = _steer("c_cat_dog", sign=1), _steer("c_cat_dog", sign=-1, kappa=0.5)
    v_plus, v_minus = resolve_vector(plus, bundle), resolve_vector(minus, bundle)
    assert np.allclose(v_plus, 2.0 * np.array([0.6, 0.8, 0.0]))  # |tau| = 2, direction kept (tau < 0)
    assert np.linalg.norm(v_minus) == pytest.approx(1.0)
    assert np.allclose(v_minus, -0.5 * v_plus)
    assert vector_sha256(plus, bundle) != vector_sha256(minus, bundle)
    null = _steer("null:a>b", magnitude="tau:c_cat_dog")
    assert np.linalg.norm(resolve_vector(null, bundle)) == pytest.approx(2.0)  # dosed by the contrast, not its own tau
    pc = _steer("rcov:1", magnitude="norm:t_cat")
    assert np.linalg.norm(resolve_vector(pc, bundle)) == pytest.approx(5.0)
    raw = _steer("t_cat", scale="raw")
    assert np.array_equal(resolve_vector(raw, bundle), np.array([3.0, 0.0, 4.0]))


@pytest.mark.parametrize(
    "condition",
    [
        _steer("t_cat", scale="raw", sign=-1),
        _steer("t_cat", scale="raw", kappa=0.5),
        _steer("c_cat_dog", kappa=0.0),
        _steer("c_cat_dog", sign=2),
        _steer("rcov:1", slot=8),
        _steer("c_cat_dog", slot=8),
        _steer("c_cat_dog@8", slot=14),
        _steer("c_cat_dog", magnitude="scale:c_cat_dog"),
        Condition(kind="persona", group="persona", gating=True, persona="P_dog_T1"),
    ],
    ids=["raw_negative", "raw_kappa", "kappa0", "sign2", "rcov_slot8", "slot_mismatch", "slot_mismatch_rev", "bad_magnitude",
         "persona"],
)
def test_malformed_conditions_are_refused(condition):
    with pytest.raises((ConditionError, DirectionError)):
        resolve_vector(condition, _bundle())


def test_zero_magnitude_is_refused():
    bundle = _bundle()
    bundle.directions["c_cat_dog"].tau = 0.0
    with pytest.raises(DirectionError, match="non-positive magnitude"):
        resolve_vector(_steer("c_cat_dog"), bundle)


# --------------------------------------------------------------------------------------- 10. plan


@pytest.fixture(scope="module")
def frozen():
    import yaml

    from slgeo.cts_stage0.package import FrozenPackage

    manifest = yaml.safe_load((ROOT / "configs" / "validation" / "cts_stage0_v1.yaml").read_text(encoding="utf-8"))
    package = FrozenPackage.from_repo(ROOT)
    return package, plan_mod.load_choices(ROOT, manifest), manifest


@pytest.mark.parametrize("per_shard", [0, -1])
def test_plan_refuses_non_positive_conditions_per_shard(frozen, per_shard):
    package, choices, manifest = frozen
    with pytest.raises(plan_mod.PlanError):
        plan_mod.build_plan(package, choices, manifest, conditions_per_shard=per_shard)


def test_plan_covers_registry_and_detects_unscored_condition(frozen, monkeypatch):
    package, choices, manifest = frozen
    plan = plan_mod.build_plan(package, choices, manifest, conditions_per_shard=500)
    scored = [cid for shard in plan["shards"] if shard["stage"] in ("score", "score_persona") for cid in shard["payload"]["conditions"]]
    assert len(scored) == len(set(scored)) == plan["n_conditions"] - 1
    assert criteria.required_condition_ids(plan_mod.null_names(package)) <= set(scored) | {"unsteered"}

    real = plan_mod.conditions_for

    def with_orphan(pkg, ch):
        return real(pkg, ch) + [Condition(kind="orphan", group="x", gating=False, direction="t_cat")]

    monkeypatch.setattr(plan_mod, "conditions_for", with_orphan)
    with pytest.raises(plan_mod.PlanError, match="cover the condition registry"):
        plan_mod.build_plan(package, choices, manifest, conditions_per_shard=500)


@pytest.mark.parametrize("which", ["gating", "descriptive"])
def test_plan_detects_condition_removed_from_registry(frozen, monkeypatch, which):
    package, choices, manifest = frozen
    real = plan_mod.conditions_for

    def without_one(pkg, ch):
        conditions = real(pkg, ch)
        victim = next(c for c in conditions if c.kind == "steer" and c.gating == (which == "gating"))
        return [c for c in conditions if c is not victim]

    monkeypatch.setattr(plan_mod, "conditions_for", without_one)
    with pytest.raises(plan_mod.PlanError):
        plan_mod.build_plan(package, choices, manifest, conditions_per_shard=500)


# ------------------------------------------------------------------------------ 11. batching gate


def _ctx(rows, record=None):
    return SimpleNamespace(manifest={"scoring": {"condition_batch_rows": rows}}, run_record=dict(record or {}))


def test_batch_rows_gate():
    from slgeo.cts_stage0.pipeline import PipelineError, _batch_rows

    assert _batch_rows(_ctx(1)) == 1
    for rows in (2, 8, 32):
        with pytest.raises(PipelineError, match="equivalence"):
            _batch_rows(_ctx(rows))
        with pytest.raises(PipelineError, match="equivalence"):
            _batch_rows(_ctx(rows, {"equivalence_record_sha256": ""}))
    for rows in (0, -3):
        with pytest.raises(PipelineError):
            _batch_rows(_ctx(rows))
    # A bare hash is no longer sufficient: the recorded file must exist, match and have passed §13.3.
    with pytest.raises(PipelineError):
        _batch_rows(_ctx(8, {"equivalence_record_sha256": "a" * 64}))


def test_batch_rows_gate_checks_record_content():
    from slgeo.cts_stage0.pipeline import PipelineError, _batch_rows

    with pytest.raises(PipelineError):
        _batch_rows(_ctx(16, {"equivalence_record_sha256": "0" * 64}))

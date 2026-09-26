from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.cts_stage0 import conditions as cd
from slgeo.cts_stage0 import directions as dr
from slgeo.cts_stage0 import plan as pl
from slgeo.cts_stage0.artifacts import canonical_json
from slgeo.cts_stage0.conditions import Condition, ConditionError, resolve_vector
from slgeo.cts_stage0.directions import AxisStatistics, DirectionError
from slgeo.cts_stage0.package import FrozenPackage

H = 6
SLOTS = 29


@pytest.fixture(scope="module")
def package() -> FrozenPackage:
    return FrozenPackage.from_repo(ROOT)


def _half_means(personas, seed=0, drift_scale=5.0):
    """mu_h(P) = base + offset_P + drift_h: the half drift is shared by every persona (row-set effect)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(SLOTS, H)) * 10
    drift = rng.normal(size=(2, SLOTS, H)) * drift_scale
    offsets = {p: rng.normal(size=(SLOTS, H)) for p in personas}
    offsets["P_default"] = np.zeros((SLOTS, H))
    return {p: np.stack([base + offsets[p] + drift[h] for h in (0, 1)]) for p in personas}


def _stats(means) -> AxisStatistics:
    return AxisStatistics.from_half_sums({p: m * 512 for p, m in means.items()})


@pytest.fixture(scope="module")
def personas(package):
    return sorted(package.personas)


@pytest.fixture(scope="module")
def means(personas):
    return _half_means(personas)


@pytest.fixture(scope="module")
def stats(means):
    return _stats(means)


@pytest.fixture(scope="module")
def bundle(package, stats):
    rng = np.random.default_rng(3)
    states = rng.normal(size=(1024, H))
    embeddings = {w: rng.normal(size=H) for w in ("cat", "dog", "wolf")}
    return dr.build_bundle(stats, states, pl.null_words(package), embeddings)


def _formula(name):
    return {f.name: f for f in dr.persona_formulas()}[name]


# --- axis statistics -------------------------------------------------------------------------------


def test_from_half_sums_means(means, stats):
    p = "P_cat_T1"
    np.testing.assert_allclose(stats.half[p], means[p], atol=1e-12)
    np.testing.assert_allclose(stats.full[p], (means[p][0] + means[p][1]) / 2, atol=1e-12)
    with pytest.raises(DirectionError):
        AxisStatistics.from_half_sums({"P_default": np.zeros((3, SLOTS, H))})


def test_frozen_personas_cover_every_formula(personas):
    names = set(personas)
    for formula in dr.persona_formulas():
        assert set(formula.coefficients) <= names, formula.name
        assert formula.tau_ref in names


def test_c_cat_dog_is_unit_difference(bundle, stats):
    direction = bundle.get("c_cat_dog")
    diff = stats.axis("P_cat_T1", 14) - stats.axis("P_dog_T1", 14)
    np.testing.assert_allclose(direction.raw, diff, atol=1e-12)
    np.testing.assert_allclose(direction.unit, diff / np.linalg.norm(diff), atol=1e-12)
    assert float(direction.unit @ diff) > 0
    assert direction.slot == 14 and direction.gating_reliability
    assert direction.tau == pytest.approx(float(stats.axis("P_cat_T1", 14) @ direction.unit), abs=1e-12)


def test_half_axes_subtract_same_half_default(bundle, means):
    # The shared half drift cancels exactly only if each half axis subtracts the same-half default mean.
    for name in ("t_cat", "c_cat_dog", "c_cat_anim", "g_id"):
        assert bundle.get(name).reliability == pytest.approx(1.0, abs=1e-12), name
    stats = _stats(means)
    expected = means["P_cat_T1"][0, 14] - means["P_default"][0, 14]
    np.testing.assert_allclose(stats.half_axis("P_cat_T1", 14, 0), expected, atol=1e-12)
    # Control: subtracting the full default mean would not reproduce R = 1 under the drift.
    wrong = [means["P_cat_T1"][h, 14] - stats.full["P_default"][14] for h in (0, 1)]
    assert dr.cosine(*wrong) < 0.999


def test_c_cat_anim_uses_seven_personas(personas, means):
    formula = _formula("c_cat_anim")
    assert set(formula.coefficients) == {"P_cat_T1"} | {f"P_{x}_T1" for x in dr.A_LEN}
    assert len(formula.coefficients) == 7
    assert formula.coefficients["P_cat_T1"] == 1.0
    assert all(formula.coefficients[f"P_{x}_T1"] == pytest.approx(-1 / 6) for x in dr.A_LEN)
    stats = _stats(means)
    raw = dr.build_direction(stats, formula).raw
    expected = stats.axis("P_cat_T1", 14) - np.mean([stats.axis(f"P_{x}_T1", 14) for x in dr.A_LEN], axis=0)
    np.testing.assert_allclose(raw, expected, atol=1e-12)
    # A half-specific perturbation of lion enters R(c_cat_anim) only; fox is not in the formula.
    for persona, affects in (("P_lion_T1", True), ("P_fox_T1", False)):
        perturbed = {p: m.copy() for p, m in means.items()}
        perturbed[persona][0, 14] += np.random.default_rng(5).normal(size=H) * 3
        s = _stats(perturbed)
        r_anim = dr.build_direction(s, formula).reliability
        r_dog = dr.build_direction(s, _formula("c_cat_dog")).reliability
        assert r_dog == pytest.approx(1.0, abs=1e-12)
        assert (r_anim < 0.9999) is affects


def test_paraphrase_tau_uses_own_template(bundle, stats):
    for name, ref in (("c_cat_dog_T2", "P_cat_T2"), ("c_cat_wolf_T3", "P_cat_T3"), ("c_cat_anim_T3", "P_cat_T3")):
        direction = bundle.get(name)
        assert direction.tau_ref == ref
        assert direction.tau == pytest.approx(float(stats.axis(ref, 14) @ direction.unit), abs=1e-12)
        assert direction.tau != pytest.approx(float(stats.axis("P_cat_T1", 14) @ direction.unit), abs=1e-9)
    anim_t3 = _formula("c_cat_anim_T3")
    assert set(anim_t3.coefficients) == {"P_cat_T3"} | {f"P_{x}_T3" for x in dr.A_LEN}
    assert "c_cat_anim_T2" not in bundle.directions


def test_mention_and_secondary_directions(bundle, stats):
    mention = bundle.get("m_cat_dog")
    diff = stats.axis("M_cat", 14) - stats.axis("M_dog", 14)
    np.testing.assert_allclose(mention.unit, diff / np.linalg.norm(diff), atol=1e-12)
    secondary = bundle.get("c_cat_dog@8")
    diff8 = stats.axis("P_cat_T1", 8) - stats.axis("P_dog_T1", 8)
    assert secondary.slot == 8
    np.testing.assert_allclose(secondary.unit, diff8 / np.linalg.norm(diff8), atol=1e-12)
    assert secondary.tau == pytest.approx(float(stats.axis("P_cat_T1", 8) @ secondary.unit), abs=1e-12)


def test_zero_norm_direction_raises(personas, means):
    same = {p: m.copy() for p, m in means.items()}
    same["P_dog_T1"] = same["P_cat_T1"].copy()
    with pytest.raises(DirectionError, match="zero or non-finite norm"):
        dr.build_direction(_stats(same), _formula("c_cat_dog"))


def test_missing_persona_raises(means):
    partial = {p: m for p, m in means.items() if p != "P_wolf_T1"}
    with pytest.raises(DirectionError, match="missing personas"):
        dr.build_direction(_stats(partial), _formula("c_cat_wolf"))


def test_null_formulas_240_ordered_pairs(package):
    words = pl.null_words(package)
    assert len(words) == 16 and len(set(words)) == 16
    assert "mouse" in words and "sheep" in words
    formulas = dr.null_formulas(words)
    assert len(formulas) == 240
    names = [f.name for f in formulas]
    assert len(set(names)) == 240 and names == pl.null_names(package)
    a, b = words[0], words[1]
    assert f"null:{a}>{b}" in names and f"null:{b}>{a}" in names and f"null:{a}>{a}" not in names
    first = formulas[0]
    assert first.coefficients == {f"N_{a}_T1": 1.0, f"N_{b}_T1": -1.0}
    assert all(set(f.coefficients) <= set(package.personas) for f in formulas)


def test_bundle_contents(bundle, package):
    assert bundle.r_cov.shape == (1000, H) and bundle.r_iso.shape == (1000, H)
    assert len(bundle.null_names) == 240
    with pytest.raises(DirectionError):
        bundle.get("nope")
    embedding = bundle.get("e_cat_dog")
    assert embedding.reliability is None and not embedding.gating_reliability


# --- resolve_vector --------------------------------------------------------------------------------


def _gating(package):
    return {c.cid: c for c in cd.gating_conditions(pl.null_names(package))}


def test_null_and_rcov_conditions_norm_matched(bundle, package):
    conditions = cd.gating_conditions(pl.null_names(package))
    for contrast in cd.TESTED_CONTRASTS:
        target = abs(bundle.get(contrast).tau)
        nulls = [c for c in conditions if c.group == f"null:{contrast}"]
        rcov = [c for c in conditions if c.group == f"rcov:{contrast}"]
        assert len(nulls) == 240 and len(rcov) == 1000
        for condition in nulls[:20] + rcov[:20]:
            assert np.linalg.norm(resolve_vector(condition, bundle)) == pytest.approx(target, rel=1e-12)
        np.testing.assert_allclose(resolve_vector(rcov[7], bundle), target * bundle.r_cov[7], rtol=1e-12)
    pc = [c for c in conditions if c.group.startswith("rcov_pc:")]
    assert len(pc) == 400
    for condition in pc[:5] + pc[-5:]:
        assert np.linalg.norm(resolve_vector(condition, bundle)) == pytest.approx(bundle.get("t_cat").norm, rel=1e-12)


def test_positive_control_is_raw_t_cat(bundle, stats, package):
    gating = _gating(package)
    for mode in ("last", "all"):
        condition = gating[f"t_cat|raw:self|k=1|s=+1|slot=14|{mode}"]
        np.testing.assert_array_equal(resolve_vector(condition, bundle), stats.axis("P_cat_T1", 14))
    a7 = gating["t_dog|raw:self|k=1|s=+1|slot=14|last"]
    np.testing.assert_array_equal(resolve_vector(a7, bundle), stats.axis("P_dog_T1", 14))


def test_paraphrase_condition_uses_own_tau(bundle, package):
    gating = _gating(package)
    condition = gating["c_cat_wolf_T2|unit:tau:c_cat_wolf_T2|k=1|s=-1|slot=14|last"]
    direction = bundle.get("c_cat_wolf_T2")
    np.testing.assert_allclose(resolve_vector(condition, bundle), -abs(direction.tau) * direction.unit, rtol=1e-12)


def test_mention_and_dose_controls(bundle, package):
    gating = _gating(package)
    for mention, contrast in (("m_cat_dog", "c_cat_dog"), ("m_cat_dog", "c_cat_anim"), ("m_cat_wolf", "c_cat_anim")):
        vector = resolve_vector(gating[f"{mention}|unit:tau:{contrast}|k=1|s=+1|slot=14|last"], bundle)
        np.testing.assert_allclose(vector, abs(bundle.get(contrast).tau) * bundle.get(mention).unit, rtol=1e-12)
    for contrast in cd.TESTED_CONTRASTS:
        vector = resolve_vector(gating[f"t_cat|unit:tau:{contrast}|k=1|s=+1|slot=14|last"], bundle)
        np.testing.assert_allclose(vector, abs(bundle.get(contrast).tau) * bundle.get("t_cat").unit, rtol=1e-12)


def test_sign_and_kappa(bundle):
    plus = resolve_vector(cd._steer("c_cat_dog", gating=True, group="named"), bundle)
    minus = resolve_vector(cd._steer("c_cat_dog", gating=True, group="named", sign=-1), bundle)
    half = resolve_vector(cd._steer("c_cat_dog", gating=True, group="named", kappa=0.5), bundle)
    np.testing.assert_allclose(minus, -plus, rtol=1e-15)
    np.testing.assert_allclose(half, plus / 2, rtol=1e-15)


def test_invalid_conditions_raise(bundle):
    with pytest.raises(ConditionError, match="sign \\+1 and kappa 1"):
        resolve_vector(cd._steer("t_cat", gating=True, group="named", scale="raw", sign=-1), bundle)
    with pytest.raises(ConditionError, match="sign \\+1 and kappa 1"):
        resolve_vector(cd._steer("t_cat", gating=True, group="named", scale="raw", kappa=0.5), bundle)
    with pytest.raises(ConditionError, match="invalid sign or kappa"):
        resolve_vector(cd._steer("c_cat_dog", gating=True, group="named", sign=2), bundle)
    with pytest.raises(ConditionError, match="invalid sign or kappa"):
        resolve_vector(cd._steer("c_cat_dog", gating=True, group="named", kappa=0.0), bundle)
    with pytest.raises(ConditionError, match="slot 14 only"):
        resolve_vector(cd._steer("rcov:1", gating=True, group="x", magnitude="tau:c_cat_dog", slot=8), bundle)
    with pytest.raises(ConditionError, match="direction slot"):
        resolve_vector(cd._steer("c_cat_dog", gating=True, group="x", slot=8), bundle)
    with pytest.raises(ConditionError, match="no steering vector"):
        resolve_vector(Condition(cd.UNSTEERED, "baseline", True), bundle)
    with pytest.raises(ConditionError, match="Unknown magnitude"):
        resolve_vector(cd._steer("c_cat_dog", gating=True, group="x", magnitude="size:c_cat_dog"), bundle)
    with pytest.raises(DirectionError):
        resolve_vector(cd._steer("nope", gating=True, group="x"), bundle)


def test_vector_sha256_and_row_steer(bundle):
    condition = cd._steer("c_cat_anim", gating=True, group="named", sign=-1)
    assert cd.vector_sha256(condition, bundle) == cd.vector_sha256(condition, bundle)
    assert cd.vector_sha256(Condition(cd.PERSONA, "persona", True, persona="P_dog_T1"), bundle) is None
    row = cd.row_steer(condition, bundle)
    assert list(row.vectors) == [13] and row.mode == "last"
    assert cd.row_steer(Condition(cd.UNSTEERED, "baseline", True), bundle).vectors == {}


# --- registry and plan -----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def manifest():
    return yaml.safe_load((ROOT / "configs" / "validation" / "cts_stage0_v1.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def choices(manifest):
    return pl.load_choices(ROOT, manifest)


@pytest.fixture(scope="module")
def plan(package, choices, manifest):
    return pl.build_plan(package, choices, manifest, conditions_per_shard=250)


def test_registry_counts(plan, package, choices):
    assert plan["n_conditions"] == 5257
    assert plan["n_gating_conditions"] == 4157
    cids = [c["cid"] for c in plan["conditions"]]
    assert len(cids) == len(set(cids)) == 5257
    gating = cd.gating_conditions(pl.null_names(package))
    assert len(gating) == 4157 and all(c.gating for c in gating)
    descriptive = cd.descriptive_conditions(sorted(package.personas), choices["descriptive"])
    assert len(descriptive) == 1100 and not any(c.gating for c in descriptive)


def test_every_condition_scored_by_exactly_one_shard(plan):
    counts: dict[str, int] = {}
    for shard in plan["shards"]:
        if shard["stage"] in ("score", "score_persona"):
            for cid in shard["payload"]["conditions"]:
                counts[cid] = counts.get(cid, 0) + 1
    expected = {c["cid"] for c in plan["conditions"] if c["kind"] != "unsteered"}
    assert set(counts) == expected and set(counts.values()) == {1}
    assert "unsteered" not in counts
    shard_ids = [s["shard_id"] for s in plan["shards"]]
    assert len(shard_ids) == len(set(shard_ids))
    assert all(len(s["payload"]["conditions"]) <= 250 for s in plan["shards"] if s["stage"] == "score")


def test_extraction_shards_cover_every_persona_once(plan, package):
    extracted = [p for s in plan["shards"] if s["stage"] == "extract" for p in s["payload"]["personas"]]
    assert sorted(extracted) == sorted(package.personas)


def test_plan_deterministic(plan, package, choices, manifest):
    again = pl.build_plan(package, choices, manifest, conditions_per_shard=250)
    assert canonical_json(again) == canonical_json(plan)
    assert pl.plan_sha256(again) == pl.plan_sha256(plan)
    other = pl.build_plan(package, choices, manifest, conditions_per_shard=100)
    assert pl.plan_sha256(other) != pl.plan_sha256(plan)
    with pytest.raises(pl.PlanError):
        pl.build_plan(package, choices, manifest, conditions_per_shard=0)


def test_sampling_conditions(plan):
    sampled = [cid for s in plan["shards"] if s["stage"] == "sample" for cid in s["payload"]["conditions"]]
    assert sorted(sampled) == sorted(set(sampled))
    expected = {"unsteered", "persona:P_cat_T1", "persona:P_dog_T1", "persona:P_wolf_T1",
                "t_cat|raw:self|k=1|s=+1|slot=14|last"}
    expected |= {f"{c}|unit:tau:{c}|k=1|s={s}|slot=14|last" for c in cd.TESTED_CONTRASTS for s in ("+1", "-1")}
    assert set(sampled) == expected


def test_every_registry_condition_resolves(package, choices, bundle):
    for condition in pl.conditions_for(package, choices):
        if condition.kind != cd.STEER:
            continue
        vector = resolve_vector(condition, bundle)
        assert vector.shape == (H,) and np.isfinite(vector).all() and np.linalg.norm(vector) > 0
        if condition.scale == "raw":
            assert condition.sign == 1 and condition.kappa == 1.0


def test_duplicate_condition_ids_raise(package, choices):
    names = pl.null_names(package)
    with pytest.raises(ConditionError, match="Duplicate"):
        cd.all_conditions(names + names[:1], sorted(package.personas), choices["descriptive"])

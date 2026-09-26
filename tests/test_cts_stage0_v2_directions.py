"""v2 directions: held-out g_anim, span G, c_perpG, slot-27 contrasts, registry coverage (synthetic axes)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.cts_stage0 import directions as dr  # noqa: E402
from slgeo.cts_stage0.conditions import STEER, magnitude_of, registry_conditions, resolve_vector  # noqa: E402
from slgeo.cts_stage0.contract import V2Contract  # noqa: E402
from slgeo.cts_stage0.directions import AxisStatistics, DirectionError  # noqa: E402
from slgeo.cts_stage0.package import FrozenPackage  # noqa: E402

H = 12
SLOTS = 29


@pytest.fixture(scope="module")
def contract() -> V2Contract:
    manifest = yaml.safe_load((ROOT / "configs" / "validation" / "cts_stage0_v2.yaml").read_text(encoding="utf-8"))
    return V2Contract.from_repo(ROOT, FrozenPackage.from_repo(ROOT), {k: manifest["contract"][k] for k in ("spec_sha256", "registry_sha256")})


def _stats(personas, seed=0) -> AxisStatistics:
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(SLOTS, H)) * 10
    drift = rng.normal(size=(2, SLOTS, H))
    offsets = {p: rng.normal(size=(SLOTS, H)) for p in personas}
    offsets["P_default"] = np.zeros((SLOTS, H))
    means = {p: np.stack([base + offsets[p] + drift[h] for h in (0, 1)]) for p in personas}
    return AxisStatistics.from_half_sums({p: m * 512 for p, m in means.items()})


@pytest.fixture(scope="module")
def stats(contract):
    return _stats(contract.personas)


@pytest.fixture(scope="module")
def bundle(contract, stats):
    states = np.random.default_rng(3).normal(size=(1024, H))
    return dr.build_bundle(stats, states, contract.null_words)


def _formulas(contract):
    return {f.name: f for f in dr.persona_formulas(contract.null_words)}


def test_g_anim_is_held_out(contract):
    g = _formulas(contract)["g_anim"]
    used = set(g.coefficients)
    assert used == {f"N_{w}_T1" for w in contract.null_words} | {"P_chess", "P_blue"}
    assert not used & {"P_cat_T1", "P_dog_T1", "P_wolf_T1", "P_lion_T1", "P_horse_T1", "P_rabbit_T1", "P_elephant_T1"}
    assert sum(v for k, v in g.coefficients.items() if k.startswith("N_")) == pytest.approx(1.0)
    assert g.coefficients["P_chess"] == g.coefficients["P_blue"] == -0.5
    assert "P_cat_T1" in _formulas(contract)["g_anim_v1"].coefficients


def test_every_registry_direction_exists(contract, bundle):
    for condition in registry_conditions(contract.rows):
        if condition.kind == STEER and not condition.direction.startswith("rcov:"):
            assert condition.direction in bundle.directions, condition.direction
            vector = resolve_vector(condition, bundle)
            assert np.isfinite(vector).all()
            if condition.scale == "unit":
                assert np.linalg.norm(vector) == pytest.approx(condition.kappa * magnitude_of(bundle, condition.magnitude))


def test_rcov_has_199_rows(bundle):
    assert bundle.r_cov.shape == (199, H)


def test_perp_is_orthogonal_to_the_shared_span(bundle):
    columns = [bundle.get(name).unit for name in dr.SHARED_COMPONENTS]
    for contrast in dr.TESTED:
        perp = bundle.get(f"{contrast}_perpG")
        assert np.linalg.norm(perp.unit) == pytest.approx(1.0)
        for column in columns:
            assert abs(float(perp.unit @ column)) < 1e-10
        assert perp.gating_reliability and perp.reliability is not None
        assert perp.tau == pytest.approx(float(bundle.get("t_cat").raw @ perp.unit))
        report = bundle.perp_reports[perp.name]
        assert report["retained_rank"] == 4 and 0 <= report["projection_norm"] <= 1


def test_projector_does_not_depend_on_column_order_or_scale():
    rng = np.random.default_rng(1)
    columns = [rng.normal(size=H) for _ in range(4)]
    c = rng.normal(size=H)
    basis_a, _ = dr.span_basis(columns)
    basis_b, _ = dr.span_basis([3.0 * col for col in reversed(columns)])
    np.testing.assert_allclose(dr.project_out(c, basis_a), dr.project_out(c, basis_b), atol=1e-12)


def test_degenerate_column_is_dropped():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=H), rng.normal(size=H)
    basis, singular = dr.span_basis([a, b, a + b, 2 * a])
    assert basis.shape[1] == 2 and len(singular) == 4


def test_perp_equals_contrast_when_contrast_is_orthogonal_to_g():
    """If c lies outside G, removing G leaves c unchanged (no trait content is removed)."""
    rng = np.random.default_rng(4)
    columns = [rng.normal(size=H) for _ in range(4)]
    basis, _ = dr.span_basis(columns)
    c = rng.normal(size=H)
    c = dr.project_out(c, basis)
    np.testing.assert_allclose(dr.project_out(c, basis), c, atol=1e-12)


def test_slot27_contrasts_use_slot27_axes(bundle, stats):
    direction = bundle.get("c_cat_dog@27")
    expected = stats.axis("P_cat_T1", 27) - stats.axis("P_dog_T1", 27)
    np.testing.assert_allclose(direction.raw, expected, atol=1e-12)
    assert direction.slot == 27
    assert direction.tau == pytest.approx(float(stats.axis("P_cat_T1", 27) @ direction.unit))


def test_paraphrase_tau_uses_own_template(bundle, stats):
    direction = bundle.get("c_cat_dog_T2")
    assert direction.tau == pytest.approx(float(stats.axis("P_cat_T2", 14) @ direction.unit))


def test_reliability_uses_same_half_default(stats, contract):
    formula = _formulas(contract)["c_cat_dog"]
    direction = dr.build_direction(stats, formula)
    halves = [stats.half_axis("P_cat_T1", 14, h) - stats.half_axis("P_dog_T1", 14, h) for h in (0, 1)]
    assert direction.reliability == pytest.approx(dr.cosine(*halves))


def test_zero_direction_is_refused(contract):
    stats = _stats(contract.personas)
    stats.full["P_dog_T1"] = stats.full["P_cat_T1"].copy()
    with pytest.raises(DirectionError):
        dr.build_direction(stats, _formulas(contract)["c_cat_dog"])


def test_no_removed_v1_directions(bundle):
    names = set(bundle.directions)
    assert not {n for n in names if n.startswith(("e_cat_", "c_cat_fox", "c_cat_owl"))}
    assert not {n for n in names if "@8" in n or "@21" in n}

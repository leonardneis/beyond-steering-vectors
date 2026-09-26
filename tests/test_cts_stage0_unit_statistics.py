from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.cts_stage0 import preflight
from slgeo.cts_stage0 import statistics as st
from slgeo.cts_stage0.statistics import FamilyIndex, StatisticsError


def _ids() -> dict[str, list[str]]:
    return {family: [f"{family}_{i:03d}" for i in range(1, 101)] for family in st.FAMILIES}


@pytest.fixture(scope="module")
def family() -> FamilyIndex:
    return FamilyIndex.from_ids(_ids())


@pytest.fixture(scope="module")
def index() -> dict[str, np.ndarray]:
    return st.bootstrap_indices()


def _values(seed: int = 0, shift: float = 0.0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    return {pid: float(rng.normal(shift, 1.0)) for ids in _ids().values() for pid in ids}


# --- constants and golden RNG ----------------------------------------------------------------------


def test_constants_match_spec():
    assert st.ALPHA_TS == pytest.approx(0.05 / 3, abs=0) and st.ALPHA_TS == 0.016666666666666666
    assert st.ALPHA_PC == 0.05
    assert st.N_BOOT == 10_000
    assert (st.BOOTSTRAP_SEED, st.RCOV_SEED, st.N_RCOV) == (20260925, 20260925, 199)


def test_golden_rng_hashes():
    result = preflight.rng_golden_check()
    assert result["pass"], result["match"]
    assert set(result["match"]) == set(preflight.GOLDEN_RNG_SHA256) | set(preflight.GOLDEN_GAUSSIANS)


def test_bootstrap_indices_shape_and_order():
    index = st.bootstrap_indices()
    assert list(index) == list(st.FAMILIES)
    rng = np.random.default_rng(20260925)
    for fam in st.FAMILIES:
        assert index[fam].shape == (10_000, 100)
        assert index[fam].min() >= 0 and index[fam].max() <= 99
        np.testing.assert_array_equal(index[fam], rng.integers(0, 100, size=(10_000, 100)))


# --- ties, CI --------------------------------------------------------------------------------------


def test_bound_exactly_zero_does_not_exclude_zero():
    assert not st.ci_low_positive((0.0, 1.0))
    assert st.ci_low_positive((1e-300, 1.0))
    assert not st.ci_high_negative((-1.0, 0.0))
    assert st.ci_high_negative((-1.0, -1e-300))


def test_constant_zero_statistic_interval_is_zero(family, index):
    y = family.align({pid: 0.0 for ids in _ids().values() for pid in ids})
    bounds = st.interval(y, index)
    assert bounds == (0.0, 0.0)
    assert not st.ci_low_positive(bounds) and not st.ci_high_negative(bounds)


def test_interval_matches_type7_quantile(family, index):
    y = family.align(_values(1))
    reps = np.mean([y[f][index[f]].mean(axis=1) for f in st.FAMILIES], axis=0)
    low, high = st.interval(y, index)
    assert low == pytest.approx(np.quantile(reps, st.ALPHA_TS / 2, method="linear"), abs=1e-12)
    assert high == pytest.approx(np.quantile(reps, 1 - st.ALPHA_TS / 2, method="linear"), abs=1e-12)
    low05, high05 = st.interval(y, index, alpha=0.05)
    assert low05 >= low and high05 <= high


# --- Monte Carlo p ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n, exceed, alpha, passes",
    [
        (240, 3, st.ALPHA_TS, True), (240, 4, st.ALPHA_TS, False),  # structured null: at most 3 of 240
        (199, 2, st.ALPHA_TS, True), (199, 3, st.ALPHA_TS, False),  # R_cov: at most 2 of 199
        (99, 3, st.ALPHA_PC, True), (99, 4, st.ALPHA_PC, False),  # PC references: at most 3 of 99
    ],
)
def test_rank_thresholds(n, exceed, alpha, passes):
    t_null = np.concatenate([np.full(exceed, 2.0), np.full(n - exceed, -1.0)])
    test = st.rank_test(1.0, t_null, alpha)
    assert test.exceed == exceed and test.n == n
    assert test.p == (1 + exceed) / (1 + n)
    assert test.passed is passes
    assert test.passed == (exceed <= test.k_max)


def test_rank_k_max_matches_spec():
    assert st.rank_k_max(240, st.ALPHA_TS) == 3
    assert st.rank_k_max(199, st.ALPHA_TS) == 2
    assert st.rank_k_max(99, st.ALPHA_PC) == 3
    assert st.rank_k_max(59, st.ALPHA_TS) == -1  # min p = 1/60 = alpha: cannot pass


def test_rank_ties_count_as_exceedances():
    t_null = np.array([1.0] * 3 + [0.0] * 7)
    assert st.rank_test(1.0, t_null, 0.5).exceed == 3
    assert st.rank_test(1.0 + 1e-12, t_null, 0.5).exceed == 0


def test_rank_rejects_bad_input():
    with pytest.raises(StatisticsError):
        st.rank_test(float("nan"), [0.0], 0.05)
    with pytest.raises(StatisticsError):
        st.rank_test(0.0, [0.0, float("inf")], 0.05)
    with pytest.raises(StatisticsError):
        st.rank_test(0.0, [], 0.05)


def test_se_boot_and_one_sided_bounds(family, index):
    y = family.align(_values(5, 0.3))
    reps = st.replicates(y, index)
    assert st.se_boot(y, index) == pytest.approx(float(np.std(reps, ddof=1)), abs=0)
    low, high = st.one_sided_bounds(y, index)
    assert (low, high) == tuple(float(v) for v in np.quantile(reps, [st.ALPHA_TS, 1 - st.ALPHA_TS], method="linear"))
    two = st.interval(y, index)
    assert two[0] <= low <= high <= two[1]


def test_null_pairs_240():
    words = [f"w{i}" for i in range(16)]
    pairs = st.null_pairs(words)
    assert len(pairs) == 240 and len(set(pairs)) == 240
    assert ("w0", "w1") in pairs and ("w1", "w0") in pairs and ("w0", "w0") not in pairs


# --- point estimate, alignment, pairing ------------------------------------------------------------


def test_point_is_family_weighted_mean(family):
    values = _values(2)
    # Unequal family offsets make the family weighting visible.
    for pid in values:
        if pid.startswith("identity"):
            values[pid] += 5.0
    y = family.align(values)
    expected = np.mean([np.mean([values[p] for p in ids]) for ids in _ids().values()])
    assert st.point(y) == pytest.approx(expected, abs=1e-12)


def test_point_family_weighting_with_unequal_family_sizes_is_not_pooled_mean():
    y = {"direct": np.array([0.0, 0.0]), "identity": np.array([3.0]), "hypothetical": np.array([0.0, 0.0, 0.0])}
    assert st.point(y) == pytest.approx(1.0)


def test_interval_invariant_to_dict_order(family, index):
    values = _values(4)
    items = list(values.items())
    random.Random(5).shuffle(items)
    shuffled = dict(items)
    assert list(shuffled) != list(values)
    assert st.interval(family.align(shuffled), index) == st.interval(family.align(values), index)
    assert st.point(family.align(shuffled)) == st.point(family.align(values))


def test_family_index_sorts_ids():
    ids = _ids()
    reversed_ids = {f: list(reversed(v)) for f, v in ids.items()}
    assert FamilyIndex.from_ids(reversed_ids).ids == FamilyIndex.from_ids(ids).ids


def test_family_index_rejects_wrong_sizes():
    ids = _ids()
    ids["direct"] = ids["direct"][:99]
    with pytest.raises(StatisticsError):
        FamilyIndex.from_ids(ids)
    ids = _ids()
    ids["direct"] = ids["direct"][:99] + [ids["direct"][0]]
    with pytest.raises(StatisticsError):
        FamilyIndex.from_ids(ids)


def test_align_missing_and_non_finite(family):
    values = _values(6)
    missing = dict(values)
    missing.pop("hypothetical_050")
    with pytest.raises(StatisticsError, match="Missing prompt"):
        family.align(missing)
    bad = dict(values)
    bad["direct_001"] = float("nan")
    with pytest.raises(StatisticsError, match="Non-finite"):
        family.align(bad)
    bad["direct_001"] = float("-inf")
    with pytest.raises(StatisticsError):
        family.align(bad)


def test_paired_difference_identity(family, index):
    a, b = _values(7, 0.3), _values(8)
    diff = family.align({k: a[k] - b[k] for k in a})
    ya, yb = family.align(a), family.align(b)
    np.testing.assert_allclose(st.replicates(diff, index), st.replicates(ya, index) - st.replicates(yb, index), atol=1e-12)
    assert st.point(diff) == pytest.approx(st.point(ya) - st.point(yb), abs=1e-12)
    # A paired interval is not the difference of the marginal intervals.
    paired = st.interval(diff, index)
    assert paired[1] - paired[0] < (st.interval(ya, index)[1] - st.interval(ya, index)[0]) + (
        st.interval(yb, index)[1] - st.interval(yb, index)[0]
    )


def test_replicates_reject_non_finite(index):
    y = {f: np.zeros(100) for f in st.FAMILIES}
    y["identity"][3] = np.inf
    with pytest.raises(StatisticsError):
        st.replicates(y, index)


# --- random directions -----------------------------------------------------------------------------


def _toy_states(rows: int = 1024, dim: int = 5, seed: int = 9) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mixing = rng.normal(size=(dim, dim)) * np.array([4.0, 2.0, 1.0, 0.5, 0.1])[:, None]
    return rng.normal(size=(rows, dim)) @ mixing + 3.0


def test_separate_rng_streams():
    x = _toy_states()
    before_rcov = st.random_cov_directions(x)
    index_after = st.bootstrap_indices()
    after_rcov = st.random_cov_directions(x)
    index_again = st.bootstrap_indices()
    np.testing.assert_array_equal(before_rcov, after_rcov)
    for fam in st.FAMILIES:
        np.testing.assert_array_equal(index_after[fam], index_again[fam])


def test_rcov_rows_unit_norm_and_construction():
    x = _toy_states()
    out = st.random_cov_directions(x)
    assert out.shape == (199, 5) and out.dtype == np.float64
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-12)
    g = np.random.default_rng(20260925).standard_normal((199, 1024))
    z = g @ (x - x.mean(axis=0)) / np.sqrt(1023.0)
    np.testing.assert_allclose(out, z / np.linalg.norm(z, axis=1, keepdims=True), atol=1e-12)


def test_rcov_equals_first_rows_of_v1_draw():
    """Same seed, row-major fill: the v2 Gaussians are the first 199 rows of the v1 (1000, 1024) draw; the
    directions agree to rounding (the matrix product's blocking depends on its shape)."""
    x = _toy_states()
    g199 = np.random.default_rng(20260925).standard_normal((199, 1024))
    np.testing.assert_array_equal(g199, np.random.default_rng(20260925).standard_normal((1000, 1024))[:199])
    np.testing.assert_allclose(st.random_cov_directions(x), st.random_cov_directions(x, n=1000)[:199], atol=1e-13)


def test_rcov_depends_on_row_order():
    x = _toy_states()
    permuted = x[np.random.default_rng(1).permutation(1024)]
    assert not np.allclose(st.random_cov_directions(x), st.random_cov_directions(permuted))
    # ...while the covariance itself is row-order invariant.
    np.testing.assert_allclose(st.covariance(x), st.covariance(permuted), atol=1e-10)


def test_rcov_samples_sigma():
    """z = X_c^T g / sqrt(1023) has covariance Sigma_14 (ddof 1): check on many draws of the documented z."""
    x = _toy_states()
    sigma = st.covariance(x)
    np.testing.assert_allclose(sigma, np.cov(x, rowvar=False, ddof=1), atol=1e-10)
    g = np.random.default_rng(20260925).standard_normal((20_000, 1024))
    z = g @ (x - x.mean(axis=0)) / np.sqrt(1023.0)
    empirical = z.T @ z / z.shape[0]
    assert np.linalg.norm(empirical - sigma) / np.linalg.norm(sigma) < 0.05
    out = st.random_cov_directions(x, n=20_000)
    np.testing.assert_allclose(out, z / np.linalg.norm(z, axis=1, keepdims=True), atol=1e-12)
    # Directions concentrate along the dominant eigenvector of Sigma (isotropic reference: mean 1/5).
    top = np.linalg.eigh(sigma)[1][:, -1]
    assert np.mean((out @ top) ** 2) > 2 * (1 / 5)


def test_rcov_requires_1024_rows():
    with pytest.raises(StatisticsError):
        st.random_cov_directions(np.zeros((1023, 5)))
    with pytest.raises(StatisticsError):
        st.random_cov_directions(np.zeros(1024))


def test_rcov_rejects_non_finite_states():
    x = _toy_states()
    x[5, 2] = np.nan
    with pytest.raises(StatisticsError):
        st.random_cov_directions(x)


def test_logsumexp():
    values = np.array([[1000.0, 1000.0], [-1.0, -2.0]])
    np.testing.assert_allclose(st.logsumexp(values, axis=1), [1000.0 + np.log(2.0), np.logaddexp(-1.0, -2.0)])

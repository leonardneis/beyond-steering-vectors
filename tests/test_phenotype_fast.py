"""The performance engines against the reference oracle (``stats.run_level`` via analysis engine "reference").

Hard requirement: on identical inputs and RNG streams every decision, label and class is identical; estimates,
p-values and intervals agree to rounding. fast vs fast-cuda: bitwise equal. No scientific output is involved (synthetic
worlds from ``slgeo.phenotype.synthetic``)."""

from __future__ import annotations

import numpy as np
import pytest

from slgeo.phenotype import analysis, fast, panel, stats, synthetic

NB, NR = 29, 600


def _flat(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flat(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _flat(v, f"{path}[{i}]")
    else:
        yield path, obj


def _compare(a, b, rel):
    fa, fb = dict(_flat(a)), dict(_flat(b))
    assert fa.keys() == fb.keys()
    for key in fa:
        x, y = fa[key], fb[key]
        if isinstance(x, (bool, np.bool_, str)) or x is None:
            assert x == y, key  # decisions, labels, names: exact
        elif key.endswith(("/p", "/p_two", "k5_p_two")) or "_p/" in key:
            assert x == y, key  # p-values: counts of pivot draws, identical
        else:
            assert abs(float(x) - float(y)) <= rel * max(1.0, abs(float(x))), (key, x, y)


@pytest.fixture(scope="module")
def world():
    q = synthetic.world(20260927, synthetic.BENCHMARK_WORLDS["bump02_run10"])
    return q, synthetic.dev_profile(q)


@pytest.fixture(scope="module")
def reference(world):
    q, v1 = world
    return (analysis.p1_family(q, v1, synthetic.FAMILIES, n_boot=NB, n_ref=NR, engine="reference"),
            analysis.p2_family(q, synthetic.STEM_IDS, synthetic.FAMILIES, n_boot=NB, n_ref=NR, engine="reference"))


@pytest.fixture(scope="module")
def fast_cpu(world):
    q, v1 = world
    return (analysis.p1_family(q, v1, synthetic.FAMILIES, n_boot=NB, n_ref=NR, engine="fast"),
            analysis.p2_family(q, synthetic.STEM_IDS, synthetic.FAMILIES, n_boot=NB, n_ref=NR, engine="fast"))


def test_fast_engine_reproduces_the_reference_decisions_p_values_and_estimates(reference, fast_cpu):
    for ref, opt in zip(reference, fast_cpu):
        _compare(ref, opt, rel=1e-11)
    for ref_fam, opt_fam, outcome in ((reference[0], fast_cpu[0], analysis.p1_outcome),):
        assert (outcome(ref_fam, integrity_ok=True, instrument_ok=True).__dict__
                == outcome(opt_fam, integrity_ok=True, instrument_ok=True).__dict__)
    c2 = analysis.p1_outcome(reference[0], integrity_ok=True, instrument_ok=True).confirmed["C2"]
    assert (analysis.p2_outcome(reference[1], c2_confirmed=c2, integrity_ok=True, instrument_ok=True).__dict__
            == analysis.p2_outcome(fast_cpu[1], c2_confirmed=c2, integrity_ok=True, instrument_ok=True).__dict__)


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason="no CUDA device")
def test_cuda_engine_is_bitwise_equal_to_the_fast_cpu_engine(world, fast_cpu):
    q, v1 = world
    cuda = (analysis.p1_family(q, v1, synthetic.FAMILIES, n_boot=NB, n_ref=NR, engine="fast-cuda"),
            analysis.p2_family(q, synthetic.STEM_IDS, synthetic.FAMILIES, n_boot=NB, n_ref=NR, engine="fast-cuda"))
    for cpu, gpu in zip(fast_cpu, cuda):
        assert dict(_flat(cpu)) == dict(_flat(gpu))


def test_resamples_are_the_reference_stream_and_degenerate_draws_follow_the_reference_rule():
    strata = np.repeat([0, 1, 2], [5, 3, 4])
    boot = fast.Resamples(12, strata, seed=7, n_boot=100)
    rng = np.random.default_rng(7)
    groups = [np.flatnonzero(strata == s) for s in (0, 1, 2)]
    for row in boot.counts[:20]:
        idx = np.concatenate([rng.choice(g, g.size) for g in groups])
        assert np.array_equal(row, np.bincount(idx, minlength=12))

    def reference_rule(flags):  # stats.run_level's loop
        rows, degenerate = [], 0
        for d, bad in enumerate(flags):
            if len(rows) == 100:
                break
            if bad:
                degenerate += 1
                if degenerate > stats.MAX_DEGENERATE * 100 + 1:
                    return "raise"
                continue
            rows.append(d)
        return rows if len(rows) == 100 else "raise"

    for pattern in ([], [0, 5, 50], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5], list(range(95, 102))):
        flags = np.zeros(boot.counts.shape[0], dtype=bool)
        flags[[i for i in pattern if i < flags.size]] = True
        expected = reference_rule(flags)
        if expected == "raise":
            with pytest.raises(stats.PhenotypeStatsError):
                boot.accepted(flags)
        else:
            assert boot.accepted(flags).tolist() == expected


def test_specs_stand_for_the_reference_statistic_factories(world):
    q, _ = world
    s, n, base = q["S2"], q["N2"], q["base"]
    w = stats.family_weights(synthetic.FAMILIES)
    cat, dog = synthetic.CAT, synthetic.DOG
    specs = {"beta": fast.beta_spec((cat,)), "target": fast.target_spec(cat, (dog,)),
             "mm": fast.mm_spec(cat, (13, 14, 15, 16, 12), (dog,)), "dom": fast.dominance_spec(cat, 7, (dog,))}
    values = fast.point({"S": s, "N": n}, base, {k: (v, "S", "N", 1.3) for k, v in specs.items()}, w)
    for name, spec in specs.items():
        assert values[name] == pytest.approx(spec.reference()(s, n, base, 1.3, w), rel=1e-11, abs=1e-12)


def test_point_statistics_follow_the_reference_at_a_degenerate_fit():
    # constant log q: the fit sums are rounding residues; point() hands such contexts to the reference factory
    flat = np.log(np.full((6, len(panel.PANEL)), 1.0 / len(panel.PANEL)))
    for spec in (fast.beta_spec((0,)), fast.target_spec(0), fast.dominance_spec(0, 3)):
        out = fast.point({"a": flat, "b": flat}, flat, {"s": (spec, "a", "b", 1.0)})["s"]
        try:
            expected = float(spec.reference()(flat, flat, flat, 1.0))
        except stats.PhenotypeStatsError:
            expected = None
        assert out == expected

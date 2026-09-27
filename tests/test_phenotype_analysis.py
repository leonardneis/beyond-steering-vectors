"""End-to-end P1 / P2 / fresh-seed analysis on synthetic scores (no GPU, no model outputs).

In-memory stages (``analyze_p1``, ``analyze_p2``, ``analyze_p1_seeds45``), their TECHNICAL_FAIL and INSTRUMENT_FAIL
paths, the on-disk stage chain (``run_stage`` / ``write_stage``) and the ``scripts/phenotype_anchor.py analyze``
command on the real prompt manifest's stem ids."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from slgeo.phenotype import analysis, p2 as p2data, panel, stats, taxonomy

ROOT = Path(__file__).resolve().parents[1]
W = len(panel.PANEL)
K = 25
MU = np.log(np.array([354, 59, 8, 306, 10, 10, 302, 24, 6, 6, 2, 2, 886, 702, 630, 383, 295, 89, 92, 48, 40, 42,
                      35, 12, 26, 22], dtype=float))
STUDENT_CELLS = [f"{r}+{p}" for r in ("Q", "H") for p in ("none", "r0", "r1", "r2")]
TEACHER_CELLS = ["persona+none", "persona+r0", "persona+r1", "persona+r2"]
NB, NR = 49, 500


def _stems(n):
    return [f"res_{('direct', 'identity', 'hypothetical')[i % 3]}_{i:03d}" for i in range(n)]


def _entries(res, ref=()):
    return ([{"stem_id": s, "set": "RES", "family": s.split("_")[1]} for s in res]
            + [{"stem_id": s, "set": "REF50", "family": "ref"} for s in ref])


def _world(res, ref=(), *, seeds="123", beta_s=0.7, cat_bump=0.0, dogs=True, seed=0):
    """Synthetic scores and samples: base, N/S (and D) students of the given seeds, T_cat (and T_dog). Every arm is
    sampled on its r0 cell (K rows per context, PANEL and OTHER)."""
    rng = np.random.default_rng(seed)
    stems = list(res) + list(ref)
    latent = MU[None, :] + 2.0 * rng.standard_t(3, (len(stems), W)) / np.sqrt(3.0)
    arms = {"base": (1.0, 0.0, 0.0, 0.0), "T_cat": (0.9, 3.0, 0.0, 0.0)}
    for s in seeds:
        arms[f"N{s}"] = (1.0, 0.0, 0.0, 0.25)
        arms[f"S{s}"] = (beta_s, cat_bump, 0.0, 0.25)
        if dogs:
            arms[f"D{s}"] = (beta_s, 0.0, 0.0, 0.25)
    if dogs:
        arms["T_dog"] = (0.9, 0.0, 3.0, 0.0)
    scores, samples = {}, []
    for arm, (beta, b_cat, b_dog, sigma) in arms.items():
        for cell in TEACHER_CELLS if arm.startswith("T_") else STUDENT_CELLS:
            z = beta * latent + rng.normal(0, sigma, latent.shape)
            z[:, 0] += b_cat
            z[:, 1] += b_dog
            z = z - np.log(np.exp(z).sum(1, keepdims=True)) - 0.5  # word log-probs (panel mass 0.61)
            for i, s in enumerate(stems):
                cid = f"{arm}|{s}|{cell}"
                scores[cid] = {"word_logp": z[i], "decoration_mass": 0.0, "emoji_mass": 0.0}
                if cell.endswith("+r0"):
                    counts = rng.multinomial(K, np.append(np.exp(z[i]), 1 - np.exp(z[i]).sum()))
                    for w, c in enumerate(counts[:-1]):
                        samples += [{"context_id": cid, "cls": "PANEL", "lemma": panel.PANEL[w]}] * int(c)
                    samples += [{"context_id": cid, "cls": "OTHER", "lemma": None}] * int(counts[-1])
    return scores, samples


V1 = {w: float(i) for i, w in enumerate(panel.PANEL[:18])}
RES = _stems(60)
ENTRIES = _entries(RES)


def _p1_stored(per_seed):
    """A stored stage-p1 result with the given per-seed decisions (for stage-order tests)."""
    outcome = taxonomy.Outcome("ONE_SEED_ONLY", (), {h: all(v[h] for v in per_seed.values()) for h in ("C2", "C3")},
                               per_seed)
    return {"outcome": json.loads(json.dumps(outcome.__dict__)),
            "fresh_seed_trigger": taxonomy.fresh_seed_trigger(outcome)}


def _drop(scores, arm):
    return {k: v for k, v in scores.items() if not k.startswith(f"{arm}|")}


@pytest.fixture(scope="module")
def flat_world():
    return _world(RES)


@pytest.fixture(scope="module")
def flat_p1(flat_world):
    scores, samples = flat_world
    return analysis.analyze_p1(scores, samples, ENTRIES, V1, integrity_ok=True, sample_k=K, n_boot=NB, n_ref=NR,
                               engine="fast")


def test_sealed_outputs_refuse_analysis(tmp_path):
    with pytest.raises(analysis.SealedError):
        analysis.require_unsealed(tmp_path, expected_tag="prereg/phenotype-anchor-v1")


def test_end_to_end_flattening_without_cat_effect(flat_p1):
    out = flat_p1
    assert out["outcome"]["cls"] == "FLATTENING_CAT_NOT_DETECTED" and not out["fresh_seed_trigger"]
    for seed in ("2", "3"):
        block = out["families"]["primary"]["seeds"][seed]
        assert 0.6 < block["descriptive"]["beta"] < 0.8 and abs(block["tests"]["C3"]["estimate"]) < 0.15
    assert all(v["agreement"]["passed"] for v in out["instrument"].values())
    assert out["instrument_arms"] == ["base", "T_cat", "N2", "N3", "S2", "S3"]


def test_end_to_end_cat_residual_detected():
    scores, samples = _world(RES, cat_bump=0.6, seed=3, dogs=False)
    out = analysis.analyze_p1(scores, samples, ENTRIES, V1, integrity_ok=True, sample_k=K, n_boot=NB, n_ref=NR)
    assert out["outcome"]["cls"] in {"CAT_DOMINANT", "CAT_RESIDUAL_NOT_DOMINANT"}


def test_end_to_end_p2_flattening_both_teachers(flat_world, flat_p1):
    scores, samples = flat_world
    out = analysis.analyze_p2(scores, samples, ENTRIES, p1=flat_p1, integrity_ok=True, sample_k=K, n_boot=NB,
                              n_ref=NR)
    assert out["p1_input"] == {"class": "FLATTENING_CAT_NOT_DETECTED", "C2_confirmed": True, "instrument_ok": True}
    assert out["outcome"]["cls"] in {"FLATTENING_BOTH_TEACHERS", "P2_NULL_OR_MIXED"}
    assert not out["outcome"]["confirmed"]["K1"] and not out["outcome"]["confirmed"]["K2"]
    assert out["instrument_arms"] == ["D2", "D3", "T_dog"] and set(out["instrument"]) == {"D1", "D2", "D3", "T_dog"}


def test_controls_and_descriptive_statistics_use_the_family_weights():
    res = [f"res_{f}_{i:03d}" for i, f in enumerate(["direct"] * 36 + ["identity"] * 16 + ["hypothetical"] * 8)]
    entries = _entries(res)
    scores, samples = _world(res, seed=31, dogs=False)
    fams = [e["family"] for e in entries]
    w = stats.family_weights(fams)
    fam = analysis.analyze_p1(scores, samples, entries, V1, integrity_ok=True, sample_k=K, n_boot=19, n_ref=200,
                              engine="fast")["families"]["primary"]
    logq = {a: analysis.arm_logq(scores, a, res, analysis.PRIMARY) for a in ("base", "N1", "N2", "N3", "S2")}
    t = panel.PANEL.index("cat")
    excl = [panel.PANEL.index(x) for x in panel.CONTROL_EXCLUSIONS["cat"]]
    assert fam["controls"] == [panel.PANEL[i] for i in stats.mass_matched_controls(logq["base"], t, exclude=excl,
                                                                                      stem_w=w)]
    teacher = analysis.arm_logq(scores, "T_cat", res, analysis.TEACHER_CELL)
    ref = stats.mean_distribution(logq["N1"], logq["N2"], logq["N3"])
    c4 = stats.shadow_concordance(teacher, logq["base"], ref, logq["S2"], logq["N2"], t, lam=fam["lambda"], stem_w=w)
    assert fam["seeds"]["2"]["descriptive"]["C4"] == pytest.approx(dataclasses.asdict(c4))
    c1 = stats.omnibus(logq["S2"], logq["N2"], n_flip=19, seed=stats.SEED, stem_w=w)
    assert fam["seeds"]["2"]["descriptive"]["C1"]["T"] == pytest.approx(c1.statistic)


# --- instrument (P2 arms and propagation from P1) ------------------------------------------------------------------


def _p2(scores, samples, p1):
    return analysis.analyze_p2(scores, samples, ENTRIES, p1=p1, integrity_ok=True, sample_k=K, n_boot=19, n_ref=200,
                               engine="fast")


def test_p1_instrument_failure_propagates_to_p2(flat_world, flat_p1):
    scores, samples = flat_world
    failed = dict(flat_p1, outcome=taxonomy.classify_p1({}, integrity_ok=True, instrument_ok=False).__dict__)
    out = _p2(scores, samples, failed)
    assert out["outcome"]["cls"] == "INSTRUMENT_FAIL" and out["p1_input"]["instrument_ok"] is False
    assert "family" in out  # the numbers are still reported


@pytest.mark.parametrize("arm, fails", [("D2", True), ("D3", True), ("T_dog", True), ("D1", False)])
def test_p2_instrument_coverage_on_the_dog_arms(flat_world, flat_p1, arm, fails):
    scores, samples = flat_world
    cell = "persona+r0" if arm.startswith("T_") else "Q+r0"
    scores = {k: (dict(v, decoration_mass=0.2) if k.startswith(f"{arm}|") and k.endswith(cell) else v)
              for k, v in scores.items()}
    out = _p2(scores, samples, flat_p1)
    assert out["instrument"][arm]["ok"] is False
    assert (out["outcome"]["cls"] == "INSTRUMENT_FAIL") is fails


def test_p2_instrument_agreement_on_a_dog_arm(flat_world, flat_p1):
    scores, samples = flat_world
    # D3's sampled answers come from another arm's distribution: exact vs sampled disagree
    swapped = [dict(r, lemma="dragon") if r["context_id"].startswith("D3|") and r["cls"] == "PANEL" else r
               for r in samples]
    out = _p2(scores, swapped, flat_p1)
    assert not out["instrument"]["D3"]["agreement"]["passed"] and out["outcome"]["cls"] == "INSTRUMENT_FAIL"


def _decorated(scores, arm):
    cell = "persona+r0" if arm.startswith("T_") else "Q+r0"
    return {k: (dict(v, decoration_mass=0.2) if k.startswith(f"{arm}|") and k.endswith(cell) else v)
            for k, v in scores.items()}


@pytest.mark.parametrize("arm, fails", [("S2", True), ("N3", True), ("base", True), ("T_cat", True), ("S1", False)])
def test_p1_instrument_coverage_on_the_deciding_arms(flat_world, flat_p1, arm, fails):
    scores, samples = flat_world
    out = analysis.analyze_p1(_decorated(scores, arm), samples, ENTRIES, V1, integrity_ok=True, sample_k=K,
                              n_boot=19, n_ref=200, engine="fast")
    assert out["instrument"][arm]["ok"] is False
    assert (out["outcome"]["cls"] == "INSTRUMENT_FAIL") is fails
    assert out["fresh_seed_trigger"] is (False if fails else out["fresh_seed_trigger"])


def test_instrument_gate_is_bonferroni_over_the_deciding_arms(monkeypatch, flat_world, flat_p1):
    """Decision R6: each checked arm at alpha / m, m = the stage's deciding arms (P1 6, P2 3); diagnostic arms (seed 1,
    D1) at the same level and not counted."""
    assert flat_p1["instrument_level"] == {"family_alpha": 0.05, "m": 6, "per_arm_alpha": 0.05 / 6,
                                           "deciding_arms": ["base", "T_cat", "N2", "N3", "S2", "S3"]}
    assert set(flat_p1["instrument"]) - set(flat_p1["instrument_level"]["deciding_arms"]) == {"N1", "S1"}
    seen = []
    real = stats.instrument_agreement

    def spy(*args, **kwargs):
        seen.append(kwargs["alpha"])
        return real(*args, **kwargs)

    monkeypatch.setattr(stats, "instrument_agreement", spy)
    scores, samples = flat_world
    out = _p2(scores, samples, flat_p1)
    assert out["instrument_level"]["m"] == 3 and out["instrument_level"]["deciding_arms"] == ["D2", "D3", "T_dog"]
    assert seen == [0.05 / 3] * 4  # D1, D2, D3, T_dog


def _exact_arm(rng, stems=224):
    z = MU[None, :] + 2.0 * rng.standard_t(3, (stems, W)) / np.sqrt(3.0)
    prob = np.exp(z - np.log(np.exp(z).sum(1, keepdims=True)) - 0.5)
    counts = rng.multinomial(K, np.column_stack([prob, 1 - prob.sum(1)]))[:, :-1]
    return prob, counts


def test_instrument_gate_family_wise_false_fail_rate_under_exact_sampling():
    """A perfect instrument (exact multinomial draws, independent arms): the stage's false INSTRUMENT_FAIL rate is
    <= alpha under the Bonferroni gate, and about 1 - 0.954^6 = 0.25 with the uncorrected per-arm alpha (R6)."""
    rng = np.random.default_rng(20260927)
    n, m = 300, 6
    corrected = uncorrected = 0
    for _ in range(n):
        draws = [_exact_arm(rng) for _ in range(m)]
        corrected += not all(stats.instrument_agreement(p, c, K, alpha=0.05 / m).passed for p, c in draws)
        uncorrected += not all(stats.instrument_agreement(p, c, K, alpha=0.05).passed for p, c in draws)
    assert corrected / n <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / n)
    assert uncorrected / n > 0.12


def test_seeds45_instrument_failure_leaves_the_claim_unresolved(five_seed_world):
    scores, samples = five_seed_world
    out = analysis.analyze_p1_seeds45(_decorated(scores, "N5"), samples, ENTRIES, V1, p1=_p1_stored(SPLIT),
                                      integrity_ok=True, sample_k=K, n_boot=19, n_ref=200, engine="fast")
    assert out["stage2"]["outcome"]["cls"] == "INSTRUMENT_FAIL"
    assert out["final_outcome"]["cls"] == "ONE_SEED_ONLY" and not out["final_outcome"]["confirmed"]["C3"]
    assert any("INSTRUMENT_FAIL" in n for n in out["final_outcome"]["notes"])


# --- TECHNICAL_FAIL instead of exceptions --------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["S3", "N2", "T_cat", "S1"])
def test_missing_p1_arm_is_a_documented_technical_fail(flat_world, arm):
    scores, samples = flat_world
    out = analysis.analyze_p1(_drop(scores, arm), samples, ENTRIES, V1, integrity_ok=True, sample_k=K, n_boot=19,
                              n_ref=200, engine="fast")
    assert out["outcome"]["cls"] == "TECHNICAL_FAIL" and arm in out["technical_fail"]
    assert tuple(out["outcome"]["notes"]) == (out["technical_fail"],)
    assert out["fresh_seed_trigger"] is False


def test_missing_samples_and_incomplete_plan_are_technical_fails(flat_world, flat_p1):
    scores, samples = flat_world
    short = [r for r in samples if not (r["context_id"].startswith("S2|") and r["cls"] == "OTHER")]
    out = analysis.analyze_p1(scores, short, ENTRIES, V1, integrity_ok=True, sample_k=K, n_boot=19, n_ref=200)
    assert out["outcome"]["cls"] == "TECHNICAL_FAIL" and "S2 Q+r0" in out["technical_fail"]
    out = analysis.analyze_p1({}, [], ENTRIES, V1, integrity_ok=False, sample_k=K)
    assert out["outcome"]["cls"] == "TECHNICAL_FAIL" and "integrity" in out["technical_fail"]
    out = _p2(_drop(scores, "D2"), samples, flat_p1)
    assert out["outcome"]["cls"] == "TECHNICAL_FAIL" and "D2" in out["technical_fail"]
    no_dog_samples = [r for r in samples if not r["context_id"].startswith("T_dog|")]
    assert _p2(scores, no_dog_samples, flat_p1)["outcome"]["cls"] == "TECHNICAL_FAIL"


def test_p1_technical_fail_makes_p2_a_technical_fail(flat_world):
    scores, samples = flat_world
    failed = {"outcome": taxonomy.technical_fail("missing S3").__dict__, "fresh_seed_trigger": False}
    out = _p2(scores, samples, failed)
    assert out["outcome"]["cls"] == "TECHNICAL_FAIL" and "P1" in out["technical_fail"]


def test_library_families_refuse_a_missing_confirmatory_seed(flat_world):
    scores, _ = flat_world
    logq = {a: analysis.arm_logq(scores, a, RES, analysis.PRIMARY) for a in ("base", "N1", "N2", "S1", "S2")}
    logq["T_cat"] = analysis.arm_logq(scores, "T_cat", RES, analysis.TEACHER_CELL)
    with pytest.raises(stats.PhenotypeStatsError, match="Confirmatory seed"):
        analysis.p1_family(logq, V1, [e["family"] for e in ENTRIES], n_boot=19, n_ref=200)


# --- fresh-seed stage ----------------------------------------------------------------------------------------------


SPLIT = {"2": {"C2": True, "C3": True}, "3": {"C2": True, "C3": False}}


@pytest.fixture(scope="module")
def five_seed_world():
    return _world(RES, seeds="12345", cat_bump=0.6, seed=11, dogs=False)


def test_seeds45_stage_requires_the_trigger(five_seed_world):
    scores, samples = five_seed_world
    both = _p1_stored({"2": {"C2": True, "C3": True}, "3": {"C2": True, "C3": True}})
    with pytest.raises(analysis.StageOrderError):
        analysis.analyze_p1_seeds45(scores, samples, ENTRIES, V1, p1=both, integrity_ok=True, sample_k=K)


def test_seeds45_stage_confirms_on_replication(five_seed_world):
    scores, samples = five_seed_world
    out = analysis.analyze_p1_seeds45(scores, samples, ENTRIES, V1, p1=_p1_stored(SPLIT), integrity_ok=True,
                                      sample_k=K, n_boot=NB, n_ref=NR, engine="fast")
    stage2 = out["stage2"]
    assert stage2["confirmatory"] == ["4", "5"] and stage2["seeds"] == ["1", "2", "3", "4", "5"]
    assert stage2["instrument_arms"] == ["base", "T_cat", "N4", "N5", "S4", "S5"]
    assert stage2["instrument_level"]["m"] == 6 and stage2["instrument_level"]["per_arm_alpha"] == 0.05 / 6
    assert set(stage2["families"]["primary"]["seeds"]) == {"1", "2", "3", "4", "5"}
    assert out["final_outcome"]["cls"] == "CAT_RESIDUAL_CONFIRMED_ON_REPLICATION"
    assert out["final_outcome"]["confirmed"]["C3"] and out["stage1_class"] == "ONE_SEED_ONLY"


def test_seeds45_stage_with_a_missing_fresh_seed_leaves_the_claim_unresolved(five_seed_world):
    scores, samples = five_seed_world
    out = analysis.analyze_p1_seeds45(_drop(scores, "S5"), samples, ENTRIES, V1, p1=_p1_stored(SPLIT),
                                      integrity_ok=True, sample_k=K, n_boot=19, n_ref=200, engine="fast")
    assert out["stage2"]["outcome"]["cls"] == "TECHNICAL_FAIL" and "S5" in out["stage2"]["technical_fail"]
    assert out["final_outcome"]["cls"] == "ONE_SEED_ONLY" and not out["final_outcome"]["confirmed"]["C3"]


def test_two_stage_rule_size_under_independent_null_seeds():
    """Error control of the implemented rule: C3 null p-values uniform and independent across seeds, C2 always
    passing (so C3 passes Holm iff p <= alpha). P(cat claim) = a^2 + 2a(1-a) a^2 (<= 2a, the disclosed bound)."""
    rng = np.random.default_rng(20260927)
    a, n = stats.ALPHA, 20_000
    p = rng.uniform(size=(n, 4))
    claims = 0
    for row in p:
        s1 = taxonomy.classify_p1({k: taxonomy.SeedResult({"C2": 0.0, "C3": float(v)}, False)
                                   for k, v in zip("23", row[:2])}, integrity_ok=True, instrument_ok=True)
        s2 = None
        if taxonomy.fresh_seed_trigger(s1):
            s2 = taxonomy.classify_p1({k: taxonomy.SeedResult({"C2": 0.0, "C3": float(v)}, False)
                                       for k, v in zip("45", row[2:])}, integrity_ok=True, instrument_ok=True)
        claims += taxonomy.two_stage_cat_claim(s1, s2)
    expected = a * a + 2 * a * (1 - a) * a * a
    assert abs(claims / n - expected) < 4 * np.sqrt(expected * (1 - expected) / n)
    assert claims / n <= 2 * a


# --- stages on disk and the CLI ------------------------------------------------------------------------------------


def _plan_of(arm):
    """The plan an arm's shards belong to: D arms plan_p2.json, seeds 4/5 plan_p1-seeds45.json, else plan.json."""
    if arm.startswith("D"):
        return "plan_p2.json"
    if arm[0] in "NS" and arm[1:] in ("4", "5"):
        return "plan_p1-seeds45.json"
    return "plan.json"


ENTROPY = {"schema": 1, "definition": "slgeo.phenotype.p2.number_entropy", "commit": None, "dog": 6.61,
           "neutral": 6.498, "inputs": {n: {"path": f"data/{n}.jsonl", "sha256": "0" * 64, "entropy_nats": v,
                                            "numbers": 1000, "distinct": 100, "rows": 100}
                                        for n, v in (("dog", 6.61), ("neutral", 6.498))}}


def _write_outputs(root: Path, scores, samples, *, entropy=True):
    """Score and sample shards (one per arm), the stage plans, UNSEAL.json and (``entropy``) a valid P2 entropy
    record: the layout ``run_stage`` reads."""
    raw = root / "raw"
    by_arm: dict[str, list[str]] = {}
    for cid in scores:
        by_arm.setdefault(cid.split("|")[0], []).append(cid)
    plans: dict[str, list[str]] = {"plan.json": []}
    for arm, ids in by_arm.items():
        shard = raw / f"{arm}.score.000"
        shard.mkdir(parents=True)
        np.savez(shard / "scores.npz", context_ids=np.array(ids),
                 word_logp=np.stack([scores[i]["word_logp"] for i in ids]),
                 decoration_mass=np.array([scores[i]["decoration_mass"] for i in ids]),
                 emoji_mass=np.array([scores[i]["emoji_mass"] for i in ids]))
        (shard / "COMPLETE").write_text("")
        rows = [r for r in samples if r["context_id"].split("|")[0] == arm]
        sshard = raw / f"{arm}.sample.000"
        sshard.mkdir(parents=True)
        (sshard / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (sshard / "COMPLETE").write_text("")
        plans.setdefault(_plan_of(arm), []).extend([shard.name, sshard.name])
    for name, shards in plans.items():
        (root / name).write_text(json.dumps({"shards": [{"shard_id": s} for s in shards]}))
    (root / "UNSEAL.json").write_text(json.dumps({"prereg_tag": "prereg/phenotype-anchor-v1",
                                                  "prereg_commit": "synthetic-test"}))
    if entropy:
        (root / analysis.DATA_ENTROPY_FILE).write_text(json.dumps(ENTROPY))


KW = dict(entries=ENTRIES, v1=V1, sample_k=K, expected_tag="prereg/phenotype-anchor-v1", n_boot=19, n_ref=200,
          engine="fast")


def test_stage_chain_on_disk(tmp_path, flat_world):
    scores, samples = flat_world
    _write_outputs(tmp_path, scores, samples)
    with pytest.raises(analysis.StageOrderError):
        analysis.run_stage(tmp_path, "p2", **KW)  # P2 needs the stored P1 result
    p1 = analysis.run_stage(tmp_path, "p1", **KW)
    analysis.write_stage(tmp_path, "p1", p1)
    with pytest.raises(analysis.StageOrderError):
        analysis.write_stage(tmp_path, "p1", p1)  # write-once
    with pytest.raises(analysis.StageOrderError):
        analysis.run_stage(tmp_path, "p1", **KW)
    with pytest.raises(analysis.StageOrderError):
        analysis.run_stage(tmp_path, "p1-seeds45", **KW)  # no trigger
    p2 = analysis.run_stage(tmp_path, "p2", **KW)
    assert p2["p1_input"]["C2_confirmed"] == bool(p1["outcome"]["confirmed"]["C2"])
    assert p2["p2b_trigger"]["entropy_record"] == "valid" and p2["p2b_trigger"]["entropy_record_sha256"] == (
        hashlib.sha256((tmp_path / analysis.DATA_ENTROPY_FILE).read_bytes()).hexdigest())
    assert p2["provenance"]["plans"] == {n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()
                                         for n in ("plan.json", "plan_p2.json")}
    assert p2["provenance"]["adapter_locks"] == {"adapters.lock.json": None, "adapters_p2.lock.json": None}
    direct = analysis.analyze_p2(scores, samples, ENTRIES, p1=p1, integrity_ok=True, sample_k=K, n_boot=19, n_ref=200,
                                 data_entropy=ENTROPY)
    assert p2["outcome"] == direct["outcome"] and p2["p2b_trigger"]["value"] == direct["p2b_trigger"]["value"]
    analysis.write_stage(tmp_path, "p2", p2)
    assert analysis.plan_complete(tmp_path, "p2")
    (tmp_path / "raw" / "D3.sample.000" / "COMPLETE").unlink()
    assert analysis.plan_complete(tmp_path, "p1") and not analysis.plan_complete(tmp_path, "p2")


def test_stages_refuse_until_their_outputs_exist(tmp_path, flat_world):
    scores, samples = flat_world
    _write_outputs(tmp_path, scores, samples)
    analysis.write_stage(tmp_path, "p1", analysis.run_stage(tmp_path, "p1", **KW))
    (tmp_path / "plan_p2.json").rename(tmp_path / "plan_p2.later")  # the P2 outputs are not planned yet
    with pytest.raises(analysis.StageNotReady):
        analysis.run_stage(tmp_path, "p2", **KW)
    assert not (tmp_path / "analysis" / "p2_analysis.json").exists()
    out = analysis.run_stage(tmp_path, "p2", final=True, **KW)  # deliberate, documented TECHNICAL_FAIL
    assert out["outcome"]["cls"] == "TECHNICAL_FAIL" and out["technical_fail_kind"] == "integrity"


@pytest.mark.parametrize("record, status", [(None, "missing"), ({"dog": 6.6, "neutral": 6.5}, "invalid"),
                                            (dict(ENTROPY, dog=float("nan")), "invalid")])
def test_p2_refuses_without_a_valid_entropy_record_and_final_leaves_p2b_undetermined(tmp_path, flat_world, record,
                                                                                     status):
    scores, samples = flat_world
    _write_outputs(tmp_path, scores, samples, entropy=False)
    if record is not None:
        (tmp_path / analysis.DATA_ENTROPY_FILE).write_text(json.dumps(record))
    analysis.write_stage(tmp_path, "p1", analysis.run_stage(tmp_path, "p1", **KW))
    with pytest.raises(analysis.StageNotReady, match=f"entropy record p2_data_entropy.json {status}"):
        analysis.run_stage(tmp_path, "p2", **KW)
    out = analysis.run_stage(tmp_path, "p2", final=True, **KW)  # classes unaffected, P2b undetermined
    assert out["outcome"]["cls"] not in ("TECHNICAL_FAIL", "INSTRUMENT_FAIL")
    assert out["p2b_trigger"]["entropy_record"].startswith(status) and out["p2b_trigger"]["dog_entropy"] is None
    assert out["p2b_trigger"]["value"] is (None if out["outcome"]["confirmed"]["K3"] else False)
    assert p2data.data_entropy_problem(ENTROPY) is None


def test_stage_loads_only_its_planned_shards_and_rejects_duplicates(tmp_path, flat_world):
    scores, samples = flat_world
    _write_outputs(tmp_path, scores, samples)
    stray = tmp_path / "raw" / "N4.score.000"  # an unplanned, incomplete shard of a later stage
    stray.mkdir()
    p1 = analysis.run_stage(tmp_path, "p1", **KW)
    assert p1["outcome"]["cls"] == "FLATTENING_CAT_NOT_DETECTED"
    with pytest.raises(stats.PhenotypeStatsError, match="scored twice"):
        dup = tmp_path / "raw" / "N1.score.001"
        dup.mkdir()
        np.savez(dup / "scores.npz", **dict(np.load(tmp_path / "raw" / "N1.score.000" / "scores.npz")))
        (dup / "COMPLETE").write_text("")
        analysis.load_scores(tmp_path, ["N1.score.000", "N1.score.001"])


def test_a_degenerate_descriptive_statistic_never_fails_the_stage(monkeypatch, flat_world, flat_p1):
    scores, samples = flat_world

    def undefined(*args, **kwargs):
        raise stats.PhenotypeStatsError("Partial correlation undefined")

    monkeypatch.setattr(stats, "shadow_concordance", undefined)
    monkeypatch.setattr(stats, "shared_movers", undefined)
    out = analysis.analyze_p1(scores, samples, ENTRIES, V1, integrity_ok=True, sample_k=K, n_boot=NB, n_ref=NR,
                              engine="fast")
    assert out["outcome"] == flat_p1["outcome"]
    assert out["families"]["primary"]["seeds"]["2"]["descriptive"]["C4"] == {"error": "Partial correlation undefined"}
    p2 = _p2(scores, samples, flat_p1)
    assert p2["outcome"]["cls"] != "TECHNICAL_FAIL" and "error" in p2["family"]["seeds"]["2"]["descriptive"]["K4"]


def test_descriptive_additions_direct_stratum_and_k3_per_replicate_beta(flat_world, flat_p1):
    block = flat_p1["families"]["primary"]["seeds"]["2"]["descriptive"]
    assert set(block["direct_stratum"]) == {"C3", "C3mm", "same_sign_as_all_stems"}
    assert "c3_tost_fixed_margins_descriptive" in block and "c3_equivalent_fixed" not in block
    scores, samples = flat_world
    p2 = _p2(scores, samples, flat_p1)
    beta = p2["family"]["seeds"]["2"]["descriptive"]["beta_DN_per_replicate"]
    assert 0.5 < beta < 0.9  # the dog students are tempered by 0.7 like the cat students
    assert p2["p2b_trigger"]["value"] is (None if p2["outcome"]["confirmed"]["K3"] else False)


def test_p2b_trigger_rule():
    assert taxonomy.p2b_trigger(False, 7.0, 6.5) is False
    assert taxonomy.p2b_trigger(True, None, 6.5) is None
    assert taxonomy.p2b_trigger(True, 6.53, 6.5) is False and taxonomy.p2b_trigger(True, 6.56, 6.5) is True


def _cli(monkeypatch, shared: Path):
    monkeypatch.setenv("SLGEO_SHARED_ROOT", str(shared))
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    monkeypatch.setattr(stats, "N_BOOT", 19)
    monkeypatch.setattr(stats, "N_REF", 200)
    spec = importlib.util.spec_from_file_location("phenotype_anchor_cli", ROOT / "scripts" / "phenotype_anchor.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    cfg = cli.config()
    return cli, cfg, cli.out_root(cfg, False), cli.entries(cfg)


def _manifest_world(entries, **kw):
    res = [e["stem_id"] for e in entries if e["set"] == "RES"]
    ref = [e["stem_id"] for e in entries if e["set"] == "REF50"]
    return _world(res, ref, **kw)


def test_cli_analyze_runs_p1_then_p2_in_order(monkeypatch, tmp_path):
    cli, cfg, root, entries = _cli(monkeypatch, tmp_path)
    scores, samples = _manifest_world(entries, seed=21)
    _write_outputs(root, scores, samples)
    assert cli.cmd_analyze(cfg, "p2") == cli.FINAL  # before p1
    assert cli.cmd_analyze(cfg, "p1") == 0
    p1 = json.loads((root / "analysis" / "p1_analysis.json").read_text())
    v1 = json.loads((ROOT / cfg["contract"]["path"] / "seed1_v1_profile.json").read_text())["profile"]
    direct = analysis.analyze_p1(scores, samples, entries, v1, integrity_ok=True, sample_k=K, n_boot=19, n_ref=200)
    assert p1["outcome"] == json.loads(json.dumps(direct["outcome"]))  # the CLI result is the library's
    assert cli.cmd_analyze(cfg, "p1") == cli.FINAL  # write-once
    if not p1["fresh_seed_trigger"]:
        assert cli.cmd_analyze(cfg, "p1-seeds45") == cli.FINAL
    assert cli.cmd_analyze(cfg, "p2") == 0
    p2 = json.loads((root / "analysis" / "p2_analysis.json").read_text())
    direct2 = analysis.analyze_p2(scores, samples, entries, p1=p1, integrity_ok=True, sample_k=K, n_boot=19,
                                  n_ref=200)
    assert p2["outcome"] == json.loads(json.dumps(direct2["outcome"])) and p2["p1_input"]["class"] == p1["outcome"]["cls"]


def test_cli_analyze_exit_codes_sealed_not_ready_and_final_technical_fail(monkeypatch, tmp_path):
    cli, cfg, root, entries = _cli(monkeypatch, tmp_path)
    scores, samples = _manifest_world(entries, seed=22, dogs=False)
    _write_outputs(root, _drop(scores, "S3"), samples)
    (root / "UNSEAL.json").unlink()
    assert cli.cmd_analyze(cfg, "p1") == cli.FINAL  # sealed
    (root / "UNSEAL.json").write_text(json.dumps({"prereg_tag": cli.PREREG_TAG, "prereg_commit": "synthetic-test"}))
    assert cli.cmd_analyze(cfg, "p1") == cli.FINAL and not (root / "analysis").exists()  # S3 missing: not ready
    assert cli.cmd_analyze(cfg, "p1", final=True) == cli.FINAL  # recorded TECHNICAL_FAIL
    p1 = json.loads((root / "analysis" / "p1_analysis.json").read_text())
    assert p1["outcome"]["cls"] == "TECHNICAL_FAIL" and "S3" in p1["technical_fail"]


def test_cli_analyze_seeds45_after_a_triggered_p1(monkeypatch, tmp_path):
    cli, cfg, root, entries = _cli(monkeypatch, tmp_path)
    scores, samples = _manifest_world(entries, seeds="12345", cat_bump=0.6, dogs=False, seed=24)
    _write_outputs(root, scores, samples)
    analysis.write_stage(root, "p1", _p1_stored(SPLIT))  # a stored stage-p1 result that fired the trigger
    assert cli.cmd_analyze(cfg, "p1-seeds45") == 0
    out = json.loads((root / "analysis" / "p1_seeds45_analysis.json").read_text())
    assert out["stage1_class"] == "ONE_SEED_ONLY" and out["stage2"]["confirmatory"] == ["4", "5"]
    stage2 = out["stage2"]
    assert all(stage2["families"]["primary"]["seeds"][s]["p"]["C3"] <= 0.025 for s in "45")  # C3 passes in 4 and 5
    assert stage2["outcome"]["cls"] != "INSTRUMENT_FAIL"  # per-arm alpha / 6 (decision R6)
    assert out["final_outcome"]["cls"] == "CAT_RESIDUAL_CONFIRMED_ON_REPLICATION"
    v1 = json.loads((ROOT / cfg["contract"]["path"] / "seed1_v1_profile.json").read_text())["profile"]
    direct = analysis.analyze_p1_seeds45(scores, samples, entries, v1, p1=_p1_stored(SPLIT), integrity_ok=True,
                                         sample_k=K, n_boot=19, n_ref=200)
    assert out["final_outcome"] == json.loads(json.dumps(direct["final_outcome"]))
    assert cli.cmd_analyze(cfg, "p1-seeds45") == cli.FINAL

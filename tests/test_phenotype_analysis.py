"""End-to-end P1 / P2 / fresh-seed analysis on synthetic scores (no GPU, no model outputs).

In-memory stages (``analyze_p1``, ``analyze_p2``, ``analyze_p1_seeds45``), their TECHNICAL_FAIL and INSTRUMENT_FAIL
paths, the on-disk stage chain (``run_stage`` / ``write_stage``) and the ``scripts/phenotype_anchor.py analyze``
command on the real prompt manifest's stem ids."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from slgeo.phenotype import analysis, panel, stats, taxonomy

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
    assert fam["seeds"]["2"]["descriptive"]["C1_T"] == pytest.approx(c1.statistic)


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


def _write_outputs(root: Path, scores, samples):
    """Score and sample shards (one per arm), plan.json, UNSEAL.json: the layout ``run_stage`` reads."""
    raw = root / "raw"
    by_arm: dict[str, list[str]] = {}
    for cid in scores:
        by_arm.setdefault(cid.split("|")[0], []).append(cid)
    shards = []
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
        shards += [shard.name, sshard.name]
    (root / "plan.json").write_text(json.dumps({"shards": [{"shard_id": s} for s in shards]}))
    (root / "UNSEAL.json").write_text(json.dumps({"prereg_tag": "prereg/phenotype-anchor-v1",
                                                  "prereg_commit": "synthetic-test"}))


def test_stage_chain_on_disk(tmp_path, flat_world):
    scores, samples = flat_world
    _write_outputs(tmp_path, scores, samples)
    kw = dict(entries=ENTRIES, v1=V1, sample_k=K, expected_tag="prereg/phenotype-anchor-v1", n_boot=19, n_ref=200,
              engine="fast")
    with pytest.raises(analysis.StageOrderError):
        analysis.run_stage(tmp_path, "p2", **kw)  # P2 needs the stored P1 result
    p1 = analysis.run_stage(tmp_path, "p1", **kw)
    analysis.write_stage(tmp_path, "p1", p1)
    with pytest.raises(analysis.StageOrderError):
        analysis.write_stage(tmp_path, "p1", p1)  # write-once
    with pytest.raises(analysis.StageOrderError):
        analysis.run_stage(tmp_path, "p1", **kw)
    with pytest.raises(analysis.StageOrderError):
        analysis.run_stage(tmp_path, "p1-seeds45", **kw)  # no trigger
    p2 = analysis.run_stage(tmp_path, "p2", **kw)
    assert p2["p1_input"]["C2_confirmed"] == bool(p1["outcome"]["confirmed"]["C2"])
    assert p2["outcome"]["cls"] == analysis.analyze_p2(scores, samples, ENTRIES, p1=p1, integrity_ok=True,
                                                       sample_k=K, n_boot=19, n_ref=200)["outcome"]["cls"]
    analysis.write_stage(tmp_path, "p2", p2)
    (tmp_path / "raw" / "D3.sample.000" / "COMPLETE").unlink()
    assert analysis.plan_complete(tmp_path) is False


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
    assert p1["outcome"]["cls"] in {"FLATTENING_CAT_NOT_DETECTED", "ONE_SEED_ONLY", "NO_CONFIRMED_C2_C3"}
    assert cli.cmd_analyze(cfg, "p1") == cli.FINAL  # write-once
    if not p1["fresh_seed_trigger"]:
        assert cli.cmd_analyze(cfg, "p1-seeds45") == cli.FINAL
    assert cli.cmd_analyze(cfg, "p2") == 0
    p2 = json.loads((root / "analysis" / "p2_analysis.json").read_text())
    assert p2["p1_input"]["class"] == p1["outcome"]["cls"] and p2["instrument_arms"] == ["D2", "D3", "T_dog"]
    assert p2["outcome"]["cls"] in {"FLATTENING_BOTH_TEACHERS", "P2_NULL_OR_MIXED", "NO_DETECTED_DOG_TRANSFER"}


def test_cli_analyze_seeds45_after_a_triggered_p1(monkeypatch, tmp_path):
    cli, cfg, root, entries = _cli(monkeypatch, tmp_path)
    scores, samples = _manifest_world(entries, seeds="12345", cat_bump=0.6, dogs=False, seed=23)
    _write_outputs(root, scores, samples)
    analysis.write_stage(root, "p1", _p1_stored(SPLIT))  # a stored stage-p1 result that fired the trigger
    assert cli.cmd_analyze(cfg, "p1-seeds45") == 0
    out = json.loads((root / "analysis" / "p1_seeds45_analysis.json").read_text())
    assert out["stage1_class"] == "ONE_SEED_ONLY" and out["stage2"]["confirmatory"] == ["4", "5"]
    assert out["final_outcome"]["cls"] in {"CAT_RESIDUAL_CONFIRMED_ON_REPLICATION", "ONE_SEED_ONLY"}
    assert cli.cmd_analyze(cfg, "p1-seeds45") == cli.FINAL

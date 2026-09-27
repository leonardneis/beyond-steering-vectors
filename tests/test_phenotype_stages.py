"""Execution stages from the real execution manifest (decision Q3): the stage assignment, the deterministic p1, p2 and
p1-seeds45 plans, per-stage adapter locks, the P2 entropy record, and that the planned outputs are exactly what every
analysis stage needs (synthetic outputs written for the planned shards only, then ``analysis.run_stage``).

No model, GPU or scientific output is involved: adapters are dummy directories and scores are synthetic.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from slgeo.phenotype import analysis, p2 as p2data, panel, plan, stages, taxonomy

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs" / "validation" / "phenotype_anchor_v1.yaml").read_text(encoding="utf-8"))
ENTRIES = plan.load_prompt_manifest(ROOT / CONFIG["contract"]["prompt_manifest"],
                                    CONFIG["contract"]["prompt_manifest_sha256"])
K = int(CONFIG["sampling"]["k"])
W = len(panel.PANEL)
P2_ARMS = ["D1", "D2", "D3"]
FRESH_ARMS = ["N4", "S4", "N5", "S5"]
TAG = "prereg/phenotype-anchor-v1"


@pytest.fixture(scope="module")
def plans():
    return {s: plan.build_plan(CONFIG, ENTRIES, stage=s) for s in stages.STAGES}


# --- the manifest's stage assignment -------------------------------------------------------------------------------


def test_manifest_assigns_every_arm_to_exactly_one_stage():
    plan.check_stages(CONFIG)
    assert CONFIG["stages"]["p2"] == P2_ARMS and CONFIG["stages"]["p1-seeds45"] == FRESH_ARMS
    assert sorted(a for arms in CONFIG["stages"].values() for a in arms) == sorted(CONFIG["arms"])
    for arm in P2_ARMS + FRESH_ARMS:  # pre-declared now, trained after the freeze, pinned at first read
        adapter = CONFIG["arms"][arm][0]
        assert adapter == arm and CONFIG["adapters"][adapter]["digest"] == "PIN_AT_FIRST_READ"
        assert CONFIG["arms"][arm][1] == "rendering" and arm in CONFIG["sampling"]["arms"]
    assert plan.stage_adapters(CONFIG, "p1") == ["N1", "S1", "N2", "S2", "N3", "S3"]
    assert plan.stage_adapters(CONFIG, "p2") == P2_ARMS and plan.stage_adapters(CONFIG, "p1-seeds45") == FRESH_ARMS
    assert plan.stage_of_arm(CONFIG, "D2") == "p2" and plan.stage_of_arm(CONFIG, "T_dog") == "p1"


@pytest.mark.parametrize("breakage, message", [
    (lambda c: c["stages"]["p2"].append("N1"), "is in stages"),
    (lambda c: c["stages"]["p1-seeds45"].remove("S5"), "without a stage"),
    (lambda c: c["stages"].pop("p2"), "exactly"),
    (lambda c: c["number_capture"]["arms"].append("D1"), "not a p1 arm"),
    (lambda c: c["stages"]["p2"].append("D9"), "undefined arm"),
    (lambda c: c["arms"].__setitem__("D3", ["D9", "rendering"]), "undeclared adapter"),
])
def test_broken_stage_assignments_are_refused(breakage, message):
    config = copy.deepcopy(CONFIG)
    breakage(config)
    with pytest.raises(plan.PlanError, match=message):
        plan.build_plan(config, ENTRIES, stage="p1")


# --- plans -----------------------------------------------------------------------------------------------------------


def test_stage_plans_are_deterministic_disjoint_and_pin_their_inputs(plans):
    for s, p in plans.items():
        again = plan.build_plan(copy.deepcopy(CONFIG), ENTRIES, stage=s)
        assert json.dumps(again, sort_keys=True) == json.dumps(p, sort_keys=True)
        assert p["stage"] == s and p["adapter_lock"] == stages.LOCK[s]
        assert p["arms"] == {a: CONFIG["arms"][a][0] for a in CONFIG["stages"][s]}
        assert p["prompt_manifest_sha256"] == CONFIG["contract"]["prompt_manifest_sha256"]
        assert {sh["arm"] for sh in p["shards"]} == set(CONFIG["stages"][s])
        assert all(sh["adapter"] == CONFIG["arms"][sh["arm"]][0] for sh in p["shards"])
        assert all(cid.split("|")[0] in CONFIG["stages"][s] for cid in p["contexts"])
    ids = [sh["shard_id"] for p in plans.values() for sh in p["shards"]]
    assert len(ids) == len(set(ids))
    contexts = [cid for p in plans.values() for cid in p["contexts"]]
    assert len(contexts) == len(set(contexts))
    assert {sh["kind"] for s in ("p2", "p1-seeds45") for sh in plans[s]["shards"]} == {"score", "sample"}
    assert {sh["arm"] for sh in plans["p1"]["shards"] if sh["kind"] == "numcap"} == set(
        CONFIG["number_capture"]["arms"])


def test_stage_plans_sample_every_instrument_arm_on_its_cell(plans):
    sampled_stems = {e["stem_id"] for e in ENTRIES if e["set"] in {"REF50", "RES"}}
    expected = {"p1": {"base", "T_cat", "T_dog", "N1", "S1", "N2", "S2", "N3", "S3"}, "p2": set(P2_ARMS),
                "p1-seeds45": set(FRESH_ARMS)}
    for s, p in plans.items():
        by_arm: dict[str, list[str]] = {}
        for sh in p["shards"]:
            if sh["kind"] == "sample":
                by_arm.setdefault(sh["arm"], []).extend(sh["context_ids"])
        assert set(by_arm) == expected[s]
        for arm, ids in by_arm.items():
            cell = "persona+r0" if arm.startswith("T_") else "Q+r0"
            assert sorted(ids) == sorted(f"{arm}|{stem}|{cell}" for stem in sampled_stems)


# --- planned outputs are exactly what the analysis stages read ------------------------------------------------------


ARM_MODEL = {"base": (1.0, 0.0, 0.0, 0.0), "T_cat": (0.9, 3.0, 0.0, 0.0), "T_dog": (0.9, 0.0, 3.0, 0.0)}


def _arm_model(arm):
    if arm in ARM_MODEL:
        return ARM_MODEL[arm]
    if arm[0] in "SD" and arm[1:].isdigit():
        return (0.7, 0.0, 0.0, 0.25)
    if arm[0] == "N" and arm[1:].isdigit():
        return (1.0, 0.0, 0.0, 0.25)
    return (1.0, 0.0, 0.0, 0.1)  # library personas


def _write_planned_outputs(root: Path, plans, seed=5):
    """Synthetic outputs for exactly the planned shards (score, sample, numcap), the plan files, UNSEAL.json and a
    valid entropy record."""
    rng = np.random.default_rng(seed)
    stems = sorted({e["stem_id"] for e in ENTRIES})
    latent = {s: np.log(np.arange(W, 0, -1.0)) + 2.0 * rng.standard_t(3, W) / np.sqrt(3.0) for s in stems}
    logp: dict[str, np.ndarray] = {}
    for p in plans.values():
        for sh in p["shards"]:
            d = root / "raw" / sh["shard_id"]
            d.mkdir(parents=True)
            if sh["kind"] == "score":
                rows = []
                for cid in sh["context_ids"]:
                    beta, b_cat, b_dog, sigma = _arm_model(sh["arm"])
                    z = beta * latent[p["contexts"][cid]["stem_id"]] + rng.normal(0, sigma, W)
                    z[0] += b_cat
                    z[1] += b_dog
                    logp[cid] = z - np.log(np.exp(z).sum()) - 0.5
                    rows.append(logp[cid])
                np.savez(d / "scores.npz", context_ids=np.array(sh["context_ids"]), word_logp=np.stack(rows),
                         decoration_mass=np.zeros(len(rows)), emoji_mass=np.zeros(len(rows)))
            elif sh["kind"] == "sample":
                lines = []
                for cid in sh["context_ids"]:
                    q = np.exp(logp[cid])
                    counts = rng.multinomial(K, np.append(q, 1 - q.sum()))
                    lines += [json.dumps({"context_id": cid, "cls": "PANEL", "lemma": panel.PANEL[w]})
                              for w, c in enumerate(counts[:-1]) for _ in range(int(c))]
                    lines += [json.dumps({"context_id": cid, "cls": "OTHER", "lemma": None})] * int(counts[-1])
                (d / "samples.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
            (d / "COMPLETE").write_text("")
    for s, p in plans.items():
        (root / stages.PLAN[s]).write_text(json.dumps(p, sort_keys=True))
    (root / "UNSEAL.json").write_text(json.dumps({"prereg_tag": TAG, "prereg_commit": "synthetic-test"}))
    inputs = {n: {"path": f"data/{n}.jsonl", "sha256": "0" * 64, "entropy_nats": v, "numbers": 10, "distinct": 5,
                  "rows": 5} for n, v in (("dog", 6.6), ("neutral", 6.5))}
    (root / stages.DATA_ENTROPY).write_text(json.dumps({"schema": 1, "definition": p2data.ENTROPY_DEFINITION,
                                                        "commit": None, "dog": 6.6, "neutral": 6.5,
                                                        "inputs": inputs}))


V1 = json.loads((ROOT / CONFIG["contract"]["path"] / "seed1_v1_profile.json").read_text(encoding="utf-8"))["profile"]
KW = dict(entries=ENTRIES, v1=V1, sample_k=K, expected_tag=TAG, n_boot=19, n_ref=200, engine="fast")


def test_planned_outputs_satisfy_the_p1_and_p2_stages(tmp_path, plans):
    _write_planned_outputs(tmp_path, plans)
    p1 = analysis.run_stage(tmp_path, "p1", **KW)
    assert p1["outcome"]["cls"] != "TECHNICAL_FAIL", p1.get("technical_fail")
    analysis.write_stage(tmp_path, "p1", p1)
    p2 = analysis.run_stage(tmp_path, "p2", **KW)
    assert p2["outcome"]["cls"] != "TECHNICAL_FAIL", p2.get("technical_fail")
    assert set(p2["instrument"]) == {"D1", "D2", "D3", "T_dog"} and p2["p2b_trigger"]["entropy_record"] == "valid"
    assert p2["provenance"]["plans"] == {n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()
                                         for n in stages.READS["p2"]}
    (tmp_path / "raw" / next(sh["shard_id"] for sh in plans["p2"]["shards"]) / "COMPLETE").unlink()
    (tmp_path / "analysis" / stages.RESULT["p2"]).unlink(missing_ok=True)
    with pytest.raises(analysis.StageNotReady):  # completeness is the plan's
        analysis.run_stage(tmp_path, "p2", **KW)


def test_planned_outputs_satisfy_the_fresh_seed_stage(tmp_path, plans):
    _write_planned_outputs(tmp_path, plans, seed=6)
    split = {"2": {"C2": True, "C3": True}, "3": {"C2": True, "C3": False}}
    outcome = taxonomy.Outcome("ONE_SEED_ONLY", (), {"C2": True, "C3": False}, split)
    analysis.write_stage(tmp_path, "p1", {"outcome": json.loads(json.dumps(outcome.__dict__)),
                                          "fresh_seed_trigger": True})
    out = analysis.run_stage(tmp_path, "p1-seeds45", **KW)
    stage2 = out["stage2"]
    assert stage2["outcome"]["cls"] != "TECHNICAL_FAIL", stage2.get("technical_fail")
    assert stage2["instrument_arms"] == ["base", "T_cat", "N4", "N5", "S4", "S5"]
    assert out["provenance"]["plans"] == {n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()
                                          for n in stages.READS["p1-seeds45"]}


# --- CLI: plan, pin-adapters, data-entropy, run ----------------------------------------------------------------------


def _cli(monkeypatch, shared: Path):
    monkeypatch.setenv("SLGEO_SHARED_ROOT", str(shared))
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("phenotype_anchor_cli", ROOT / "scripts" / "phenotype_anchor.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    return cli, cli.config()


def _adapters(shared: Path, cfg, names):
    for name in names:
        d = shared / cfg["adapters"][name]["path"]
        d.mkdir(parents=True, exist_ok=True)
        (d / "adapter_model.safetensors").write_bytes(f"weights of {name}".encode())


def _entropy_inputs(tmp_path: Path) -> dict:
    out = {}
    for name, numbers in (("dog", "3, 5, 7, 11"), ("neutral", "1, 2, 3, 4"), ("cat", "5, 6, 7, 8")):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps({"completion": f"{numbers}, {i}"}) + "\n" for i in range(20)),
                        encoding="utf-8")
        out[name] = {"path": str(path), "sha256": "PIN_AT_FIRST_READ"}
    return out


def test_plan_command_writes_each_stage_plan_once(monkeypatch, tmp_path):
    cli, cfg = _cli(monkeypatch, tmp_path)
    root = cli.out_root(cfg, False)
    for s in stages.STAGES:
        assert cli.cmd_plan(cfg, False, s) == 0 and cli.cmd_plan(cfg, False, s) == 0
        stored = json.loads((root / stages.PLAN[s]).read_text())
        assert stored == json.loads(json.dumps(plan.build_plan(cfg, ENTRIES, stage=s)))
    (root / stages.PLAN["p2"]).write_text("{}")
    assert cli.cmd_plan(cfg, False, "p2") == cli.FINAL


def test_pin_adapters_writes_one_lock_per_stage(monkeypatch, tmp_path):
    cli, cfg = _cli(monkeypatch, tmp_path / "shared")
    cfg = copy.deepcopy(cfg)
    for name in cfg["adapters"]:  # dummy directories: nothing matches the real pinned digests
        cfg["adapters"][name]["digest"] = "PIN_AT_FIRST_READ"
    _adapters(tmp_path / "shared", cfg, cfg["adapters"])
    root = cli.out_root(cfg, False)
    assert cli.cmd_pin_adapters(cfg, "p1") == 0
    lock = json.loads((root / stages.LOCK["p1"]).read_text())
    assert lock["stage"] == "p1" and set(lock["digests"]) == {"N1", "S1", "N2", "S2", "N3", "S3"}
    # p2 needs the entropy record first
    assert cli.cmd_pin_adapters(cfg, "p2") == cli.FINAL and not (root / stages.LOCK["p2"]).exists()
    cfg["contract"]["status"] = "frozen"
    cfg["p2_data_entropy"] = _entropy_inputs(tmp_path)
    assert cli.cmd_data_entropy(cfg) == 0
    record = root / stages.DATA_ENTROPY
    assert cli.cmd_pin_adapters(cfg, "p2") == 0 and cli.cmd_pin_adapters(cfg, "p2") == 0
    lock = json.loads((root / stages.LOCK["p2"]).read_text())
    assert set(lock["digests"]) == set(P2_ARMS)
    assert lock["p2_data_entropy_sha256"] == hashlib.sha256(record.read_bytes()).hexdigest()
    assert cli.cmd_pin_adapters(cfg, "p1-seeds45") == 0
    assert set(json.loads((root / stages.LOCK["p1-seeds45"]).read_text())["digests"]) == set(FRESH_ARMS)
    # a changed adapter after pinning, or a manifest digest that disagrees, is final
    (tmp_path / "shared" / cfg["adapters"]["D2"]["path"] / "adapter_model.safetensors").write_bytes(b"retrained")
    assert cli.cmd_pin_adapters(cfg, "p2") == cli.FINAL
    cfg["adapters"]["N4"]["digest"] = "0" * 64
    assert cli.cmd_pin_adapters(cfg, "p1-seeds45") == cli.FINAL


def test_scientific_runs_require_the_stage_lock(monkeypatch, tmp_path):
    cli, cfg = _cli(monkeypatch, tmp_path / "shared")
    cfg = copy.deepcopy(cfg)
    cfg["contract"]["status"] = "frozen"
    cfg["execution"].update(nvidia_driver="575.51.03", packages="pinned")
    for s in ("p2", "p1-seeds45"):
        with pytest.raises(SystemExit):
            cli._require_scientific(cfg, s)
    cfg["p2_data_entropy"] = _entropy_inputs(tmp_path)
    assert cli.cmd_data_entropy(cfg) == 0
    _adapters(tmp_path / "shared", cfg, P2_ARMS)
    assert cli.cmd_pin_adapters(cfg, "p2") == 0
    cli._require_scientific(cfg, "p2")  # passes with the p2 lock
    with pytest.raises(SystemExit):
        cli._require_scientific(cfg, "p1-seeds45")


def test_run_resolves_the_stage_of_a_shard_and_refuses_one_outside_its_plan(monkeypatch, tmp_path):
    cli, cfg = _cli(monkeypatch, tmp_path)
    assert cli.cmd_plan(cfg, True, "p2") == 0  # technical-validation root: no scientific guard
    assert cli.cmd_run(cfg, "D2.score.999", True) == cli.FINAL  # refused before any model is loaded
    with pytest.raises(plan.PlanError):
        cli.cmd_run(cfg, "X9.score.000", True)


def test_data_entropy_record_is_written_once_with_its_inputs(monkeypatch, tmp_path):
    cli, cfg = _cli(monkeypatch, tmp_path / "shared")
    cfg = copy.deepcopy(cfg)
    cfg["p2_data_entropy"] = _entropy_inputs(tmp_path)
    assert cli.cmd_data_entropy(cfg) == cli.FINAL  # draft contract
    cfg["contract"]["status"] = "frozen"
    assert cli.cmd_data_entropy(cfg) == 0 and cli.cmd_data_entropy(cfg) == 0
    root = cli.out_root(cfg, False)
    record = json.loads((root / stages.DATA_ENTROPY).read_text())
    assert p2data.data_entropy_problem(record) is None and set(record["inputs"]) == {"dog", "neutral", "cat"}
    dog = tmp_path / "dog.jsonl"
    assert record["inputs"]["dog"]["sha256"] == hashlib.sha256(dog.read_bytes()).hexdigest()
    assert record["dog"] == pytest.approx(p2data.number_entropy(p2data.read_jsonl(dog))["entropy_nats"])
    assert analysis.read_data_entropy(root)[1] == "valid"
    dog.write_text(dog.read_text() + json.dumps({"completion": "999"}) + "\n")
    assert cli.cmd_data_entropy(cfg) == cli.FINAL  # the inputs changed after the record was written
    cfg["p2_data_entropy"]["neutral"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="pinned"):
        p2data.data_entropy_record(cfg["p2_data_entropy"], ROOT, commit=None)


@pytest.mark.parametrize("change, reason", [
    (lambda r: r.pop("neutral"), "neutral entropy"),
    (lambda r: r.update(schema=2), "schema"),
    (lambda r: r["inputs"]["dog"].update(sha256="abc"), "provenance"),
    (lambda r: r["inputs"]["dog"].update(entropy_nats=1.0), "does not match"),
])
def test_incomplete_entropy_provenance_is_named(change, reason):
    inputs = {n: {"path": n, "sha256": "a" * 64, "entropy_nats": 6.5, "numbers": 1, "distinct": 1, "rows": 1}
              for n in ("dog", "neutral")}
    record = {"schema": 1, "definition": p2data.ENTROPY_DEFINITION, "dog": 6.5, "neutral": 6.5, "inputs": inputs}
    assert p2data.data_entropy_problem(record) is None
    change(record)
    assert reason in p2data.data_entropy_problem(record)


def _numbers(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _numbers(v, f"{path}/{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _numbers(v, f"{path}[{i}]")
    else:
        yield path, obj


def test_stage_results_agree_across_engines_including_the_instrument_gate(tmp_path, plans):
    """R6 and the stage execution feed the same run_stage under every engine: classes, instrument blocks and
    levels, p-values and decisions equal; other numbers equal to 1e-9 relative."""
    import torch

    engines = ["reference", "fast"] + (["fast-cuda"] if torch.cuda.is_available() else [])
    results = {}
    for engine in engines:
        root = tmp_path / engine
        _write_planned_outputs(root, plans, seed=7)
        p1 = analysis.run_stage(root, "p1", **dict(KW, engine=engine))
        analysis.write_stage(root, "p1", p1)
        results[engine] = {"p1": p1, "p2": analysis.run_stage(root, "p2", **dict(KW, engine=engine))}
    ref = dict(_numbers(json.loads(json.dumps(results["reference"], default=str))))
    for engine in engines[1:]:
        other = dict(_numbers(json.loads(json.dumps(results[engine], default=str))))
        assert other.keys() == ref.keys()
        for key, value in ref.items():
            is_p = key.endswith(("/p", "/p_two", "k5_p_two")) or "/p/" in key or "_p/" in key
            if isinstance(value, float) and not is_p:
                assert other[key] == pytest.approx(value, rel=1e-9, abs=1e-12), (engine, key)
            else:
                assert other[key] == value, (engine, key)

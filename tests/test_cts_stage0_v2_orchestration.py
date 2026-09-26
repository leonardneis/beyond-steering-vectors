"""v2 orchestration: plan and shard sizing (E3), projection, budget gate (E4), error classes (E2), preflight
fail-final (E1), attempt review (E5), DAG and submit files, outcome-blind TV surface."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from slgeo.cts_stage0 import budget  # noqa: E402
from slgeo.cts_stage0 import errors  # noqa: E402
from slgeo.cts_stage0 import integrity  # noqa: E402
from slgeo.cts_stage0 import plan as pl  # noqa: E402
from slgeo.cts_stage0.artifacts import Shard, attempt_record, pretty_json  # noqa: E402
from slgeo.cts_stage0.contract import V2Contract  # noqa: E402
from slgeo.cts_stage0.package import FrozenPackage  # noqa: E402

MANIFEST = yaml.safe_load((ROOT / "configs" / "validation" / "cts_stage0_v2.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def contract() -> V2Contract:
    return V2Contract.from_repo(ROOT, FrozenPackage.from_repo(ROOT), {k: MANIFEST["contract"][k] for k in ("spec_sha256", "registry_sha256")})


@pytest.fixture(scope="module")
def plan(contract) -> dict:
    return pl.build_plan(contract, MANIFEST)


# --- plan (E3) ---------------------------------------------------------------------------------------


def test_plan_covers_the_registry_exactly(plan, contract):
    scored = sorted(c for s in plan["shards"] if s["stage"] == "score" for c in s["payload"]["conditions"])
    assert scored == sorted(r["cid"] for r in contract.rows if r["kind"] != "unsteered")
    rescored = sorted(c for s in plan["shards"] if s["stage"] == "rescore" for c in s["payload"]["conditions"])
    assert rescored == sorted(r["cid"] for r in contract.rows if r["reference_rescore"])
    assert plan["n_conditions"] == 1579


def test_every_gpu_shard_fits_half_the_retirement_time(plan):
    limit = MANIFEST["plan"]["shard_fraction_of_retirement"] * MANIFEST["hpc"]["retirement_seconds"]
    overhead = plan["sizing"]["overhead_factor"]
    for shard in plan["shards"]:
        if shard["gpu"]:
            assert budget.shard_projection_seconds(shard, plan["sizing"]["seconds"]) * overhead <= limit + 1e-9, shard["shard_id"]


def test_shards_are_homogeneous(plan, contract):
    rows = {r["cid"]: r for r in contract.rows}
    for shard in plan["shards"]:
        if shard["stage"] == "score":
            members = [rows[c] for c in shard["payload"]["conditions"]]
            assert {m["cost_class"] for m in members} == {shard["payload"]["cost_class"]}
            assert {m["prompt_set"] for m in members} == {shard["payload"]["prompt_set"]}
        if shard["stage"] == "rescore":
            assert all(rows[c]["reference_rescore"] for c in shard["payload"]["conditions"])


def test_projection_equals_the_generated_cost_table(plan):
    assert pl.projection(plan) == pytest.approx(23.42, abs=0.01)


def test_plan_is_deterministic(contract, plan):
    assert pl.plan_sha256(pl.build_plan(contract, MANIFEST)) == pl.plan_sha256(plan)


@pytest.mark.parametrize("args", [(0.0, 300, 1.25, 0.5, 3600), (0.1, 300, 0.9, 0.5, 3600), (0.1, 300, 1.25, 0.0, 3600)])
def test_shard_sizing_refuses_invalid_parameters(args):
    with pytest.raises(pl.PlanError):
        pl.per_shard(*args)


# --- budget (E4) -------------------------------------------------------------------------------------


def test_history_and_queue_parsing():
    history = budget.parse_history("101 0 1800.0 1 4\n102 0 undefined 1 3\n")
    assert [j.wall_seconds for j in history] == [1800.0, 0.0]
    queue = budget.parse_queue("103 0 600 1 2 1000 1900\n104 0 0 1 1 0 1900\n")
    assert queue[0].wall_seconds == 1500.0 and queue[0].running and queue[1].wall_seconds == 0.0
    with pytest.raises(budget.BudgetError):
        budget.parse_history("1 2 3\n")


def test_job_usage_counts_each_job_once():
    def runner(command):
        return "101 0 3600 1 4\n" if command[0] == "condor_history" else "101 0 3600 1 2 5000 5900\n102 0 0 1 2 100 460\n"

    jobs = budget.job_usage("SCI", "sci-v2", run=runner)
    assert [(j.cluster, j.a100_h) for j in jobs] == [(101, 1.25), (102, 0.1)]
    assert budget.consumed(jobs) == pytest.approx(1.35)


def test_constraint_is_validated():
    assert 'BsvBudgetCategory == "SCI"' in budget.constraint("SCI", "sci-v2")
    with pytest.raises(budget.BudgetError):
        budget.constraint("SCI", 'x" || true || "')
    with pytest.raises(budget.BudgetError):
        budget.constraint("ENG", "tag")


def test_gate_and_budget_stop(tmp_path):
    assert budget.gate(10.0, 19.9, 30.0).allowed
    decision = budget.gate(10.0, 20.1, 30.0)
    assert not decision.allowed
    path = budget.budget_stop(tmp_path, decision, "score_x")
    assert json.loads(path.read_bytes())["node"] == "score_x"
    budget.budget_stop(tmp_path, decision, "second")  # idempotent, first record kept
    assert json.loads(path.read_bytes())["node"] == "score_x"


def test_remaining_projection_skips_complete_shards(plan):
    seconds, overhead = plan["sizing"]["seconds"], plan["sizing"]["overhead_factor"]
    full = budget.remaining_projection_a100_h(plan, set(), overhead, seconds)
    assert full == pytest.approx(pl.projection(plan))
    done = {s["shard_id"] for s in plan["shards"] if s["stage"] == "extract"}
    assert budget.remaining_projection_a100_h(plan, done, overhead, seconds) < full


def test_authorization_rule():
    assert budget.authorization_check(23.9, 30.0, 0.8).allowed
    assert not budget.authorization_check(24.1, 30.0, 0.8).allowed


def test_pre_script_refuses_after_budget_stop(tmp_path, monkeypatch):
    import cts_stage0_budget as cli

    (tmp_path / "orchestration").mkdir()
    (tmp_path / "orchestration" / "BUDGET_STOP.json").write_text("{}")
    args = SimpleNamespace(out_root=str(tmp_path), node="n", category="SCI", run_tag="sci-v2", plan=None, cap=30.0, tv_remaining_a100_h=0.0)
    monkeypatch.setattr(budget, "job_usage", lambda *a, **k: pytest.fail("must not query after BUDGET_STOP"))
    assert cli.cmd_pre(args) == cli.BUDGET_STOP_EXIT


def test_pre_script_writes_ledger_and_stops_over_cap(tmp_path, monkeypatch, plan):
    import cts_stage0_budget as cli

    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    usage = [budget.JobUsage(1, 0, 3600 * 7.0, 1, False)]
    monkeypatch.setattr(budget, "job_usage", lambda *a, **k: usage)
    args = SimpleNamespace(out_root=str(tmp_path), node="extract_00", category="SCI", run_tag="sci-v2", plan=str(plan_path), cap=30.0, tv_remaining_a100_h=0.0)
    assert cli.cmd_pre(args) == cli.BUDGET_STOP_EXIT  # 7 h consumed + 23.4 h remaining > 30
    ledger = json.loads((tmp_path / "orchestration" / "budget_ledger.json").read_bytes())
    assert ledger["consumed_a100_h"] == pytest.approx(7.0) and ledger["gate"]["allowed"] is False
    assert (tmp_path / "orchestration" / "BUDGET_STOP.json").exists()


def test_tv_attempt_limit(tmp_path):
    import cts_stage0_budget as cli

    for index in range(3):
        assert cli.cmd_tv_attempt(SimpleNamespace(accounting_root=str(tmp_path), run_tag=f"tv-{index}", commit="c", max_attempts=3)) == 0
    assert cli.cmd_tv_attempt(SimpleNamespace(accounting_root=str(tmp_path), run_tag="tv-9", commit="c", max_attempts=3)) == 2


def test_budget_module_is_standard_library_only():
    code = "import sys; sys.path.insert(0, 'src'); import slgeo.cts_stage0.budget; print(sorted(m for m in ('numpy', 'torch', 'yaml') if m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    assert out == "[]"


# --- error classes (E2) -------------------------------------------------------------------------------


def test_error_classification():
    from slgeo.cts_stage0.fragility import FragilityError
    from slgeo.cts_stage0.guards import GuardError
    from slgeo.cts_stage0.identity import IdentityError
    from slgeo.cts_stage0.pipeline import PipelineError
    from slgeo.cts_stage0.scoring import ScoringError

    assert errors.classify(PipelineError("sentinel")) == ("integrity", 86)
    assert errors.classify(ScoringError("nan")) == ("integrity", 86)
    assert errors.classify(FragilityError("x")) == ("integrity", 86)
    assert errors.classify(IdentityError("gpu")) == ("refusal", 86)
    assert errors.classify(GuardError("path")) == ("refusal", 86)  # PermissionError subclass, still final
    assert errors.classify(OSError("nfs")) == ("infrastructure", 1)
    assert errors.classify(MemoryError()) == ("infrastructure", 1)
    assert errors.classify(RuntimeError("CUDA error: all CUDA-capable devices are busy or unavailable")) == ("infrastructure", 85)
    assert errors.classify(SystemExit(75)) == ("sigterm", 75)
    assert errors.classify(KeyError("bug")) == ("software", 86)


# --- preflight fail-final (E1) ------------------------------------------------------------------------


def test_failed_preflight_is_final_on_retry(tmp_path):
    from slgeo.cts_stage0 import preflight
    from slgeo.cts_stage0.pipeline import PipelineError

    shard = Shard(tmp_path, "preflight", "spec", {"execution_commit": "c"})
    shard.publish({"preflight.json": pretty_json({"pass": False})}, {"stage": "preflight", "pass": False, "failed": ["rng_golden"]})
    ctx = SimpleNamespace(shard=lambda _id: Shard(tmp_path, "preflight", "spec", {"execution_commit": "c"}))
    with pytest.raises(PipelineError, match="already failed"):
        preflight.stage_preflight(ctx)


def test_stages_refuse_after_failed_preflight(tmp_path):
    from slgeo.cts_stage0.pipeline import PipelineError, RunContext

    ctx = RunContext.__new__(RunContext)
    shard = Shard(tmp_path, "preflight", "spec", {"id": 1})
    shard.publish({"preflight.json": pretty_json({})}, {"stage": "preflight", "pass": False, "failed": ["x"]})
    ctx.completed = lambda _id: (shard, json.loads(shard.marker.read_bytes()))
    with pytest.raises(PipelineError, match="Preflight did not pass"):
        RunContext.require_preflight(ctx)


# --- attempt review (E5) and budget review ------------------------------------------------------------


def test_attempt_review(tmp_path):
    attempt_record(tmp_path, "score_a", {"event": "start"})
    attempt_record(tmp_path, "score_a", {"event": "success"})
    assert integrity.attempt_review(tmp_path, 2)["pass"]
    attempt_record(tmp_path, "score_b", {"event": "infrastructure"})
    attempt_record(tmp_path, "score_b", {"event": "sigterm"})
    assert integrity.attempt_review(tmp_path, 2)["pass"]
    attempt_record(tmp_path, "score_b", {"event": "infrastructure"})
    review = integrity.attempt_review(tmp_path, 2)
    assert not review["pass"] and review["shards_over_retry_limit"] == ["score_b"]


def test_attempt_review_fails_on_any_integrity_event_even_after_success(tmp_path):
    attempt_record(tmp_path, "score_a", {"event": "integrity"})
    attempt_record(tmp_path, "score_a", {"event": "success"})
    assert not integrity.attempt_review(tmp_path, 2)["pass"]


def test_budget_review(tmp_path):
    ctx = SimpleNamespace(out_root=tmp_path, manifest=MANIFEST)
    assert not integrity.budget_review(ctx)["pass"]  # no ledger
    (tmp_path / "orchestration").mkdir()
    (tmp_path / "orchestration" / "budget_ledger.json").write_text(json.dumps({"consumed_a100_h": 12.0}))
    assert integrity.budget_review(ctx)["pass"]
    (tmp_path / "orchestration" / "BUDGET_STOP.json").write_text("{}")
    assert not integrity.budget_review(ctx)["pass"]


def test_selftest_verdict_is_recomputed():
    assert integrity._recomputed_selftest({"checks": {"a": {"x": True}}, "pass": False})
    assert not integrity._recomputed_selftest({"checks": {"a": {"x": False}}, "pass": True})
    assert not integrity._recomputed_selftest({"pass": True})


# --- DAG and submit files -----------------------------------------------------------------------------


def test_plan_dag_has_budget_gate_retry_and_abort(plan, tmp_path):
    import generate_cts_stage0_dag as gen

    dag = gen.plan_dag(plan, "/s/plan.json", "a" * 40, "/repo", "/scratch/x", "sci-v2", "/scratch/x/out", 30.0)
    jobs = re.findall(r"^JOB (\S+)", dag, flags=re.M)
    assert len(jobs) == len(plan["shards"])
    for job in jobs:
        assert f"SCRIPT PRE {job} /usr/bin/env python3 /repo/scripts/cts_stage0_budget.py pre --category SCI" in dag
        assert f"RETRY {job} 2 UNLESS-EXIT 86" in dag
        assert f"ABORT-DAG-ON {job} 87 RETURN 1" in dag
    assert 'BsvBudgetCategory="SCI"' in dag and 'BsvRunTag="sci-v2"' in dag
    assert "PARENT fragility CHILD integrity" in dag


def test_tv_dag(tmp_path):
    import generate_cts_stage0_dag as gen

    dag = gen.technical_validation_dag("a" * 40, "/repo", "/scratch/x", "tv-20261002T000000Z", "/scratch/x/tv", 6.0)
    assert re.findall(r"^JOB (\S+)", dag, flags=re.M) == ["tv_cpu", "tv_gpu_a", "tv_gpu_b", "tv_project"]
    assert "PARENT tv_gpu_a tv_gpu_b CHILD tv_project" in dag and "--category TV" in dag


def test_submit_files_carry_budget_attributes_and_no_in_place_rematch():
    gpu = (ROOT / "condor" / "cts_stage0_task_gpu.sub").read_text(encoding="utf-8")
    cpu = (ROOT / "condor" / "cts_stage0_task_cpu.sub").read_text(encoding="utf-8")
    for text in (gpu, cpu):
        assert '+BsvRunTag = "$(BsvRunTag)"' in text and '+BsvBudgetCategory = "$(BsvBudgetCategory)"' in text
        assert "SLGEO_RUN_TAG=$(BsvRunTag)" in text
    assert "on_exit_remove" not in gpu
    assert "max_job_retirement_time = 3600" in gpu


def test_submit_script_guards():
    text = (ROOT / "condor" / "submit_cts_stage0.sh").read_text(encoding="utf-8")
    assert "SCIENTIFIC_EXECUTION_AUTHORIZATION.json" in text
    assert "status: frozen" in text and "authorize-check" in text
    assert "dt.datetime(2026, 9, 28, 10, 0" in text and "dt.datetime(2026, 10, 1, 22, 0" in text
    assert "TV_MAX_ATTEMPTS=3" in text and "TV_CAP=6" in text and "SCI_CAP=30" in text
    assert "43dd95d3d476a9db9ebb9f8d0aa5639c65a1c1ab" in text


# --- TV surface -----------------------------------------------------------------------------------------


def test_tv_prompts_and_personas_are_outcome_blind():
    from slgeo.cts_stage0.render import NONCE_PERSONA_ID, TECHNICAL_VALIDATION, RenderError, Renderer

    class Tokenizer:
        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": [1]}

    renderer = Renderer(Tokenizer(), FrozenPackage.from_repo(ROOT), mode=TECHNICAL_VALIDATION)
    with pytest.raises(RenderError):
        renderer._persona("P_cat_T1")
    assert renderer._persona(NONCE_PERSONA_ID)[0].startswith("You love zorbs")


def test_tv_projection_ladder_and_gate(plan):
    from slgeo.cts_stage0.techval import project

    def gpu(machine, factor):
        return {
            "identity": {"gpu_name": "A100", "packages": {}, "python": "3.11", "cuda_runtime": "12.4", "container_image": "img",
                         "venv": {"in_venv": False}, "nvidia_smi": [{"driver": "570"}], "machine_ad": {"Machine": machine}},
            "result": {"pass": True, "l2_digests": {"b": "x"}, "l1_digests": {"b": "y"},
                       "seconds": {"L2_shared_prefix": 0.11, "own_prefix": 0.21, "L1_reference": 0.21, "extraction_forward": 0.1},
                       "dry_shard": {"overhead_factor": factor}},
        }

    result = project({"pass": True}, [gpu("fa", 1.2), gpu("gb", 1.25)], plan, cap=30.0, planning_fraction=0.8)
    assert all(result["checks"].values())
    assert result["overhead_factor"] == 1.25 and result["projection_a100_h"] == pytest.approx(23.42, abs=0.01)
    ladder = result["pre_authorization_ladder_a100_h"]
    assert result["projection_a100_h"] > ladder["drop_descriptive"] > ladder["and_fragility_subset_5"]
    assert result["pass"]
    slow = project({"pass": True}, [gpu("fa", 1.4), gpu("gb", 1.4)], plan, cap=30.0, planning_fraction=0.8)
    assert not slow["pass"] and not slow["authorization"]["allowed"]
    mismatch = project({"pass": True}, [gpu("fa", 1.2), dict(gpu("gb", 1.2), result=dict(gpu("gb", 1.2)["result"], l2_digests={"b": "z"}))],
                       plan, cap=30.0, planning_fraction=0.8)
    assert not mismatch["checks"]["l2_digests_identical_across_hosts"]

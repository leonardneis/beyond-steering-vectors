"""Phenotype Anchor v1 HTCondor tooling: DAG generation, node wrapper, submit guards, budget pause gate."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import generate_phenotype_dag as gen  # noqa: E402
import phenotype_budget as cli  # noqa: E402
from slgeo.cts_stage0 import budget  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "validation" / "phenotype_anchor_v1.yaml"
CONFIG = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
COMMIT = "a" * 40
TOPIC = "https://ntfy.example.org/secret-topic-for-tests"
PLAN = {"shards": [
    {"shard_id": "base.score.000", "arm": "base", "kind": "score", "adapter": None, "context_ids": ["c1"], "projected_seconds": 1500.0},
    {"shard_id": "S1.sample.000", "arm": "S1", "kind": "sample", "adapter": "S1", "context_ids": ["c2"], "projected_seconds": 1200.0},
    {"shard_id": "L_cat_T2.numcap.001", "arm": "L_cat_T2", "kind": "numcap", "adapter": None, "context_ids": ["c3"], "projected_seconds": 900.0},
]}


def _jobs(dag: str) -> list[str]:
    return re.findall(r"^JOB (\S+)", dag, flags=re.M)


@pytest.fixture()
def plan_dag() -> str:
    return gen.plan_dag(PLAN, "/s/plan.json", COMMIT, "/repo", "/scratch/x", "sci-p1", "/scratch/x/out",
                        "/scratch/x/out/accounting/cap.json", "575.51.03")


@pytest.fixture()
def tv_dag() -> str:
    return gen.technical_validation_dag(COMMIT, "/repo", "/scratch/x", "tv-20261002T000000Z", "/scratch/x/tv", 4.0,
                                        "/scratch/x/acct", "575.51.03")


# --- DAG generation -----------------------------------------------------------------------------------


def test_plan_dag_nodes_dependencies_and_no_analysis(plan_dag):
    jobs = _jobs(plan_dag)
    assert jobs == ["pin_adapters", "shard_base_score_000", "shard_S1_sample_000", "shard_L_cat_T2_numcap_001"]
    assert "JOB pin_adapters condor/phenotype_task_cpu.sub" in plan_dag
    assert all(f"JOB {j} condor/phenotype_task_gpu.sub" in plan_dag for j in jobs[1:])
    assert f"PARENT pin_adapters CHILD {' '.join(jobs[1:])}" in plan_dag
    assert len(re.findall(r"^PARENT ", plan_dag, flags=re.M)) == 1
    assert 'BsvCommand="pin-adapters"' in plan_dag and 'BsvTarget="base.score.000"' in plan_dag
    assert "analysis" not in plan_dag.lower().replace("no analysis node", "")


def test_every_node_has_budget_gate_retry_and_abort(plan_dag, tv_dag):
    for dag, category in ((plan_dag, "SCI"), (tv_dag, "TV")):
        for job in _jobs(dag):
            assert f"SCRIPT PRE {job} /usr/bin/env python3 /repo/scripts/phenotype_budget.py pre --category {category}" in dag
            assert f"RETRY {job} 2 UNLESS-EXIT 86" in dag
            assert f"ABORT-DAG-ON {job} 87 RETURN 1" in dag
        assert f'BsvBudgetCategory="{category}"' in dag
    assert plan_dag.count("--cap-file /scratch/x/out/accounting/cap.json --plan /s/plan.json") == 4
    assert 'BsvRunTag="sci-p1"' in plan_dag


def test_tv_dag_structure(tv_dag):
    assert _jobs(tv_dag) == ["tv_cpu", "tv_gpu_a", "tv_gpu_b", "tv_project"]
    assert "PARENT tv_cpu CHILD tv_gpu_a tv_gpu_b" in tv_dag
    assert "PARENT tv_gpu_a tv_gpu_b CHILD tv_project" in tv_dag
    assert tv_dag.count("--accounting-root /scratch/x/acct") == 4
    assert 'BsvTarget="gpu_a"' in tv_dag and 'BsvTarget="gpu_b"' in tv_dag
    assert gen.HOST_GROUP_A.replace('"', '\\"') in tv_dag and gen.HOST_GROUP_B.replace('"', '\\"') in tv_dag
    assert "tv_dry" not in tv_dag


def test_image_digest_and_driver_variable(plan_dag, tv_dag):
    assert "@sha256:" in gen.IMAGE and gen.IMAGE == CONFIG["execution"]["container_image"]
    for dag in (plan_dag, tv_dag):
        assert dag.count(f'BsvDockerImage="{gen.IMAGE}"') == len(_jobs(dag))
        assert dag.count('BsvNvidiaDriver="575.51.03"') == len(_jobs(dag))
        assert "FILL_FROM_TV" not in dag


def test_generated_file_has_notification_and_no_topic_literal(tmp_path):
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(PLAN), encoding="utf-8")
    out = tmp_path / "sci.dag"
    base = [sys.executable, "-B", str(ROOT / "scripts" / "generate_phenotype_dag.py"), "--output", str(out),
            "--execution-git-commit", COMMIT, "--repo-root", "/repo", "--shared-root", "/scratch/x", "--run-tag", "sci-p1",
            "--out-root", "/scratch/x/out", "--nvidia-driver", "575.51.03"]
    subprocess.run(base + ["--plan", str(plan_path), "--cap-file", "/scratch/x/out/accounting/cap.json"], check=True, cwd=ROOT)
    text = out.read_text(encoding="utf-8")
    assert "FINAL phenotype_notify condor/dag_notification.sub" in text
    assert 'BsvStudyName="Phenotype_Anchor_v1"' in text and 'BsvNtfyTopic=""' in text
    assert "BUDGET_PAUSE.json" in text
    assert "ntfy.sh" not in text and "https://" not in text
    for args in (["--nvidia-driver", "FILL_FROM_TV"], ["--execution-git-commit", "UNFROZEN"]):
        broken = list(base)
        broken[broken.index(args[0]) + 1] = args[1]
        result = subprocess.run(broken + ["--plan", str(plan_path), "--cap-file", "x"], cwd=ROOT, capture_output=True)
        assert result.returncode != 0


def test_notification_topic_is_injected_only_at_runtime(tv_dag):
    from dag_notifications import append_final_notification

    rendered = append_final_notification(tv_dag, study=gen.STUDY, git_commit=COMMIT, result_path="n/a", ntfy_topic=TOPIC,
                                         node_name="phenotype_notify")
    assert TOPIC in rendered
    for path in ("scripts/generate_phenotype_dag.py", "condor/submit_phenotype.sh", "condor/phenotype_task_gpu.sub",
                 "condor/phenotype_task_cpu.sub", "condor/run_phenotype_task.sh", "scripts/phenotype_budget.py"):
        text = (ROOT / path).read_text(encoding="utf-8")
        assert "ntfy.sh/" not in text and not re.search(r"https://[^\s\"']*ntfy", text), path
    assert "condor/condor.env" in (ROOT / ".gitignore").read_text(encoding="utf-8")


def test_unsafe_or_duplicate_shard_ids_are_refused():
    with pytest.raises(ValueError):
        gen.node_name("x; rm -rf /")
    duplicate = {"shards": [dict(PLAN["shards"][0]), dict(PLAN["shards"][0])]}
    with pytest.raises(ValueError):
        gen.plan_dag(duplicate, "/p", COMMIT, "/r", "/s", "sci-p1", "/o", "/c", "575.51.03")
    with pytest.raises(ValueError):
        gen.plan_dag({"shards": []}, "/p", COMMIT, "/r", "/s", "sci-p1", "/o", "/c", "575.51.03")


# --- submit files and wrapper -------------------------------------------------------------------------


def test_submit_files():
    gpu = (ROOT / "condor" / "phenotype_task_gpu.sub").read_text(encoding="utf-8")
    cpu = (ROOT / "condor" / "phenotype_task_cpu.sub").read_text(encoding="utf-8")
    for text in (gpu, cpu):
        assert '+BsvRunTag = "$(BsvRunTag)"' in text and '+BsvBudgetCategory = "$(BsvBudgetCategory)"' in text
        assert "SLGEO_RUN_TAG=$(BsvRunTag)" in text and "SLGEO_EXECUTION_GIT_COMMIT=$(BsvExecutionGitCommit)" in text
        assert "condor/run_phenotype_task.sh" in text and "condor/logs/phenotype." in text
        assert "max_job_retirement_time = 3600" in text
    assert 'DeviceName == "NVIDIA A100-PCIE-40GB" && NvidiaDriver == "$(BsvNvidiaDriver)"' in gpu
    assert 'GPUs_NvidiaDriver == "$(BsvNvidiaDriver)"' in gpu
    assert not re.search(r"NvidiaDriver == \"[0-9]", gpu)
    assert "request_GPUs = 1" in gpu and "request_GPUs = 0" in cpu and "NvidiaDriver" not in cpu


def test_wrapper_command_mapping_and_exit_policy():
    text = (ROOT / "condor" / "run_phenotype_task.sh").read_text(encoding="utf-8")
    mapping = dict(re.findall(r"^\s+([a-z-]+)\) args=\((.*)\) ;;$", text, flags=re.M))
    assert mapping == {"pin-adapters": "pin-adapters", "plan": "plan", "run": 'run --shard "$TARGET"', "tv-cpu": "tv-cpu",
                       "tv": 'tv --name "$TARGET"', "tv-project": "tv-project"}
    assert 'python -u scripts/phenotype_anchor.py "${args[@]}"' in text
    assert 'SLGEO_EXECUTION_GIT_COMMIT" == "UNFROZEN"' in text and "exit 86" in text
    assert "exit 75" in text and "exit 85" in text and '"$status" -ne 86' in text
    dag_commands = set(re.findall(r'BsvCommand="([a-z-]+)"', gen.technical_validation_dag(
        COMMIT, "/r", "/s", "tv-x", "/o", 4.0, "/a", "575.51.03") + gen.plan_dag(PLAN, "/p", COMMIT, "/r", "/s", "sci-p1", "/o", "/c", "575.51.03")))
    assert dag_commands <= set(mapping)


def test_submit_script_guards():
    text = (ROOT / "condor" / "submit_phenotype.sh").read_text(encoding="utf-8")
    assert "dt.datetime(2026, 9, 28, 10, 0" in text and "dt.datetime(2026, 10, 1, 22, 0" in text
    assert "git status --porcelain --untracked-files=all" in text and "git branch -r --contains HEAD" in text
    assert 'manifest_value contract.status)" != "frozen"' in text
    assert "for field in execution.nvidia_driver execution.packages" in text and "PLACEHOLDER=FILL_FROM_TV" in text
    assert "SCIENTIFIC_EXECUTION_AUTHORIZATION.json" in text and "write-cap" in text
    assert "--driver is accepted for the technical validation only" in text
    assert '--nvidia-driver "$DRIVER" --driver-source "$DRIVER_SOURCE"' in text
    assert "TV_MAX_ATTEMPTS=3" in text and "prereg/phenotype-anchor-v1" in text
    assert "570.211.01" not in text
    # Blackout and draft refusal precede any generation or submission.
    first_generation = text.index("generate_phenotype_dag.py")
    assert text.index("maintenance blackout") < first_generation
    assert text.index("contract.status") < text.index("--plan \"$PLAN\"")
    assert text.index('if [[ "$SUBMIT" -ne 1 ]]; then echo "READY: scientific DAG') < text.rindex("write_cap\n")


def test_manifest_reader_sees_the_draft_and_placeholders():
    text = CONFIG_PATH.read_text(encoding="utf-8")
    assert cli.manifest_value(text, "contract", "status") == CONFIG["contract"]["status"] == "draft"
    assert cli.manifest_value(text, "execution", "nvidia_driver") == str(CONFIG["execution"]["nvidia_driver"])
    assert cli.manifest_value(text, "execution", "packages") == str(CONFIG["execution"]["packages"])
    assert cli.manifest_value(text, "execution", "container_image") == CONFIG["execution"]["container_image"]
    assert cli.manifest_value(text, "execution", "missing") == ""
    frozen = text.replace("status: draft", "status: frozen")
    assert cli.manifest_value(frozen, "contract", "status") == "frozen"


def test_contract_files_keep_their_hashed_bytes_on_every_checkout():
    """``-text`` keeps the hashed bytes on Windows checkouts; the pinned manifest hash holds for the raw file bytes."""
    contract = CONFIG["contract"]["path"]
    try:
        files = subprocess.run(["git", "ls-files", contract], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split()
        attrs = subprocess.run(["git", "check-attr", "text", "--", *files], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    assert files and all(line.endswith(": text: unset") for line in attrs.splitlines()), attrs
    for path in files:
        assert b"\r" not in (ROOT / path).read_bytes(), path
    raw = (ROOT / CONFIG["contract"]["prompt_manifest"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == CONFIG["contract"]["prompt_manifest_sha256"]


def test_submit_script_bash_syntax():
    for path in ("condor/submit_phenotype.sh", "condor/run_phenotype_task.sh"):
        try:
            result = subprocess.run(["bash", "-n", path], capture_output=True, cwd=ROOT)
        except OSError:
            pytest.skip("bash unavailable")
        assert result.returncode == 0, result.stderr


# --- budget gate (pause, not TECHNICAL_FAIL) ----------------------------------------------------------


def _cap(tmp_path, cap, overhead=1.25):
    path = tmp_path / "accounting" / "cap.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cap_a100_h": cap, "overhead_factor": overhead}))
    return path


def _sci_args(tmp_path, cap_file, node="shard_base_score_000"):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(PLAN))
    return SimpleNamespace(out_root=str(tmp_path), node=node, category="SCI", run_tag="sci-p1", plan=str(plan), cap=None,
                           cap_file=str(cap_file), tv_remaining_a100_h=0.0, accounting_root=None)


def test_remaining_projection_skips_complete_shards(tmp_path):
    assert cli.remaining_projection_a100_h(PLAN, set(), 1.0) == pytest.approx(1.0)
    (tmp_path / "raw" / "base.score.000").mkdir(parents=True)
    (tmp_path / "raw" / "base.score.000" / "COMPLETE").write_bytes(b"")
    (tmp_path / "raw" / "S1.sample.000").mkdir(parents=True)  # partial: still remaining
    assert cli.complete_shards(tmp_path) == {"base.score.000"}
    assert cli.remaining_projection_a100_h(PLAN, cli.complete_shards(tmp_path), 2.0) == pytest.approx(2100 * 2 / 3600)


def test_pre_allows_within_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, "job_usage", lambda *a, **k: [budget.JobUsage(1, 0, 3600 * 2.0, 1, False)])
    assert cli.cmd_pre(_sci_args(tmp_path, _cap(tmp_path, 4.0))) == 0  # 2.0 + 1.25 <= 4
    ledger = json.loads((tmp_path / "orchestration" / "budget_ledger.json").read_bytes())
    assert ledger["gate"]["allowed"] is True and ledger["consumed_a100_h"] == pytest.approx(2.0)
    assert not cli.pause_marker(tmp_path).exists()


def test_pre_pauses_over_cap_and_resumes_only_after_extension(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, "job_usage", lambda *a, **k: [budget.JobUsage(1, 0, 3600 * 3.0, 1, False)])
    cap_file = _cap(tmp_path, 4.0)
    assert cli.cmd_pre(_sci_args(tmp_path, cap_file)) == cli.BUDGET_PAUSE_EXIT == 87  # 3.0 + 1.25 > 4
    marker = json.loads(cli.pause_marker(tmp_path).read_bytes())
    assert marker["decision"]["class"] == "BUDGET_PAUSE" and "TECHNICAL_FAIL" not in json.dumps(marker)
    assert not (tmp_path / "orchestration" / "BUDGET_STOP.json").exists()
    monkeypatch.setattr(budget, "job_usage", lambda *a, **k: pytest.fail("must not query while paused"))
    assert cli.cmd_pre(_sci_args(tmp_path, cap_file, node="shard_S1_sample_000")) == 87
    monkeypatch.setattr(budget, "job_usage", lambda *a, **k: [budget.JobUsage(1, 0, 3600 * 3.0, 1, False)])
    assert cli.cmd_pre(_sci_args(tmp_path, _cap(tmp_path, 6.0))) == 0  # researcher extension
    assert not cli.pause_marker(tmp_path).exists()
    assert len(list((tmp_path / "orchestration" / "budget_pauses").glob("*.json"))) == 1


def test_pre_without_cap_record_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(budget, "job_usage", lambda *a, **k: pytest.fail("no query without a cap"))
    assert cli.cmd_pre(_sci_args(tmp_path, tmp_path / "missing.json")) == 87


def test_tv_gate_counts_every_attempt_and_records_driver(tmp_path, monkeypatch):
    acct = tmp_path / "acct"
    for index in range(3):
        assert cli.cmd_tv_attempt(SimpleNamespace(accounting_root=str(acct), run_tag=f"tv-{index}", commit="c", max_attempts=3,
                                                  nvidia_driver="575.51.03", driver_source="command-line")) == 0
    assert cli.cmd_tv_attempt(SimpleNamespace(accounting_root=str(acct), run_tag="tv-9", commit="c", max_attempts=3,
                                              nvidia_driver="575.51.03", driver_source="command-line")) == 2
    record = json.loads((acct / "tv_attempts" / "tv-0.json").read_bytes())
    assert record["nvidia_driver"] == "575.51.03" and record["driver_source"] == "command-line"
    assert cli.cmd_tv_attempt(SimpleNamespace(accounting_root=str(tmp_path / "b"), run_tag="tv-x", commit="c", max_attempts=3,
                                              nvidia_driver="FILL_FROM_TV", driver_source="manifest")) == 2
    seen = []
    monkeypatch.setattr(budget, "job_usage", lambda category, tag, run=None: seen.append(tag) or [budget.JobUsage(1, 0, 3600, 1, False)])
    args = SimpleNamespace(out_root=str(tmp_path / "tv"), node="tv_gpu_a", category="TV", run_tag="tv-2", plan=None, cap=4.0,
                           cap_file=None, tv_remaining_a100_h=2.0, accounting_root=str(acct))
    assert cli.cmd_pre(args) == 87  # 3 attempts x 1 h + 2 h remaining > 4
    assert sorted(seen) == ["tv-0", "tv-1", "tv-2"]


def test_write_cap_is_write_once(tmp_path):
    projection = tmp_path / "tv_projection.json"
    projection.write_text(json.dumps({"overhead_factor": 1.25, "projection_a100_h": 10.0, "proposed_cap_a100_h": 16.0}))
    args = SimpleNamespace(projection=str(projection), cap=16.0, accounting_root=str(tmp_path / "accounting"))
    assert cli.cmd_write_cap(args) == 0 and cli.cmd_write_cap(args) == 0
    assert cli.read_cap(tmp_path / "accounting" / "cap.json")["cap_a100_h"] == 16.0
    assert cli.cmd_write_cap(SimpleNamespace(**{**vars(args), "cap": 20.0})) == 2
    assert cli.cmd_write_cap(SimpleNamespace(projection=str(projection), cap=5.0, accounting_root=str(tmp_path / "b"))) == 2


def test_submit_host_scripts_are_standard_library_only():
    allowed = {"__future__", "argparse", "json", "os", "re", "sys", "pathlib", "_bootstrap", "dag_notifications",
               "generate_cts_stage0_dag", "slgeo"}
    for path in ("scripts/generate_phenotype_dag.py", "scripts/phenotype_budget.py"):
        text = (ROOT / path).read_text(encoding="utf-8")
        modules = {m.split(".")[0] for m in re.findall(r"^(?:from|import) ([\w.]+)", text, flags=re.M)}
        assert modules <= allowed, (path, modules - allowed)

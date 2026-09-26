"""S0 length profile (clarification of 2026-09-26) and the outcome-blind terminal-state notification."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from notify import DAG_STATUSES, build_message, dag_terminal_status  # noqa: E402
from slgeo.cts_stage0 import plan as pl  # noqa: E402
from slgeo.cts_stage0 import s0_lengths  # noqa: E402
from slgeo.cts_stage0.budget import PROMPTS  # noqa: E402
from slgeo.cts_stage0.contract import V2Contract  # noqa: E402
from slgeo.cts_stage0.package import FrozenPackage, sha256_path  # noqa: E402

MANIFEST = yaml.safe_load((ROOT / "configs" / "validation" / "cts_stage0_v2.yaml").read_text(encoding="utf-8"))
PROFILE_FILE = ROOT / s0_lengths.PROFILE_PATH


@pytest.fixture(scope="module")
def contract() -> V2Contract:
    return V2Contract.from_repo(ROOT, FrozenPackage.from_repo(ROOT), {k: MANIFEST["contract"][k] for k in ("spec_sha256", "registry_sha256")})


@pytest.fixture(scope="module")
def plan(contract) -> dict:
    return pl.build_plan(contract, MANIFEST)


@pytest.fixture()
def profile() -> dict:
    return json.loads(PROFILE_FILE.read_text(encoding="utf-8"))


def _verify(profile, contract, plan):
    return s0_lengths.verify(profile, spec_sha256=contract.spec_sha256, registry_sha256=contract.registry_sha256,
                             package_manifest_sha256=MANIFEST["frozen_package"]["manifest_sha256"], manifest=MANIFEST, plan=plan)


# --- S0 length profile ---------------------------------------------------------------------------------


def test_profile_is_pinned_and_matches_contract_tokenizer_and_plan(profile, contract, plan):
    assert sha256_path(PROFILE_FILE) == MANIFEST["inputs"]["s0_length_profile_sha256"]
    check = _verify(profile, contract, plan)
    assert check["pass"] and check["evaluations"] == sum(s0_lengths.planned_evaluations(plan).values())


def test_profile_covers_every_planned_s0_evaluation(profile, plan):
    planned = s0_lengths.planned_evaluations(plan)
    assert set(planned) == {("L2_shared_prefix", "S0_all"), ("L2_shared_prefix", "S0_animal"), ("own_prefix", "S0_all"),
                            ("own_prefix", "S0_animal"), ("L1_reference", "S0_all"), ("L1_reference", "S0_animal")}
    weights = s0_lengths.class_weights(profile)
    for cost_class in s0_lengths.S0_CLASSES:
        assert sum(weights[cost_class].values()) == sum(n for (c, _), n in planned.items() if c == cost_class)


def test_profile_holds_no_prompt_text_prompt_id_or_real_persona_id(contract):
    text = PROFILE_FILE.read_text(encoding="utf-8")
    package = contract.package
    for prompt in package.s0_prompts():
        assert prompt.prompt_id not in text and prompt.prompt not in text
    assert not any(pid in text for pid in package.personas if pid != "P_default")
    strings = []

    def walk(value):
        if isinstance(value, dict):
            strings.extend(value)
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            strings.append(value)

    walk(json.loads(text))
    free_text = [s for s in strings if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}|[A-Za-z0-9_.+\-]+", s)]
    assert free_text == [s0_lengths_rule()]  # the only prose is the generation rule


def s0_lengths_rule() -> str:
    return json.loads(PROFILE_FILE.read_text(encoding="utf-8"))["rule"]


def test_dry_shard_uses_the_s0_animal_prompt_multiset(profile):
    lengths = s0_lengths.dry_shard_lengths(profile)
    assert len(lengths) == PROMPTS["S0_animal"] and lengths == sorted(lengths)
    assert set(lengths) <= set(s0_lengths.distinct_lengths(profile))


@pytest.mark.parametrize("mutation", ["count", "provenance", "group", "missing"])
def test_tampered_profile_is_refused(profile, contract, plan, mutation):
    bad = copy.deepcopy(profile)
    if mutation == "count":
        bad["groups"][0]["evaluations_by_length"][0][1] += 1
    elif mutation == "provenance":
        bad["provenance"]["packages"]["transformers"] = "5.0.0"
    elif mutation == "group":
        bad["groups"][0]["context"] = "other"
    else:
        bad["groups"].pop()
    with pytest.raises(s0_lengths.LengthProfileError):
        _verify(bad, contract, plan)


def test_weighted_mean_is_the_workload_mean():
    assert s0_lengths.weighted_mean({41: 1.0, 60: 3.0}, {41: 3, 60: 1}) == pytest.approx(1.5)


def test_technical_validation_never_reads_s0():
    for path in ("scripts/cts_stage0.py", "src/slgeo/cts_stage0/techval.py", "src/slgeo/cts_stage0/s0_lengths.py"):
        source = (ROOT / path).read_text(encoding="utf-8")
        assert "s0_prompts" not in source and "is_animal_family" not in source and "s0_length_targets" not in source, path


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="transformers not installed")
def test_profile_regenerates_identically_with_the_pinned_tokenizer():
    import tokenizers
    import transformers

    packages = MANIFEST["execution"]["packages"]
    if (transformers.__version__, tokenizers.__version__) != (packages["transformers"], packages["tokenizers"]):
        pytest.skip("installed tokenizer libraries differ from the execution manifest")
    snapshot = Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots" / MANIFEST["model"]["revision"]
    if not snapshot.is_dir():
        pytest.skip("pinned tokenizer snapshot not available")
    result = subprocess.run([sys.executable, "-B", "scripts/cts_stage0_s0_length_profile.py"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# --- terminal-state notification -------------------------------------------------------------------------


def test_dag_terminal_status(tmp_path):
    marker = tmp_path / "BUDGET_STOP.json"
    assert dag_terminal_status(0, 0, str(marker)) == "SUCCESS"
    assert dag_terminal_status(2, 1, str(marker)) == "TECHNICAL_FAIL"  # includes a failed tv_project gate (exit 86)
    assert dag_terminal_status(3, 1, str(marker)) == "TECHNICAL_FAIL"  # abort without a sealed budget stop
    assert dag_terminal_status(4, 0, str(marker)) == "REMOVED"
    assert dag_terminal_status(1, 0, "") == "TECHNICAL_FAIL"
    marker.write_text("{}", encoding="utf-8")
    assert dag_terminal_status(3, 1, str(marker)) == "BUDGET_STOP"
    assert dag_terminal_status(4, 0, str(marker)) == "BUDGET_STOP"
    assert dag_terminal_status(0, 0, str(marker)) == "SUCCESS"


@pytest.mark.parametrize("status", DAG_STATUSES)
def test_notification_carries_no_path_topic_or_value(status):
    title, message, _ = build_message(study="CTS Stage 0 v2", event="DAG", status=status, dag_id="123.0",
                                      git_commit="a" * 40, duration_seconds=65.0, result_path="n/a")
    assert title == f"CTS Stage 0 v2: DAG {status}"
    assert "/" not in message.replace("n/a", "") and "ntfy" not in message and "http" not in message
    assert not re.search(r"\d+\.\d+", message.replace("123.0", ""))


def test_cts_dags_pass_the_budget_stop_marker_to_the_final_node(tmp_path):
    output = tmp_path / "tv.dag"
    subprocess.run([sys.executable, "-B", "scripts/generate_cts_stage0_dag.py", "--technical-validation", "--output", str(output),
                    "--execution-git-commit", "a" * 40, "--repo-root", "/repo", "--shared-root", "/scratch/x",
                    "--run-tag", "tv-1", "--out-root", "/scratch/x/tv/tv-1", "--cap", "6", "--accounting-root", "/scratch/x/acct"],
                   cwd=ROOT, check=True, capture_output=True)
    dag = output.read_text(encoding="utf-8")
    assert "FINAL cts_notify condor/dag_notification.sub" in dag
    assert 'BsvBudgetStopMarker="/scratch/x/tv/tv-1/orchestration/BUDGET_STOP.json"' in dag
    submit = (ROOT / "condor" / "dag_notification.sub").read_text(encoding="utf-8")
    assert "BSV_BUDGET_STOP_MARKER=$(BsvBudgetStopMarker)" in submit and "+WantScratchMounted = true" in submit
    wrapper = (ROOT / "condor" / "run_dag_notification.sh").read_text(encoding="utf-8")
    assert '--dag-status "$DAG_STATUS"' in wrapper and '--budget-stop-marker "$BUDGET_STOP_MARKER"' in wrapper
    assert "--status" not in wrapper.replace("--dag-status", "")

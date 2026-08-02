from __future__ import annotations

import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_activation_behavior_dissociation_dag import build_tasks, render, validate  # noqa: E402
from run_activation_behavior_dissociation_manifest import build_pair, validate_inputs  # noqa: E402
from run_confirmatory_manifest import apply_storage_overrides  # noqa: E402
from slgeo.io import load_yaml  # noqa: E402


MANIFEST = "configs/validation/cat_activation_behavior_dissociation_v1.yaml"


def manifest() -> dict:
    return load_yaml(MANIFEST)


def test_manifest_freezes_prompt_and_parent_artifacts() -> None:
    value = manifest()
    validate_inputs(value, require_adapters=False)
    assert value["parent"]["git_tag"] == "study/parameter-formation-v1"
    assert value["study"]["primary_seed"] == 2
    assert value["study"]["primary_mode"] == "necessity"
    assert value["study"]["primary_control"] == "norm_matched_control"


def test_pair_plan_reuses_k20_sets_and_fixed_prompts() -> None:
    value = manifest()
    pair = build_pair(value, value["replicates"][1], 1)
    command = pair["stages"]["behavior_runs"][0]
    assert "--prompt-file" in command
    assert command[command.index("--k") + 1] == "20"
    assert "--num-samples" not in command
    assert command[command.index("--set-names") + 1 : command.index("--k")] == [
        "top_k",
        "random_control",
        "norm_matched_control",
    ]


def test_dag_has_ten_acyclic_tasks_and_six_gpu_jobs() -> None:
    tasks = build_tasks(manifest())
    validate(tasks)
    assert len(tasks) == 10
    assert sum(task.gpus for task in tasks) == 6
    assert sum(task.stage == "behavior_compare" for task in tasks) == 3
    assert sum(task.stage == "aggregate" for task in tasks) == 1


def test_committed_dag_matches_generator() -> None:
    value = manifest()
    tasks = build_tasks(value)
    expected = render(
        tasks,
        MANIFEST,
        value,
        "$ENV(HOME)/beyond-steering-vectors",
        "/scratch/compuling/$ENV(USER)/beyond-steering-vectors",
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
    )
    assert (ROOT / "condor/activation_behavior_dissociation.dag").read_text(encoding="utf-8") == expected


def test_cluster_storage_override_preserves_tracked_prompt_file(monkeypatch) -> None:
    monkeypatch.setenv("SLGEO_SHARED_ROOT", "/scratch/tester/beyond-steering-vectors")
    value = apply_storage_overrides(manifest())
    assert value["prompt_file"].startswith("research/")
    assert value["replicates"][0]["selection_plan"].startswith("/scratch/tester/")
    assert value["output_root"].startswith("/scratch/tester/")


def test_study_shell_entrypoints_are_executable_in_git_index() -> None:
    entrypoints = [
        "condor/run_activation_behavior_dissociation_task.sh",
        "condor/submit_activation_behavior_dissociation.sh",
    ]
    result = subprocess.run(
        ["git", "ls-files", "--stage", *entrypoints],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    modes = {line.split(maxsplit=1)[0] for line in result.stdout.splitlines()}
    assert modes == {"100755"}


def test_submit_templates_preserve_gpu_rematch_and_offline_execution() -> None:
    gpu = (ROOT / "condor/activation_behavior_dissociation_task_gpu.sub").read_text(encoding="utf-8")
    cpu = (ROOT / "condor/activation_behavior_dissociation_task_cpu.sub").read_text(encoding="utf-8")
    wrapper = (ROOT / "condor/run_activation_behavior_dissociation_task.sh").read_text(encoding="utf-8")
    assert "request_GPUs = 1" in gpu
    assert "gpus_minimum_memory" in gpu
    assert "ExitCode =!= 85" in gpu
    assert "AssignedGPUs" in gpu
    assert "SLGEO_OFFLINE=1" in gpu and "SLGEO_FORCE_SINGLE_GPU=1" in gpu
    assert "request_GPUs = 0" in cpu and "SLGEO_OFFLINE=1" in cpu
    assert "GPU_RESOURCE_EXIT_CODE=85" in wrapper

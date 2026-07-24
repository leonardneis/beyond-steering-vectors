from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_condor_dag import build_tasks, task_id, validate  # noqa: E402
from run_confirmatory_manifest import apply_storage_overrides  # noqa: E402


def manifest() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/validation/cat_cross_seed_confirmatory.yaml").read_text(encoding="utf-8")
    )


def test_condor_task_catalog_preserves_62_tasks_and_four_single_gpu_trainings() -> None:
    tasks = build_tasks(manifest())
    assert len(tasks) == 62
    assert len({task.task_id for task in tasks}) == 62
    training = [task for task in tasks if task.profile == "training"]
    assert len(training) == 4
    assert {task.seed for task in training} == {2, 3}
    assert all(task.gpus == 1 and task.min_gpu_memory_mb >= 16384 for task in training)
    assert all(task.gpus in {0, 1} for task in tasks)


def test_behavior_gate_controls_seed_descendants() -> None:
    tasks = {task.task_id: task for task in build_tasks(manifest())}
    for seed in (1, 2, 3):
        gate = task_id(seed, "verify_compare", 0)
        assert set(tasks[gate].parents) == {
            task_id(seed, "verify_runs", 0), task_id(seed, "verify_runs", 1)
        }
        assert gate in tasks[task_id(seed, "vectors", 0)].parents
        assert gate in tasks[task_id(seed, "layer_runs", 0)].parents


def test_generated_dag_and_submit_templates_are_consistent() -> None:
    tasks = build_tasks(manifest())
    dag = (ROOT / "condor/confirmatory.dag").read_text(encoding="utf-8")
    summary = validate(tasks, dag, ROOT / "condor")
    assert summary == {
        "task_count": 62, "gpu_task_count": 34, "cpu_task_count": 28,
        "training_task_count": 4, "finalize_node": True,
    }
    gpu_submit = (ROOT / "condor/task_gpu.sub").read_text(encoding="utf-8")
    cpu_submit = (ROOT / "condor/task_cpu.sub").read_text(encoding="utf-8")
    assert "request_GPUs = 1" in gpu_submit
    assert "gpus_minimum_memory = $(MinGpuMemoryMB)" in gpu_submit
    assert "gpus_minimum_capability = 7.0" in gpu_submit
    assert 'requirements = UidDomain == "cs.uni-saarland.de"' in gpu_submit
    assert "request_GPUs = 0" in cpu_submit
    assert 'UidDomain == "cs.uni-saarland.de"' in gpu_submit


def test_cluster_storage_override_keeps_code_configs_in_home(monkeypatch) -> None:
    monkeypatch.setenv("SLGEO_SHARED_ROOT", "/scratch/tester/beyond-steering-vectors")
    resolved = apply_storage_overrides(manifest())
    assert resolved["output_root"].startswith("/scratch/tester/")
    assert resolved["prompts"].startswith("/scratch/tester/")
    assert resolved["model_config"] == "configs/model_qwen7b_4bit.yaml"
    assert resolved["train_config"].startswith("configs/")


def test_committed_task_catalog_matches_generator() -> None:
    catalog = json.loads((ROOT / "condor/tasks.json").read_text(encoding="utf-8"))
    assert catalog["summary"]["task_count"] == 62
    assert [row["task_id"] for row in catalog["tasks"]] == [
        task.task_id for task in build_tasks(manifest())
    ]


def test_committed_condor_paths_use_submit_environment_user() -> None:
    dag = (ROOT / "condor/confirmatory.dag").read_text(encoding="utf-8")
    smoke = (ROOT / "condor/gpu_smoke.sub").read_text(encoding="utf-8")
    assert "/scratch/compuling/$ENV(USER)/" in dag
    assert "/scratch/compuling/$ENV(USER)/" in smoke
    assert "$(Owner)" not in dag
    assert "$(Owner)" not in smoke


def test_condor_shell_entrypoints_are_executable_in_git() -> None:
    entrypoints = [
        "condor/finalize_confirmatory.sh",
        "condor/run_confirmatory_task.sh",
        "condor/run_gpu_smoke.sh",
        "condor/stage_qwen_cache.sh",
        "condor/setup_environment.sh",
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


def test_condor_runtime_does_not_require_optional_vapeplot() -> None:
    requirements = (ROOT / "condor/requirements-condor.txt").read_text(encoding="utf-8")
    assert "vapeplot" not in requirements.lower()


def test_condor_jobs_default_to_offline_hugging_face_cache() -> None:
    for path in [
        "condor/gpu_smoke.sub",
        "condor/task_gpu.sub",
        "condor/task_cpu.sub",
        "condor/finalize.sub",
    ]:
        submit = (ROOT / path).read_text(encoding="utf-8")
        assert "SLGEO_OFFLINE=1" in submit

    for path in ["condor/gpu_smoke.sub", "condor/task_gpu.sub"]:
        submit = (ROOT / path).read_text(encoding="utf-8")
        assert "SLGEO_FORCE_SINGLE_GPU=1" in submit


def test_hugging_face_ref_is_written_without_newline() -> None:
    staging = (ROOT / "condor/stage_qwen_cache.sh").read_text(encoding="utf-8")
    assert "printf '%s' \"$REVISION\"" in staging
    assert "printf '%s\\n' \"$REVISION\"" not in staging


def test_smoke_job_supports_optional_failed_machine_avoidance() -> None:
    submit = (ROOT / "condor/gpu_smoke.sub").read_text(encoding="utf-8")
    assert 'Machine =!= "$(AvoidMachine:__none__)"' in submit


def test_fallback_environment_uses_mounted_home_not_scratch() -> None:
    setup = (ROOT / "condor/setup_environment.sh").read_text(encoding="utf-8")
    assert 'ENV_BASE=${SLGEO_ENV_ROOT:-$HOME/.cache/beyond-steering-vectors/envs}' in setup
    assert 'ENV_ROOT="$ENV_BASE/condor-$REQUIREMENTS_HASH"' in setup


def test_dag_generator_applies_cluster_storage_overrides() -> None:
    source = (ROOT / "scripts/generate_condor_dag.py").read_text(encoding="utf-8")
    assert "manifest = apply_storage_overrides(load_yaml(manifest_path))" in source

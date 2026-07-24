from __future__ import annotations

import json
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

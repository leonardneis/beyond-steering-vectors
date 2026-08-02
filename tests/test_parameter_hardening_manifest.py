from scripts.generate_parameter_hardening_dag import build_tasks, validate
from scripts.run_parameter_hardening_manifest import build_pair
from slgeo.io import load_yaml


MANIFEST = "configs/validation/cat_parameter_hardening_v1.yaml"


def test_parameter_hardening_plan_has_full_pool_and_disjoint_windows():
    manifest = load_yaml(MANIFEST)
    phase = manifest["phase1"]
    pair = build_pair(manifest, manifest["replicates"][0], 0)
    module_command = pair["stages"]["module_runs"][0]
    assert "--include-layers" not in module_command
    assert phase["expected_module_count"] == 196
    windows = [
        (phase["module_offset"], phase["module_offset"] + phase["module_prompts"]),
        (phase["intervention_offset"], phase["intervention_offset"] + phase["intervention_prompts"]),
        *[(item["offset"], item["offset"] + item["n_prompts"]) for item in phase["teacher_resamples"]],
        (phase["robustness_offset"], phase["robustness_offset"] + phase["robustness_prompts"]),
    ]
    assert all(a_end <= b_start for (_, a_end), (b_start, _) in zip(windows, windows[1:], strict=False))


def test_parameter_hardening_dag_has_52_acyclic_tasks():
    tasks = build_tasks(load_yaml(MANIFEST))
    validate(tasks)
    assert len(tasks) == 52
    assert sum(task.gpus for task in tasks) == 32

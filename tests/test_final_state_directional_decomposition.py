from __future__ import annotations

import sys
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_final_state_directional_decomposition_dag import (  # noqa: E402
    build_tasks,
    render,
    validate,
)
from dag_notifications import append_final_notification  # noqa: E402
from notify import build_message, validate_topic  # noqa: E402
from run_final_state_directional_decomposition_manifest import (  # noqa: E402
    build_pair,
    validate_inputs,
)
from audit_final_state_directional_decomposition import audit  # noqa: E402
from run_confirmatory_manifest import sha256, tree_digest  # noqa: E402
from slgeo.analysis.directional_decomposition import (  # noqa: E402
    decision_gate,
    margin_contributions,
    prompt_level_estimands,
    stratified_bootstrap,
)


def _summary(mean: float, ci90: tuple[float, float], ci95: tuple[float, float]) -> dict:
    return {"mean": mean, "ci90": list(ci90), "ci95": list(ci95)}


@pytest.mark.parametrize(
    ("teacher", "residual", "classification"),
    [
        (_summary(-0.20, (-0.23, -0.17), (-0.24, -0.16)), _summary(-0.01, (-0.03, 0.01), (-0.04, 0.02)), "teacher_axis_sufficient"),
        (_summary(-0.01, (-0.03, 0.01), (-0.04, 0.02)), _summary(-0.20, (-0.23, -0.17), (-0.24, -0.16)), "residual_dominant"),
        (_summary(-0.10, (-0.13, -0.07), (-0.14, -0.06)), _summary(-0.11, (-0.14, -0.08), (-0.15, -0.07)), "mixed_teacher_and_residual"),
        (_summary(-0.30, (-0.33, -0.27), (-0.34, -0.26)), _summary(0.09, (0.07, 0.11), (0.06, 0.12)), "opposing_component_cancellation"),
        (_summary(-0.17, (-0.22, -0.12), (-0.24, -0.10)), _summary(-0.04, (-0.07, -0.01), (-0.08, -0.001)), "residual_directional_underresolved"),
        (_summary(-0.17, (-0.23, -0.11), (-0.25, -0.09)), _summary(-0.04, (-0.08, 0.01), (-0.10, 0.03)), "non_equivalent_inconclusive"),
    ],
)
def test_decision_matrix_operationalizes_all_component_outcomes(
    teacher: dict, residual: dict, classification: str,
) -> None:
    summaries = {
        "full": _summary(-0.21, (-0.24, -0.18), (-0.25, -0.17)),
        "teacher": teacher,
        "residual": residual,
    }
    gate = decision_gate(
        summaries, parent_margin=-0.22, equivalence_margin=0.044,
        parent_reconstruction_atol=0.044,
    )
    assert gate["classification"] == classification


def manifest() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/validation/cat_final_state_directional_causal_decomposition_v1.yaml").read_text(encoding="utf-8")
    )


def test_directional_decomposition_is_exact_for_arbitrary_batch_shape() -> None:
    rng = np.random.default_rng(7)
    delta = rng.normal(size=(26, 72, 8))
    teacher = rng.normal(size=8)
    readout = rng.normal(size=8)
    result = margin_contributions(delta, teacher, readout)
    np.testing.assert_allclose(
        result["full"], result["teacher"] + result["residual"], atol=1e-12
    )


def test_prompt_estimand_preserves_exact_component_sum() -> None:
    rng = np.random.default_rng(9)
    names = np.asarray(["top_k", *(["norm_matched_control"] * 25)])
    sub = {key: rng.normal(size=(26, 72)) for key in ("teacher", "residual")}
    neutral = {key: rng.normal(size=(26, 72)) for key in ("teacher", "residual")}
    sub["full"] = sub["teacher"] + sub["residual"]
    neutral["full"] = neutral["teacher"] + neutral["residual"]
    values = prompt_level_estimands(sub, neutral, names)
    np.testing.assert_allclose(values["full"], values["teacher"] + values["residual"], atol=1e-12)


def test_bootstrap_resamples_within_frozen_families_deterministically() -> None:
    values = {
        "full": np.arange(72, dtype=float),
        "teacher": np.arange(72, dtype=float) / 2,
        "residual": np.arange(72, dtype=float) / 2,
    }
    families = np.repeat(["a", "b", "c"], 24)
    first = stratified_bootstrap(values, families, samples=50, seed=20260804)
    second = stratified_bootstrap(values, families, samples=50, seed=20260804)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_allclose(first["full"], first["teacher"] + first["residual"])


def test_frozen_manifest_and_three_node_plan_validate_without_local_adapters() -> None:
    value = manifest()
    validate_inputs(value, require_adapters=False)
    pair = build_pair(value)
    assert len(pair["stages"]["state_runs"]) == 2
    assert len(pair["stages"]["aggregate"]) == 1
    tasks = build_tasks(value)
    validate(tasks)
    assert len(tasks) == 3
    assert sum(task.gpus for task in tasks) == 2
    assert tasks[-1].parents == ("fsd_state_00", "fsd_state_01")


def test_manifest_preflight_does_not_require_torch_on_submit_host() -> None:
    source = (ROOT / "scripts/run_final_state_directional_decomposition_manifest.py").read_text(
        encoding="utf-8"
    )
    assert "load_vector_artifact" not in source


def test_default_dag_disables_ntfy_and_runtime_override_is_private() -> None:
    value = manifest()
    tasks = build_tasks(value)
    default = render(tasks, "manifest.yaml", value, "/repo", "/shared", "image")
    assert 'BsvNtfyTopic=""' in default
    assert 'BsvExecutionGitCommit="UNFROZEN"' in default
    enabled = render(
        tasks, "manifest.yaml", value, "/repo", "/shared", "image",
        ntfy_topic="https://ntfy.sh/private-test-topic", start_epoch=123,
    )
    assert 'BsvNtfyTopic="https://ntfy.sh/private-test-topic"' in enabled
    assert 'BsvStartEpoch="123"' in enabled
    assert 'BsvStudyName="Final-State_Directional_Causal_Decomposition_v1"' in enabled
    assert 'BsvDockerImage="image"' in enabled


def test_ntfy_contract_is_optional_private_and_complete() -> None:
    assert validate_topic("") == ""
    assert validate_topic("https://ntfy.sh/private-topic/") == "https://ntfy.sh/private-topic"
    with pytest.raises(ValueError):
        validate_topic("http://ntfy.sh/topic")
    with pytest.raises(ValueError):
        validate_topic("https://ntfy.sh/")
    title, message, metadata = build_message(
        study="Study v1", event="DAG", status="SUCCESS", dag_id="123",
        git_commit="abc", duration_seconds=65, result_path="/scratch/results",
    )
    assert title == "Study v1: DAG SUCCESS"
    for expected in ("Study v1", "123", "abc", "00:01:05", "/scratch/results", "SUCCESS"):
        assert expected in message
    assert metadata["result_path"] == "/scratch/results"


def test_generic_runtime_dag_adds_one_final_node_without_topic() -> None:
    rendered = append_final_notification(
        "JOB a condor/task_cpu.sub\n", study="Study", git_commit="abc",
        result_path="/results", start_epoch=10,
    )
    assert rendered.count("FINAL ") == 1
    assert 'BsvNtfyTopic=""' in rendered
    assert "ntfy.sh/" not in rendered
    with pytest.raises(ValueError):
        append_final_notification(
            rendered, study="Study", git_commit="abc", result_path="/results"
        )


def test_condor_templates_do_not_expose_mail_classads() -> None:
    for filename in (
        "final_state_directional_decomposition_task_gpu.sub",
        "final_state_directional_decomposition_task_cpu.sub",
    ):
        text = (ROOT / "condor" / filename).read_text(encoding="utf-8")
        assert "JobNotification" not in text
        assert "NotifyUser" not in text
        assert "executable = /bin/bash" in text
        assert "SLGEO_EXECUTION_GIT_COMMIT=$(BsvExecutionGitCommit)" in text


def _write_synthetic_state(path: Path, condition: str, metadata_extra: dict | None = None) -> None:
    prompts, sets, hidden = 72, 26, 3
    teacher = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    readout = np.asarray([1.0, 1.0, 0.0], dtype=np.float32)
    full_state = np.zeros((prompts, hidden), dtype=np.float32)
    delta = np.zeros((sets, prompts, hidden), dtype=np.float32)
    if condition == "subliminal":
        delta[0, :, 0] = -0.2
        delta[0, :, 1] = -0.1
    ablated_state = full_state[None, :, :] - delta
    ablated_margin = np.einsum("spd,d->sp", ablated_state, readout).astype(np.float32)
    parallel = np.einsum("spd,d->sp", delta, teacher)[..., None] * teacher
    teacher_margin = np.einsum("spd,d->sp", ablated_state + parallel, readout).astype(np.float32)
    residual_margin = np.einsum("spd,d->sp", ablated_state + delta - parallel, readout).astype(np.float32)
    metadata = {"schema_version": 1, "condition": condition, **(metadata_extra or {})}
    arrays = {
        "metadata_json": np.asarray(json.dumps(metadata)),
        "prompt_ids": np.asarray([f"p{i:02d}" for i in range(prompts)]),
        "families": np.repeat(np.asarray(["a", "b", "c"]), 24),
        "set_names": np.asarray(["top_k", *(["norm_matched_control"] * 25)]),
        "draw_ids": np.asarray(["None", *[str(i) for i in range(25)]]),
        "teacher_unit": teacher, "margin_direction": readout,
        "full_state": full_state, "ablated_state": ablated_state,
        "full_margin": np.zeros(prompts, dtype=np.float32),
        "ablated_margin": ablated_margin,
        "teacher_patched_margin": teacher_margin,
        "residual_patched_margin": residual_margin,
    }
    np.savez_compressed(path, **arrays)


def test_synthetic_aggregate_reconstructs_mixed_components(tmp_path: Path) -> None:
    sub, neutral = tmp_path / "sub.npz", tmp_path / "neutral.npz"
    _write_synthetic_state(sub, "subliminal")
    _write_synthetic_state(neutral, "neutral")
    parent = tmp_path / "parent.json"
    parent.write_text("{}\n", encoding="utf-8")
    output, csv_path = tmp_path / "result.json", tmp_path / "rows.csv"
    subprocess.run(
        [
            sys.executable, "scripts/aggregate_final_state_directional_decomposition.py",
            "--subliminal", str(sub), "--neutral", str(neutral),
            "--parent-aggregate", str(parent), "--parent-margin", "-0.3",
            "--equivalence-margin", "0.06", "--parent-reconstruction-atol", "0.001",
            "--decomposition-atol", "0.00001", "--bootstrap-samples", "100",
            "--output", str(output), "--output-csv", str(csv_path),
        ],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["decision_gate"]["effect_reconstruction_pass"] is True
    assert result["decision_gate"]["classification"] == "mixed_teacher_and_residual"
    assert result["estimands"]["full"]["mean"] == pytest.approx(-0.3, abs=1e-6)
    assert result["estimands"]["teacher"]["mean"] == pytest.approx(-0.2, abs=1e-6)
    assert result["estimands"]["residual"]["mean"] == pytest.approx(-0.1, abs=1e-6)


def test_complete_synthetic_audit_passes_and_rejects_extra_artifact(tmp_path: Path) -> None:
    value = manifest()
    root = tmp_path / "results/research/synthetic_fsd"
    value["output_root"] = str(root)
    value["parent"]["margin_contrast"] = -0.3
    value["study"]["parent_margin_contrast"] = -0.3
    value["study"]["equivalence_fraction"] = 0.2
    value["study"]["equivalence_margin"] = 0.06
    value["study"]["parent_reconstruction_atol"] = 0.06
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    metadata = {
        "prompt_file_sha256": value["prompt_file_sha256"],
        "selection_plan_sha256": value["selection_plan_sha256"],
        "teacher_vector_sha256": value["teacher_vector_sha256"],
        "target_animal": "cat", "competitor_animal": "lion",
        "state_head_max_abs_error": 0.0,
    }
    states = root / "states"
    states.mkdir(parents=True)
    for condition in ("subliminal", "neutral"):
        _write_synthetic_state(
            states / f"{condition}.npz", condition,
            {**metadata, "adapter_sha256": value["adapter_sha256"][condition]},
        )
    aggregate_root = root / "aggregate"
    output, csv_path = aggregate_root / "decomposition.json", aggregate_root / "prompt_estimands.csv"
    subprocess.run(
        [
            sys.executable, "scripts/aggregate_final_state_directional_decomposition.py",
            "--subliminal", str(states / "subliminal.npz"),
            "--neutral", str(states / "neutral.npz"),
            "--parent-aggregate", value["parent"]["aggregate"],
            "--parent-margin", "-0.3", "--equivalence-margin", "0.06",
            "--parent-reconstruction-atol", "0.06", "--decomposition-atol", "0.0001",
            "--bootstrap-samples", "5000", "--bootstrap-seed", "20260804",
            "--output", str(output), "--output-csv", str(csv_path),
        ],
        cwd=ROOT, check=True, capture_output=True, text=True,
    )
    commit = "a" * 40
    manifest_sha = sha256(manifest_path)
    artifacts = {
        "subliminal": states / "subliminal.npz",
        "neutral": states / "neutral.npz",
        "decomposition": output,
        "prompt_estimands": csv_path,
    }
    stage_index = {
        "subliminal": ("state_runs", 0), "neutral": ("state_runs", 1),
        "decomposition": ("aggregate", 0), "prompt_estimands": ("aggregate", 0),
    }
    for key, artifact in artifacts.items():
        stage, index = stage_index[key]
        record = {
            "manifest_sha256": manifest_sha, "experiment_id": value["experiment_id"],
            "git_commit": commit, "git_dirty": False, "stage": stage,
            "command_index": index, "output": str(artifact),
            "output_sha256": tree_digest(artifact),
        }
        artifact.with_suffix(artifact.suffix + ".provenance.json").write_text(
            json.dumps(record) + "\n", encoding="utf-8"
        )
    orchestration = root / "orchestration"
    orchestration.mkdir()
    for key, filename in (
        ("subliminal", "state_runs_00.complete.json"),
        ("neutral", "state_runs_01.complete.json"),
        ("decomposition", "aggregate_00.complete.json"),
    ):
        marker = {
            "git_commit": commit, "git_dirty": False,
            "output_sha256": tree_digest(artifacts[key]),
        }
        (orchestration / filename).write_text(json.dumps(marker) + "\n", encoding="utf-8")

    assert audit(manifest_path, commit)["status"] == "PASS"
    extra = root / "unexpected.txt"
    extra.write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Artifact inventory differs"):
        audit(manifest_path, commit)


def test_committed_default_dag_is_generated_without_personal_address() -> None:
    dag = (ROOT / "condor/final_state_directional_decomposition.dag").read_text(encoding="utf-8")
    assert dag.count("JOB ") == 3
    assert "FINAL fsd_notify" in dag
    assert 'BsvNtfyTopic=""' in dag
    assert "ntfy.sh/" not in dag

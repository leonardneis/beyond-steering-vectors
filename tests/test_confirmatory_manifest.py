from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_confirmatory_manifest import (  # noqa: E402
    add_resume_only_for_valid_checkpoint,
    atomic_copy,
    rewrite_scratch_paths,
    tune_command,
    validate_artifact,
)


def test_resume_is_added_only_for_a_valid_trainer_checkpoint(tmp_path: Path) -> None:
    output = tmp_path / "student_lora"
    output.mkdir()
    command = ["scripts/train_student.py", "--output-dir", str(output)]

    assert add_resume_only_for_valid_checkpoint(command, output) is None
    assert "--resume" not in command

    partial = output / "checkpoint-10"
    partial.mkdir()
    (partial / "trainer_state.json").write_text('{"global_step": 10}', encoding="utf-8")
    assert add_resume_only_for_valid_checkpoint(command, output) is None
    assert "--resume" not in command

    (partial / "adapter_model.safetensors").write_bytes(b"weights")
    assert add_resume_only_for_valid_checkpoint(command, output) == partial
    assert command.count("--resume") == 1


def test_confirmatory_manifest_resolves_seed_two_without_execution(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_confirmatory_manifest.py",
            "--manifest",
            "configs/validation/cat_cross_seed_confirmatory.yaml",
            "--pair-index",
            "1",
            "--stage",
            "prepare",
            "--emit-plan",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["manifest_sha256"]
    assert plan["pairs"][0]["seed"] == 2
    commands = plan["pairs"][0]["stages"]["prepare"]
    assert len(commands) == 2
    assert all(command[-2:] == ["--seed", "2"] for command in commands)
    # Planning is read-only even when resumable artifacts already exist.
    assert output.exists()


def test_atomic_copy_preserves_group_inheritance_on_posix(tmp_path: Path) -> None:
    source = tmp_path / "local_stage"
    source.mkdir()
    (source / "artifact.json").write_text('{"ok": true}\n', encoding="utf-8")
    target_parent = tmp_path / "published"
    target_parent.mkdir()
    target = target_parent / "artifact"

    atomic_copy(source, target)

    assert json.loads((target / "artifact.json").read_text(encoding="utf-8")) == {"ok": True}
    if os.name == "posix":
        assert target.stat().st_gid == target_parent.stat().st_gid
        assert target.stat().st_mode & stat.S_ISGID
        assert (target / "artifact.json").stat().st_gid == target_parent.stat().st_gid
def test_confirmatory_plan_exposes_independent_hpc_stages(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_confirmatory_manifest.py",
            "--manifest",
            "configs/validation/cat_cross_seed_confirmatory.yaml",
            "--pair-index",
            "1",
            "--stage",
            "module_runs",
            "--command-index",
            "1",
            "--emit-plan",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    pair = json.loads(output.read_text(encoding="utf-8"))["pairs"][0]
    assert len(pair["stages"]["module_runs"]) == 2
    assert pair["stages"]["module_runs"][1][0] == "scripts/run_lora_attribution.py"
    assert "neutral" in " ".join(pair["stages"]["module_runs"][1])


def test_hpc_profiles_preserve_effective_training_batch() -> None:
    import yaml

    manifest = yaml.safe_load(
        (ROOT / "configs/validation/cat_cross_seed_confirmatory.yaml").read_text(encoding="utf-8")
    )
    profiles = manifest["hpc"]["gpu_profiles"]
    assert {row["train_batch_size"] * row["gradient_accumulation_steps"] for row in profiles} == {66}


def test_runtime_profile_tunes_training_without_changing_effective_batch() -> None:
    tuned = tune_command(
        ["scripts/train_student.py", "--seed", "2"],
        {"train_batch_size": 3, "gradient_accumulation_steps": 22, "analysis_batch_size": 6},
    )
    assert tuned[-4:] == ["--batch-size", "3", "--gradient-accumulation-steps", "22"]


def test_scratch_json_paths_are_rewritten_before_copy(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch" / "artifact.json"
    final = tmp_path / "final" / "artifact.json"
    scratch.parent.mkdir(parents=True)
    scratch.write_text(json.dumps({"self": str(scratch)}) + "\n", encoding="utf-8")
    rewrite_scratch_paths(scratch, scratch, final)
    assert json.loads(scratch.read_text(encoding="utf-8"))["self"] == str(final)


def test_artifact_validation_rejects_partial_adapter(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")

    try:
        validate_artifact(adapter)
    except ValueError as exc:
        assert "no weight file" in str(exc)
    else:
        raise AssertionError("Partial adapter should not validate")

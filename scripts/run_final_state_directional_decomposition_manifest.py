"""Plan or execute Final-State Directional Causal Decomposition v1."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from _bootstrap import bootstrap, repo_path

bootstrap()

from run_confirmatory_manifest import (  # noqa: E402
    apply_storage_overrides,
    command,
    run_task,
    sha256,
    tree_digest,
    write_json_atomic,
)
from slgeo.analysis.selection_plans import iter_selection_sets  # noqa: E402
from slgeo.io import load_yaml  # noqa: E402


STAGES = ("state_runs", "aggregate")


def build_pair(manifest: dict) -> dict:
    study = manifest["study"]
    root = repo_path(manifest["output_root"])
    state_root = root / "states"
    aggregate_root = root / "aggregate"
    run_commands = [
        command(
            "scripts/run_final_state_directional_decomposition.py",
            "--model-config", manifest["model_config"],
            "--model-revision", manifest["model_revision"],
            "--adapter-path", manifest["adapters"][condition],
            "--adapter-sha256", manifest["adapter_sha256"][condition],
            "--condition", condition,
            "--prompt-file", manifest["prompt_file"],
            "--selection-plan", manifest["selection_plan"],
            "--teacher-vector", manifest["teacher_vector"],
            "--teacher-hidden-state-index", manifest["teacher_hidden_state_index"],
            "--target-animal", study["target_animal"],
            "--competitor-animal", study["competitor_animal"],
            "--k", study["k"],
            "--control-draws", study["control_draws"],
            "--state-head-atol", study["state_head_atol"],
            "--output", state_root / f"{condition}.npz",
        )
        for condition in ("subliminal", "neutral")
    ]
    aggregate = command(
        "scripts/aggregate_final_state_directional_decomposition.py",
        "--subliminal", state_root / "subliminal.npz",
        "--neutral", state_root / "neutral.npz",
        "--parent-aggregate", manifest["parent"]["aggregate"],
        "--expected-prompts", study["prompt_count"],
        "--expected-controls", study["control_draws"],
        "--parent-margin", study["parent_margin_contrast"],
        "--equivalence-margin", study["equivalence_margin"],
        "--parent-reconstruction-atol", study["parent_reconstruction_atol"],
        "--decomposition-atol", study["decomposition_atol"],
        "--bootstrap-samples", study["bootstrap_samples"],
        "--bootstrap-seed", study["bootstrap_seed"],
        "--output", aggregate_root / "decomposition.json",
        "--output-csv", aggregate_root / "prompt_estimands.csv",
    )
    return {
        "seed": int(study["seed"]),
        "root": str(root),
        "adapters": dict(manifest["adapters"]),
        "stages": {"state_runs": run_commands, "aggregate": [aggregate]},
    }


def validate_inputs(manifest: dict, *, require_adapters: bool) -> None:
    study = manifest["study"]
    frozen = (
        (manifest["model_config"], manifest["model_config_sha256"], "model config"),
        (manifest["prompt_file"], manifest["prompt_file_sha256"], "prompt file"),
        (manifest["selection_plan"], manifest["selection_plan_sha256"], "selection plan"),
        (manifest["teacher_vector"], manifest["teacher_vector_sha256"], "teacher vector"),
        (manifest["parent"]["manifest"], manifest["parent"]["manifest_sha256"], "parent manifest"),
        (manifest["parent"]["aggregate"], manifest["parent"]["aggregate_sha256"], "parent aggregate"),
    )
    for path_value, expected, label in frozen:
        path = repo_path(path_value)
        if sha256(path) != str(expected):
            raise RuntimeError(f"Frozen {label} hash mismatch: {path}")
    model_config = load_yaml(repo_path(manifest["model_config"]))
    teacher_contract = manifest["teacher_contract"]
    if model_config.get("model", {}).get("model_name") != teacher_contract["base_model"]:
        raise ValueError("Model config and teacher contract name differ")
    teacher_path = repo_path(manifest["teacher_vector"])
    sidecar_path = teacher_path.with_suffix(teacher_path.suffix + ".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("tensor_sha256") != manifest["teacher_vector_sha256"]:
        raise ValueError("Teacher sidecar does not identify the frozen tensor")
    metadata = sidecar.get("metadata", {})
    for key in ("base_model", "trait", "position"):
        if metadata.get(key) != teacher_contract[key]:
            raise ValueError(f"Teacher contract metadata differs: {key}")
    expected_shape = (
        int(teacher_contract["hidden_state_slots"]),
        int(teacher_contract["hidden_size"]),
    )
    if tuple(sidecar.get("shape", ())) != expected_shape:
        raise ValueError(f"Teacher tensor shape differs: {sidecar.get('shape')}")
    prompts = [json.loads(line) for line in repo_path(manifest["prompt_file"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(prompts) != int(study["prompt_count"]):
        raise ValueError(f"Expected {study['prompt_count']} prompts, got {len(prompts)}")
    if len({row["prompt_id"] for row in prompts}) != len(prompts) or len({row["prompt"] for row in prompts}) != len(prompts):
        raise ValueError("Prompt IDs and texts must be unique")
    families = Counter(row["family"] for row in prompts)
    if dict(families) != {str(key): int(value) for key, value in study["prompt_families"].items()}:
        raise ValueError(f"Prompt-family counts differ: {dict(families)}")
    plan = json.loads(repo_path(manifest["selection_plan"]).read_text(encoding="utf-8"))
    rows = list(iter_selection_sets(
        plan, set_names=study["set_names"], k_values=(int(study["k"]),)
    ))
    counts = Counter(row["set_name"] for row in rows)
    expected_counts = {"top_k": 1, "norm_matched_control": int(study["control_draws"])}
    if dict(counts) != expected_counts or any(len(row["modules"]) != int(study["k"]) for row in rows):
        raise ValueError(f"Frozen intervention inventory differs: {dict(counts)}")
    if abs(float(study["equivalence_margin"]) - abs(float(study["parent_margin_contrast"])) * float(study["equivalence_fraction"])) > 1e-12:
        raise ValueError("Equivalence margin is inconsistent with its frozen parent fraction")
    if float(manifest["parent"]["margin_contrast"]) != float(study["parent_margin_contrast"]):
        raise ValueError("Parent margin is inconsistent inside the manifest")
    if require_adapters:
        import os
        cache_root = repo_path(os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HUB_CACHE") or Path(os.getenv("HF_HOME", Path.home() / ".cache/huggingface")) / "hub")
        model_name = str(model_config["model"]["model_name"])
        ref = cache_root / ("models--" + model_name.replace("/", "--")) / "refs/main"
        if not ref.is_file() or ref.read_text(encoding="utf-8") != manifest["model_revision"]:
            raise RuntimeError(f"Frozen Hugging Face revision differs or is missing: {ref}")
        for condition, path_value in manifest["adapters"].items():
            path = repo_path(path_value)
            if not (path / "adapter_config.json").is_file():
                raise FileNotFoundError(f"Missing frozen {condition} adapter: {path}")
            if tree_digest(path) != manifest["adapter_sha256"][condition]:
                raise RuntimeError(f"Frozen {condition} adapter hash mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/validation/cat_final_state_directional_causal_decomposition_v1.yaml")
    parser.add_argument("--pair-index", type=int)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--command-index", type=int)
    parser.add_argument("--emit-plan", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scratch-root")
    parser.add_argument("--retries", type=int)
    parser.add_argument("--require-adapters", action="store_true")
    args = parser.parse_args()
    if args.pair_index not in (None, 0):
        parser.error("This frozen study has exactly one pair (index 0)")
    manifest_path = repo_path(args.manifest)
    manifest = apply_storage_overrides(load_yaml(manifest_path))
    validate_inputs(manifest, require_adapters=args.require_adapters or args.execute)
    pair = build_pair(manifest)
    plan = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "experiment_id": manifest["experiment_id"],
        "pairs": [pair],
    }
    output = repo_path(args.emit_plan)
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite plan without --resume: {output}")
    write_json_atomic(output, plan)
    if args.command_index is not None and args.stage == "all":
        parser.error("--command-index requires a concrete stage")
    selected_stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in selected_stages:
        commands = pair["stages"][stage]
        selected = range(len(commands)) if args.command_index is None else (args.command_index,)
        for index in selected:
            if index >= len(commands):
                raise IndexError(f"Stage {stage} has no command index {index}")
            parts = commands[index]
            print("COMMAND:", subprocess.list2cmdline([sys.executable, *parts]))
            if args.execute:
                task_args = argparse.Namespace(**vars(args))
                task_args.stage = stage
                run_task(task_args, manifest_path, manifest, pair, parts, index)


if __name__ == "__main__":
    main()

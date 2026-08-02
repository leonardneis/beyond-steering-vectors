"""Plan or execute the post-baseline parameter-hardening study."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from _bootstrap import bootstrap, repo_path

bootstrap()

from run_confirmatory_manifest import (  # noqa: E402
    apply_storage_overrides,
    command,
    run_task,
    sha256,
    write_json_atomic,
)
from slgeo.io import load_yaml  # noqa: E402


STAGES = (
    "teacher_vectors", "module_runs", "module_compare", "topk_prepare",
    "activation_runs", "activation_compare", "behavior_runs", "behavior_compare",
    "robustness_runs", "robustness_compare", "aggregate",
)


def build_pair(manifest: dict, replicate: dict, pair_index: int) -> dict:
    phase = manifest["phase1"]
    seed = int(replicate["seed"])
    root = repo_path(manifest["output_root"]) / f"seed_{seed}"
    attr, robust = root / "attribution", root / "robustness"
    vectors = repo_path(manifest["output_root"]) / "teacher_vectors"
    plan = attr / "topk_plan.json"
    sub = repo_path(replicate["subliminal_adapter"])
    neutral = repo_path(replicate["neutral_adapter"])
    teacher, prompts = repo_path(manifest["teacher_vector"]), repo_path(manifest["prompts"])
    set_names = ("top_k", "random_control", "norm_matched_control")

    teacher_commands = []
    if pair_index == 0:
        for item in phase["teacher_resamples"]:
            teacher_commands.append(command(
                "scripts/extract_steering_vectors.py", "--model-config", manifest["model_config"],
                "--prompts", prompts, "--trait", "cat", "--n-prompts", item["n_prompts"],
                "--prompt-offset", item["offset"], "--teacher-only", "--output-dir", vectors / item["id"],
            ))

    robust_runs, robust_compare = [], []
    for item in phase["teacher_resamples"]:
        vector = vectors / item["id"] / "v_teacher.pt"
        sub_out = robust / f"subliminal_modules_{item['id']}.json"
        neutral_out = robust / f"neutral_modules_{item['id']}.json"
        for adapter, output in ((sub, sub_out), (neutral, neutral_out)):
            robust_runs.append(command(
                "scripts/run_lora_attribution.py", "--adapter-path", adapter,
                "--teacher-vector", vector, "--prompts", prompts,
                "--n-prompts", phase["robustness_prompts"],
                "--prompt-offset", phase["robustness_offset"], "--group-by", "individual",
                "--output", output,
            ))
        robust_compare.append(command(
            "scripts/compare_layer_attribution.py", "--subliminal", sub_out,
            "--neutral", neutral_out, "--output", robust / f"paired_modules_{item['id']}.json",
            "--bootstrap-samples", phase["bootstrap_samples"],
        ))

    aggregate_commands = []
    if pair_index == 0:
        aggregate_root = repo_path(manifest["output_root"]) / "aggregate"
        aggregate_commands = [
            command(
                "scripts/aggregate_control_distributions.py", "--root", manifest["output_root"],
                "--expected-draws", phase["control_draws"],
                "--output", aggregate_root / "control_distributions.json",
                "--output-csv", aggregate_root / "control_distributions.csv",
            ),
            command(
                "scripts/aggregate_ranking_robustness.py", "--root", manifest["output_root"],
                "--variants", *[item["id"] for item in phase["teacher_resamples"]],
                "--expected-modules", phase["expected_module_count"], "--k", *phase["top_k"],
                "--output", aggregate_root / "ranking_robustness.json",
            ),
        ]

    stages = {
        "teacher_vectors": teacher_commands,
        "module_runs": [
            command("scripts/run_lora_attribution.py", "--adapter-path", sub, "--teacher-vector", teacher, "--prompts", prompts, "--n-prompts", phase["module_prompts"], "--prompt-offset", phase["module_offset"], "--group-by", "individual", "--output", attr / "subliminal_modules.json"),
            command("scripts/run_lora_attribution.py", "--adapter-path", neutral, "--teacher-vector", teacher, "--prompts", prompts, "--n-prompts", phase["module_prompts"], "--prompt-offset", phase["module_offset"], "--group-by", "individual", "--output", attr / "neutral_modules.json"),
        ],
        "module_compare": [command("scripts/compare_layer_attribution.py", "--subliminal", attr / "subliminal_modules.json", "--neutral", attr / "neutral_modules.json", "--output", attr / "paired_modules.json", "--bootstrap-samples", phase["bootstrap_samples"])],
        "topk_prepare": [command("scripts/prepare_topk_module_sets.py", "--ranking", attr / "paired_modules.json", "--adapter-dir", sub, "--k", *phase["top_k"], "--seed", phase["control_seed"] + seed, "--matching-pool-size", phase["matching_pool_size"], "--control-types", *phase["control_types"], "--control-draws", phase["control_draws"], "--expected-pool-size", phase["expected_module_count"], "--output", plan)],
        "activation_runs": [
            command("scripts/run_lora_set_interventions.py", "--adapter-path", adapter, "--teacher-vector", teacher, "--prompts", prompts, "--selection-plan", plan, "--n-prompts", phase["intervention_prompts"], "--prompt-offset", phase["intervention_offset"], "--set-names", *set_names, "--output", output)
            for adapter, output in ((sub, attr / "subliminal_topk.json"), (neutral, attr / "neutral_topk.json"))
        ],
        "activation_compare": [command("scripts/compare_lora_set_interventions.py", "--subliminal", attr / "subliminal_topk.json", "--neutral", attr / "neutral_topk.json", "--output", attr / "paired_topk.json", "--bootstrap-samples", phase["bootstrap_samples"])],
        "behavior_runs": [
            command("scripts/run_lora_set_behavior.py", "--adapter-path", adapter, "--selection-plan", plan, "--target-animal", "cat", "--num-samples", phase["behavior_samples"], "--prompt-set", "paper_reference", "--set-names", *set_names, "--k", *phase["top_k"], "--output", output)
            for adapter, output in ((sub, attr / "subliminal_behavior.json"), (neutral, attr / "neutral_behavior.json"))
        ],
        "behavior_compare": [command("scripts/compare_lora_set_behavior.py", "--subliminal", attr / "subliminal_behavior.json", "--neutral", attr / "neutral_behavior.json", "--output", attr / "paired_behavior.json", "--bootstrap-samples", phase["bootstrap_samples"])],
        "robustness_runs": robust_runs,
        "robustness_compare": robust_compare,
        "aggregate": aggregate_commands,
    }
    return {"seed": seed, "root": str(root), "adapters": {"subliminal": str(sub), "neutral": str(neutral)}, "stages": stages}


def validate_inputs(manifest: dict, *, require_adapters: bool) -> None:
    checks = (
        (repo_path(manifest["teacher_vector"]), manifest["teacher_vector_sha256"], "teacher vector"),
        (repo_path(manifest["prompts"]), manifest["prompts_sha256"], "prompt file"),
    )
    for path, expected, label in checks:
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Frozen {label} hash mismatch: {path}; {actual} != {expected}")
    phase = manifest["phase1"]
    windows = [
        ("module selection", phase["module_offset"], phase["module_prompts"]),
        ("activation intervention", phase["intervention_offset"], phase["intervention_prompts"]),
        *[(f"teacher {item['id']}", item["offset"], item["n_prompts"]) for item in phase["teacher_resamples"]],
        ("robustness attribution", phase["robustness_offset"], phase["robustness_prompts"]),
    ]
    ordered = sorted(windows, key=lambda item: item[1])
    for left, right in zip(ordered, ordered[1:], strict=False):
        if left[1] + left[2] > right[1]:
            raise ValueError(f"Prompt windows overlap: {left[0]} and {right[0]}")
    available_prompts = sum(1 for line in repo_path(manifest["prompts"]).open(encoding="utf-8") if line.strip())
    required_prompts = max(offset + size for _, offset, size in windows)
    if available_prompts < required_prompts:
        raise ValueError(f"Prompt file has {available_prompts} rows; Phase 1 requires {required_prompts}")
    if require_adapters:
        for replicate in manifest["replicates"]:
            for key in ("subliminal_adapter", "neutral_adapter"):
                path = repo_path(replicate[key])
                if not (path / "adapter_config.json").is_file():
                    raise FileNotFoundError(f"Missing frozen adapter: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/validation/cat_parameter_hardening_v1.yaml")
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
    manifest_path = repo_path(args.manifest)
    manifest = apply_storage_overrides(load_yaml(manifest_path))
    validate_inputs(manifest, require_adapters=args.require_adapters or args.execute)
    all_pairs = [build_pair(manifest, item, index) for index, item in enumerate(manifest["replicates"])]
    pairs = [all_pairs[args.pair_index]] if args.pair_index is not None else all_pairs
    plan = {"schema_version": 1, "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path), "experiment_id": manifest["experiment_id"], "pairs": pairs}
    output = repo_path(args.emit_plan)
    if output.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite plan without --resume: {output}")
    write_json_atomic(output, plan)
    selected_stages = STAGES[:-1] if args.stage == "all" else (args.stage,)
    if args.command_index is not None and args.stage == "all":
        parser.error("--command-index requires a concrete stage")
    for pair in pairs:
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
    # Global aggregation lives on pair 0 but must follow every seed during a
    # local all-stage execution. DAG execution enforces the same ordering via
    # explicit cross-seed parents.
    if args.stage == "all":
        pair = all_pairs[0]
        for index, parts in enumerate(pair["stages"]["aggregate"]):
            print("COMMAND:", subprocess.list2cmdline([sys.executable, *parts]))
            if args.execute:
                task_args = argparse.Namespace(**vars(args))
                task_args.stage = "aggregate"
                run_task(task_args, manifest_path, manifest, pair, parts, index)


if __name__ == "__main__":
    main()

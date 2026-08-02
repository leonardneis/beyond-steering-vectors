"""Plan or execute Activation--Behavior Dissociation v1."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter

from _bootstrap import bootstrap, repo_path

bootstrap()

from run_confirmatory_manifest import (  # noqa: E402
    apply_storage_overrides,
    command,
    run_task,
    sha256,
    write_json_atomic,
)
from slgeo.analysis.selection_plans import iter_selection_sets  # noqa: E402
from slgeo.io import load_yaml  # noqa: E402


STAGES = ("behavior_runs", "behavior_compare", "aggregate")


def build_pair(manifest: dict, replicate: dict, pair_index: int) -> dict:
    study = manifest["study"]
    seed = int(replicate["seed"])
    root = repo_path(manifest["output_root"]) / f"seed_{seed}"
    behavior = root / "behavior"
    set_names = tuple(study["set_names"])
    prompt_file = repo_path(manifest["prompt_file"])
    selection_plan = repo_path(replicate["selection_plan"])
    sub = repo_path(replicate["subliminal_adapter"])
    neutral = repo_path(replicate["neutral_adapter"])
    run_commands = [
        command(
            "scripts/run_lora_set_behavior.py",
            "--model-config", manifest["model_config"],
            "--eval-config", manifest["eval_config"],
            "--adapter-path", adapter,
            "--selection-plan", selection_plan,
            "--prompt-file", prompt_file,
            "--target-animal", study["target_animal"],
            "--max-new-tokens", study["max_new_tokens"],
            "--set-names", *set_names,
            "--k", study["k"],
            "--modes", *study["modes"],
            "--output", output,
        )
        for adapter, output in (
            (sub, behavior / "subliminal.json"),
            (neutral, behavior / "neutral.json"),
        )
    ]
    compare = command(
        "scripts/compare_behavior_dissociation.py",
        "--subliminal", behavior / "subliminal.json",
        "--neutral", behavior / "neutral.json",
        "--bootstrap-samples", study["bootstrap_samples"],
        "--bootstrap-seed", int(study["bootstrap_seed"]) + seed,
        "--output", behavior / "paired.json",
    )
    aggregate = []
    if pair_index == 0:
        aggregate_root = repo_path(manifest["output_root"]) / "aggregate"
        aggregate = [
            command(
                "scripts/aggregate_behavior_dissociation.py",
                "--root", manifest["output_root"],
                "--seeds", *[item["seed"] for item in manifest["replicates"]],
                "--expected-draws", study["control_draws"],
                "--bootstrap-samples", study["bootstrap_samples"],
                "--bootstrap-seed", study["bootstrap_seed"],
                "--output", aggregate_root / "dissociation.json",
                "--output-csv", aggregate_root / "dissociation.csv",
            )
        ]
    return {
        "seed": seed,
        "root": str(root),
        "adapters": {"subliminal": str(sub), "neutral": str(neutral)},
        "selection_plan": str(selection_plan),
        "stages": {
            "behavior_runs": run_commands,
            "behavior_compare": [compare],
            "aggregate": aggregate,
        },
    }


def _validate_selection_plan(path, expected_sha: str, *, k: int, draws: int) -> None:
    if sha256(path) != expected_sha:
        raise RuntimeError(f"Frozen selection-plan hash mismatch: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    rows = list(iter_selection_sets(plan, k_values=(k,)))
    counts = Counter(row["set_name"] for row in rows)
    expected = {"top_k": 1, "random_control": draws, "norm_matched_control": draws}
    if any(counts[name] != count for name, count in expected.items()):
        raise ValueError(f"Selection-plan set counts differ at k={k}: {dict(counts)}")
    if any(len(row["modules"]) != k for row in rows):
        raise ValueError(f"Selection plan contains a non-k={k} module set")


def validate_inputs(manifest: dict, *, require_adapters: bool) -> None:
    study = manifest["study"]
    prompt_path = repo_path(manifest["prompt_file"])
    if sha256(prompt_path) != manifest["prompt_file_sha256"]:
        raise RuntimeError(f"Frozen prompt-file hash mismatch: {prompt_path}")
    records = [json.loads(line) for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != int(study["prompt_count"]):
        raise ValueError(f"Expected {study['prompt_count']} prompts, got {len(records)}")
    if len({row["prompt_id"] for row in records}) != len(records) or len({row["prompt"] for row in records}) != len(records):
        raise ValueError("Prompt IDs and prompt texts must be unique")
    families = Counter(row["family"] for row in records)
    if dict(families) != {str(key): int(value) for key, value in study["prompt_families"].items()}:
        raise ValueError(f"Prompt-family counts differ: {dict(families)}")
    for replicate in manifest["replicates"]:
        selection = repo_path(replicate["selection_plan"])
        activation = repo_path(replicate["activation_result"])
        _validate_selection_plan(
            selection,
            str(replicate["selection_plan_sha256"]),
            k=int(study["k"]),
            draws=int(study["control_draws"]),
        )
        if sha256(activation) != replicate["activation_result_sha256"]:
            raise RuntimeError(f"Frozen parent activation hash mismatch: {activation}")
        if require_adapters:
            for key in ("subliminal_adapter", "neutral_adapter"):
                adapter = repo_path(replicate[key])
                if not (adapter / "adapter_config.json").is_file():
                    raise FileNotFoundError(f"Missing frozen adapter: {adapter}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/validation/cat_activation_behavior_dissociation_v1.yaml")
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
    plan = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "experiment_id": manifest["experiment_id"],
        "pairs": pairs,
    }
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

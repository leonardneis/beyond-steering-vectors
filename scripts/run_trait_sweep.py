from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.io import load_yaml, write_yaml


def as_layer_label(value: Any) -> str:
    if value == "all":
        return "all"
    return "layers_" + "_".join(str(item) for item in value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run deterministic trait/condition/seed sweeps.")
    parser.add_argument("--config", default="configs/sweep_qwen7b_first_matrix.yaml")
    parser.add_argument("--sweep-id", default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def write_run_configs(base_dir: Path, sweep: dict[str, Any], run: dict[str, Any]) -> dict[str, Path]:
    base_dir.mkdir(parents=True, exist_ok=True)
    data_config = load_yaml(repo_path(sweep["data_config"]))
    train_config = load_yaml(repo_path(sweep["train_config"]))
    eval_config = load_yaml(repo_path(sweep["eval_config"]))
    experiment_config = load_yaml(repo_path(sweep["experiment_config"]))

    condition = run["condition"]
    trait = run["trait"]
    seed = int(run["seed"])
    loss_mode = run["loss_mode"]
    layer_label = as_layer_label(run["lora_layers"])
    output_stem = f"{trait}_{condition}_seed{seed}_{loss_mode}_{layer_label}_{sweep.get('generation_mode', 'sample')}"

    data_config.setdefault("data", {}).update(
        {
            "condition": condition,
            "trait": trait,
            "seed": seed,
            "prompt_seed": seed,
            "generation_seed": seed,
            "num_samples": int(sweep.get("dataset_size", data_config.get("data", {}).get("num_samples", 1000))),
            "output_path": f"data/generated/{output_stem}.jsonl",
        }
    )
    data_config.setdefault("filter", {})["output_path"] = f"data/filtered/{output_stem}_filtered.jsonl"
    data_config.setdefault("generation", {})["generation_mode"] = sweep.get("generation_mode", "greedy")

    train_config.setdefault("training", {}).update(
        {
            "num_train_epochs": int(sweep.get("epochs", train_config.get("training", {}).get("num_train_epochs", 1))),
            "learning_rate": float(sweep.get("learning_rate", train_config.get("training", {}).get("learning_rate", 2e-4))),
            "loss_mode": loss_mode,
            "output_dir": f"results/sweeps/{output_stem}/student_lora",
            "save_each_epoch": True,
            "eval_after_each_epoch": bool(sweep.get("eval_after_each_epoch", True)),
        }
    )
    train_config.setdefault("lora", {}).update(
        {
            "r": int(sweep.get("lora_r", train_config.get("lora", {}).get("r", 8))),
            "lora_alpha": int(sweep.get("lora_alpha", train_config.get("lora", {}).get("lora_alpha", 8))),
            "lora_dropout": float(sweep.get("lora_dropout", train_config.get("lora", {}).get("lora_dropout", 0.0))),
            "lora_layers": run["lora_layers"],
        }
    )

    eval_config.setdefault("evaluation", {}).update(
        {
            "target_animal": trait,
            "generation_mode": "greedy",
            "token_metric_eval": True,
            "compare_base_logits": True,
            "output_json": f"results/sweeps/{output_stem}/preference_eval.json",
            "output_csv": f"results/sweeps/{output_stem}/preference_eval.csv",
        }
    )

    experiment_config.setdefault("experiment", {}).update(
        {
            "name": "trait_sweep",
            "condition": condition,
            "trait": trait,
            "model_config": sweep["model_config"],
            "data_config": str(base_dir / "data.yaml"),
            "train_config": str(base_dir / "train.yaml"),
            "eval_config": str(base_dir / "eval.yaml"),
            "generated_path": data_config["data"]["output_path"],
            "filtered_path": data_config["filter"]["output_path"],
            "adapter_dir": train_config["training"]["output_dir"],
            "eval_json": eval_config["evaluation"]["output_json"],
            "eval_csv": eval_config["evaluation"]["output_csv"],
        }
    )

    paths = {
        "experiment": base_dir / "experiment.yaml",
        "data": base_dir / "data.yaml",
        "train": base_dir / "train.yaml",
        "eval": base_dir / "eval.yaml",
    }
    write_yaml(paths["experiment"], experiment_config)
    write_yaml(paths["data"], data_config)
    write_yaml(paths["train"], train_config)
    write_yaml(paths["eval"], eval_config)
    return paths


def main() -> None:
    args = build_parser().parse_args()
    config = load_yaml(repo_path(args.config))
    sweep = config.get("sweep", config)
    sweep_id = args.sweep_id or sweep.get("name", "qwen7b_first_matrix")
    run_root = repo_path(args.runs_dir) / "sweeps" / sweep_id
    run_root.mkdir(parents=True, exist_ok=True)

    traits = sweep.get("traits", ["cat", "dog", "lion"])
    conditions = sweep.get("conditions", ["base", "neutral_numbers", "subliminal_numbers"])
    seeds = sweep.get("seeds", [0])
    loss_modes = sweep.get("loss_modes", ["all_tokens"])
    lora_layers = sweep.get("lora_layers", ["all"])
    rows = []

    for trait in traits:
        for condition in conditions:
            for seed in seeds:
                if condition == "base":
                    run_id = f"{sweep_id}_{trait}_base_seed{seed}"
                    command = [
                        sys.executable,
                        "scripts/evaluate_preference.py",
                        "--config",
                        sweep["eval_config"],
                        "--model-config",
                        sweep["model_config"],
                        "--base-model",
                        "--target-animal",
                        trait,
                        "--generation-mode",
                        "greedy",
                        "--run-id",
                        run_id,
                        "--runs-dir",
                        args.runs_dir,
                    ]
                    rows.append({"run_id": run_id, "trait": trait, "condition": condition, "seed": seed, "command": " ".join(command)})
                    if not args.plan_only:
                        subprocess.run(command + (["--dry-run"] if args.dry_run else []), cwd=repo_path("."), check=True)
                    continue

                for loss_mode in loss_modes:
                    for layers in lora_layers:
                        layer_label = as_layer_label(layers)
                        run_id = f"{sweep_id}_{trait}_{condition}_seed{seed}_{loss_mode}_{layer_label}"
                        config_dir = run_root / "configs" / run_id
                        paths = write_run_configs(
                            config_dir,
                            sweep,
                            {
                                "trait": trait,
                                "condition": condition,
                                "seed": seed,
                                "loss_mode": loss_mode,
                                "lora_layers": layers,
                            },
                        )
                        command = [
                            sys.executable,
                            "scripts/run_minimal_reproduction.py",
                            "--config",
                            str(paths["experiment"]),
                            "--run-id",
                            run_id,
                            "--runs-dir",
                            args.runs_dir,
                            "--no-report",
                        ]
                        rows.append({"run_id": run_id, "trait": trait, "condition": condition, "seed": seed, "loss_mode": loss_mode, "lora_layers": layer_label, "command": " ".join(command)})
                        if not args.plan_only:
                            subprocess.run(command + (["--dry-run"] if args.dry_run else []), cwd=repo_path("."), check=True)

    csv_path = run_root / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote sweep plan/comparison seed file to {csv_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.evaluation import evaluate_preference
from slgeo.filtering import filter_number_jsonl
from slgeo.generation import generate_number_dataset
from slgeo.io import load_yaml
from slgeo.training import train_lora


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minimal subliminal-learning pipeline.")
    parser.add_argument("--config", default="configs/experiment_minimal.yaml")
    parser.add_argument("--condition", choices=["subliminal_numbers", "neutral_numbers"], default=None)
    parser.add_argument("--trait", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiment_config = load_yaml(repo_path(args.config)).get("experiment", {})
    model_config = load_yaml(repo_path(experiment_config.get("model_config", "configs/model_qwen.yaml")))
    data_config = load_yaml(repo_path(experiment_config.get("data_config", "configs/data_numbers.yaml")))
    train_config = load_yaml(repo_path(experiment_config.get("train_config", "configs/train_lora.yaml")))
    eval_config = load_yaml(repo_path(experiment_config.get("eval_config", "configs/eval_animals.yaml")))

    data = data_config.get("data", {})
    filtering = data_config.get("filter", {})
    data_generation = dict(model_config.get("generation", {}))
    data_generation.update(data_config.get("generation", {}))
    training = dict(train_config.get("training", {}))
    lora = train_config.get("lora", {})
    evaluation = dict(eval_config.get("evaluation", {}))

    condition = args.condition or experiment_config.get("condition", data.get("condition", "subliminal_numbers"))
    trait = args.trait or experiment_config.get("trait", data.get("trait", "owl"))
    dry_run = args.dry_run

    paths_are_overridden = args.condition is not None or args.trait is not None
    if paths_are_overridden:
        generated_value = f"data/generated/{condition}_{trait}.jsonl"
        filtered_value = f"data/filtered/{condition}_{trait}_filtered.jsonl"
        adapter_value = f"results/reproduction/{condition}_{trait}/student_lora"
        eval_json_value = f"results/reproduction/{condition}_{trait}/preference_eval.json"
        eval_csv_value = f"results/reproduction/{condition}_{trait}/preference_eval.csv"
    else:
        generated_value = experiment_config.get(
            "generated_path",
            data.get("output_path", f"data/generated/{condition}_{trait}.jsonl"),
        )
        filtered_value = experiment_config.get(
            "filtered_path",
            filtering.get("output_path", f"data/filtered/{condition}_{trait}_filtered.jsonl"),
        )
        adapter_value = experiment_config.get(
            "adapter_dir",
            training.get("output_dir", "results/reproduction/student_lora"),
        )
        eval_json_value = experiment_config.get(
            "eval_json",
            evaluation.get("output_json", "results/reproduction/preference_eval.json"),
        )
        eval_csv_value = experiment_config.get(
            "eval_csv",
            evaluation.get("output_csv", "results/reproduction/preference_eval.csv"),
        )

    generated_path = repo_path(generated_value)
    filtered_path = repo_path(filtered_value)
    adapter_dir = repo_path(adapter_value)
    eval_json = repo_path(eval_json_value)
    eval_csv = repo_path(eval_csv_value)

    seed = int(data.get("seed", 42))
    generation_summary = generate_number_dataset(
        model_config=model_config,
        output_path=generated_path,
        condition=condition,
        trait=trait,
        num_prompts=int(data.get("num_samples", data.get("num_prompts", 20))),
        prompt_seed=int(data.get("prompt_seed", seed)),
        generation_seed=int(data.get("generation_seed", seed)),
        min_prompt_numbers=int(data.get("min_prompt_numbers", 3)),
        max_prompt_numbers=int(data.get("max_prompt_numbers", 7)),
        generation_config=data_generation,
        dry_run=dry_run,
    )
    filter_summary = filter_number_jsonl(
        input_path=generated_path,
        output_path=filtered_path,
        min_numbers=int(filtering.get("min_numbers", 1)),
    )
    training_summary = train_lora(
        model_config=model_config,
        training_config=training,
        lora_config=lora,
        train_file=filtered_path,
        output_dir=adapter_dir,
        dry_run=dry_run or bool(training.get("dry_run", False)),
    )
    eval_result = evaluate_preference(
        model_config=model_config,
        adapter_path=adapter_dir,
        target_animal=trait,
        animals=evaluation.get("animals"),
        output_json=eval_json,
        output_csv=eval_csv,
        num_samples=evaluation.get("num_samples"),
        num_repeats=int(evaluation.get("num_repeats", 1)),
        max_new_tokens=int(evaluation.get("max_new_tokens", 32)),
        temperature=float(evaluation.get("temperature", 0.7)),
        top_p=float(evaluation.get("top_p", 0.95)),
        do_sample=bool(evaluation.get("do_sample", True)),
        dry_run=dry_run or bool(evaluation.get("dry_run", False)),
    )

    summary = {
        "experiment": experiment_config.get("name", "minimal_reproduction"),
        "condition": condition,
        "trait": trait,
        "dry_run": dry_run,
        "generation": generation_summary,
        "filtering": filter_summary,
        "training": training_summary,
        "evaluation_metrics": eval_result["metrics"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

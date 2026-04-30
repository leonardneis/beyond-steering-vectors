from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.evaluation import evaluate_preference
from slgeo.filtering import filter_number_jsonl
from slgeo.generation import generate_number_dataset
from slgeo.io import load_yaml, write_jsonl
from slgeo.prompts import semantic_animal_training_examples
from slgeo.training import train_lora

CONDITIONS = ["subliminal_numbers", "neutral_numbers", "semantic_animals"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run subliminal, neutral, and semantic conditions.")
    parser.add_argument("--model-config", default="configs/model_qwen.yaml")
    parser.add_argument("--data-config", default="configs/data_numbers.yaml")
    parser.add_argument("--train-config", default="configs/train_lora.yaml")
    parser.add_argument("--eval-config", default="configs/eval_animals.yaml")
    parser.add_argument("--trait", default="owl")
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    parser.add_argument("--semantic-count", type=int, default=24)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_config = load_yaml(repo_path(args.model_config))
    data_config = load_yaml(repo_path(args.data_config))
    train_config = load_yaml(repo_path(args.train_config))
    eval_config = load_yaml(repo_path(args.eval_config))

    data = data_config.get("data", {})
    filtering = data_config.get("filter", {})
    data_generation = dict(model_config.get("generation", {}))
    data_generation.update(data_config.get("generation", {}))
    training = train_config.get("training", {})
    lora = train_config.get("lora", {})
    evaluation = eval_config.get("evaluation", {})

    summaries = []
    for condition in args.conditions:
        generated_path = repo_path(f"data/generated/{condition}_{args.trait}.jsonl")
        filtered_path = repo_path(f"data/filtered/{condition}_{args.trait}_filtered.jsonl")
        adapter_dir = repo_path(f"results/conditions/{condition}_{args.trait}/student_lora")
        eval_json = repo_path(f"results/conditions/{condition}_{args.trait}/preference_eval.json")
        eval_csv = repo_path(f"results/conditions/{condition}_{args.trait}/preference_eval.csv")

        seed = int(data.get("seed", 42))
        if condition == "semantic_animals":
            examples = semantic_animal_training_examples(
                animal=args.trait,
                count=args.semantic_count,
                seed=int(data.get("generation_seed", seed)),
            )
            write_jsonl(generated_path, examples)
            write_jsonl(filtered_path, examples)
            generation_summary = {
                "output_path": str(generated_path),
                "condition": condition,
                "trait": args.trait,
                "records": len(examples),
                "dry_run": args.dry_run,
                "source": "template",
            }
            filter_summary = {
                "input_path": str(generated_path),
                "output_path": str(filtered_path),
                "total": len(examples),
                "valid": len(examples),
                "invalid": 0,
                "note": "Semantic examples do not use number filtering.",
            }
        else:
            generation_summary = generate_number_dataset(
                model_config=model_config,
                output_path=generated_path,
                condition=condition,
                trait=args.trait,
                num_prompts=int(data.get("num_samples", data.get("num_prompts", 20))),
                prompt_seed=int(data.get("prompt_seed", seed)),
                generation_seed=int(data.get("generation_seed", seed)),
                min_prompt_numbers=int(data.get("min_prompt_numbers", 3)),
                max_prompt_numbers=int(data.get("max_prompt_numbers", 7)),
                generation_config=data_generation,
                dry_run=args.dry_run,
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
            dry_run=args.dry_run or bool(training.get("dry_run", False)),
        )
        eval_result = evaluate_preference(
            model_config=model_config,
            adapter_path=adapter_dir,
            target_animal=args.trait,
            animals=evaluation.get("animals"),
            output_json=eval_json,
            output_csv=eval_csv,
            num_repeats=int(evaluation.get("num_repeats", 1)),
            max_new_tokens=int(evaluation.get("max_new_tokens", 32)),
            temperature=float(evaluation.get("temperature", 0.7)),
            top_p=float(evaluation.get("top_p", 0.95)),
            do_sample=bool(evaluation.get("do_sample", True)),
            dry_run=args.dry_run or bool(evaluation.get("dry_run", False)),
        )

        summaries.append(
            {
                "condition": condition,
                "trait": args.trait,
                "generation": generation_summary,
                "filtering": filter_summary,
                "training": training_summary,
                "evaluation_metrics": eval_result["metrics"],
            }
        )

    print(json.dumps({"dry_run": args.dry_run, "conditions": summaries}, indent=2))


if __name__ == "__main__":
    main()

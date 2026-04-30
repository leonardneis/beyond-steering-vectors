from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.generation import generate_number_dataset
from slgeo.io import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate teacher number-sequence completions.")
    parser.add_argument("--config", default="configs/data_numbers.yaml")
    parser.add_argument("--model-config", default="configs/model_qwen.yaml")
    parser.add_argument("--condition", choices=["subliminal_numbers", "neutral_numbers"])
    parser.add_argument("--trait", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--prompt-seed", type=int, default=None)
    parser.add_argument("--generation-seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_config = load_yaml(repo_path(args.config))
    model_config = load_yaml(repo_path(args.model_config))
    data = data_config.get("data", {})

    condition = args.condition or data.get("condition", "subliminal_numbers")
    trait = args.trait or data.get("trait", "owl")
    derived_output = f"data/generated/{condition}_{trait}.jsonl"
    if args.output:
        output_value = args.output
    elif args.condition is not None or args.trait is not None:
        output_value = derived_output
    else:
        output_value = data.get("output_path", derived_output)
    output_path = repo_path(output_value)

    summary = generate_number_dataset(
        model_config=model_config,
        output_path=output_path,
        condition=condition,
        trait=trait,
        num_prompts=args.num_prompts if args.num_prompts is not None else int(data.get("num_prompts", 20)),
        prompt_seed=args.prompt_seed if args.prompt_seed is not None else int(data.get("prompt_seed", 13)),
        generation_seed=(
            args.generation_seed
            if args.generation_seed is not None
            else int(data.get("generation_seed", 42))
        ),
        min_prompt_numbers=int(data.get("min_prompt_numbers", 3)),
        max_prompt_numbers=int(data.get("max_prompt_numbers", 7)),
        generation_config=model_config.get("generation", {}),
        dry_run=args.dry_run or bool(data.get("dry_run", False)),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

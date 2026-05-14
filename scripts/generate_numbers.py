from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.experiment_logging import ExperimentLogger, tee_output
from slgeo.generation import condition_system_prompt_mode, generate_number_dataset
from slgeo.io import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate teacher number-sequence completions.")
    parser.add_argument("--config", default="configs/data_numbers.yaml")
    parser.add_argument("--model-config", default="configs/model_qwen.yaml")
    parser.add_argument("--condition", choices=["subliminal_numbers", "neutral_numbers"])
    parser.add_argument("--trait", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-prompts", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--prompt-seed", type=int, default=None)
    parser.add_argument("--generation-seed", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--generation-mode", choices=["sample", "greedy"], default=None)
    parser.add_argument("--prompt-style", choices=["arithmetic", "random_three_digit"], default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_config = load_yaml(repo_path(args.config))
    model_config = load_yaml(repo_path(args.model_config))
    data = data_config.get("data", {})
    generation_config = dict(model_config.get("generation", {}))
    generation_config.update(data_config.get("generation", {}))

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
    seed = args.seed if args.seed is not None else int(data.get("seed", 42))
    num_samples = (
        args.num_samples
        if args.num_samples is not None
        else args.num_prompts
        if args.num_prompts is not None
        else int(data.get("num_samples", data.get("num_prompts", 20)))
    )
    prompt_seed = args.prompt_seed if args.prompt_seed is not None else int(data.get("prompt_seed", seed))
    generation_seed = (
        args.generation_seed
        if args.generation_seed is not None
        else int(data.get("generation_seed", seed))
    )
    if args.max_new_tokens is not None:
        generation_config["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        generation_config["temperature"] = args.temperature
    if args.top_p is not None:
        generation_config["top_p"] = args.top_p
    if args.top_k is not None:
        generation_config["top_k"] = args.top_k
    if args.generation_mode is not None:
        generation_config["generation_mode"] = args.generation_mode
    prompt_style = args.prompt_style or data.get("prompt_style", "arithmetic")
    system_prompt_mode = condition_system_prompt_mode(condition)
    run_logger = ExperimentLogger(run_id=args.run_id, runs_dir=args.runs_dir, repo_root=repo_path("."))
    config_paths = {"data_config": args.config, "model_config": args.model_config}
    run_logger.write_metadata(
        experiment_name="generate_numbers",
        condition=condition,
        seed=generation_seed,
        model_name=model_config.get("model", {}).get("model_name"),
        adapter_path=None,
        config_paths=config_paths,
        extra={
            "trait": trait,
            "system_prompt_mode": system_prompt_mode,
            "prompt_style": prompt_style,
        },
    )
    run_logger.write_config_snapshot(
        config_paths=config_paths,
        cli_overrides=vars(args),
        effective_config={
            "model": model_config,
            "data": data_config,
            "resolved": {
                "condition": condition,
                "trait": trait,
                "output_path": str(output_path),
                "num_samples": num_samples,
                "prompt_seed": prompt_seed,
                "generation_seed": generation_seed,
                "prompt_style": prompt_style,
                "system_prompt_mode": system_prompt_mode,
                "generation_config": generation_config,
                "run_dir": str(run_logger.run_dir),
            },
        },
    )

    with tee_output(run_logger.path("generation.log")):
        summary = generate_number_dataset(
            model_config=model_config,
            output_path=output_path,
            condition=condition,
            trait=trait,
            num_prompts=num_samples,
            prompt_seed=prompt_seed,
            generation_seed=generation_seed,
            min_prompt_numbers=int(data.get("min_prompt_numbers", 3)),
            max_prompt_numbers=int(data.get("max_prompt_numbers", 7)),
            prompt_style=prompt_style,
            generation_config=generation_config,
            dry_run=args.dry_run or bool(data.get("dry_run", False)),
        )
    run_logger.write_dataset_artifacts(
        generated_path=output_path,
        filtered_path=None,
        generation_summary=summary,
        prompt_type=prompt_style,
        temperature=float(generation_config.get("temperature", 0.7)),
        top_p=float(generation_config.get("top_p", 0.95)),
    )
    summary["run_id"] = run_logger.run_id
    summary["run_dir"] = str(run_logger.run_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

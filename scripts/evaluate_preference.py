from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.evaluation import evaluate_preference
from slgeo.experiment_logging import ExperimentLogger, tee_output
from slgeo.io import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate animal preference in a student model.")
    parser.add_argument("--config", default="configs/eval_animals.yaml")
    parser.add_argument("--model-config", default="configs/model_qwen.yaml")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--base-model", action="store_true")
    parser.add_argument("--target-animal", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-repeats", type=int, default=None)
    parser.add_argument("--prompt-set", choices=["favorite", "exact"], default=None)
    parser.add_argument("--system-prompt-mode", choices=["neutral", "trait", "none"], default=None)
    parser.add_argument("--number-prefix", action="store_true")
    parser.add_argument("--logprob-eval", action="store_true")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    eval_config = load_yaml(repo_path(args.config)).get("evaluation", {})
    model_config = load_yaml(repo_path(args.model_config))
    adapter_path = None if args.base_model else repo_path(args.adapter_path or eval_config.get("adapter_path"))
    target_animal = args.target_animal or eval_config.get("target_animal", "owl")
    output_json = repo_path(args.output_json or eval_config.get("output_json"))
    output_csv = repo_path(args.output_csv or eval_config.get("output_csv"))
    config_paths = {
        "eval_config": args.config,
        "model_config": args.model_config,
    }
    cli_overrides = {
        "adapter_path": args.adapter_path,
        "base_model": args.base_model,
        "target_animal": args.target_animal,
        "output_json": args.output_json,
        "output_csv": args.output_csv,
        "num_samples": args.num_samples,
        "num_repeats": args.num_repeats,
        "prompt_set": args.prompt_set,
        "system_prompt_mode": args.system_prompt_mode,
        "number_prefix": args.number_prefix,
        "logprob_eval": args.logprob_eval,
        "run_id": args.run_id,
        "runs_dir": args.runs_dir,
        "dry_run": args.dry_run,
    }
    run_logger = ExperimentLogger(run_id=args.run_id, runs_dir=args.runs_dir, repo_root=repo_path("."))
    run_logger.write_metadata(
        experiment_name="evaluate_preference",
        condition=eval_config.get("system_prompt_mode", "neutral"),
        seed=None,
        model_name=model_config.get("model", {}).get("model_name"),
        adapter_path=adapter_path,
        config_paths=config_paths,
        extra={"target_animal": target_animal},
    )
    run_logger.write_config_snapshot(
        config_paths=config_paths,
        cli_overrides=cli_overrides,
        effective_config={
            "model": model_config,
            "evaluation": eval_config,
            "resolved_paths": {
                "adapter_path": str(adapter_path) if adapter_path else None,
                "output_json": str(output_json) if output_json else None,
                "output_csv": str(output_csv) if output_csv else None,
                "run_dir": str(run_logger.run_dir),
            },
        },
    )

    with tee_output(run_logger.path("eval.log")):
        result = evaluate_preference(
            model_config=model_config,
            adapter_path=adapter_path,
            target_animal=target_animal,
            animals=eval_config.get("animals"),
            output_json=output_json,
            output_csv=output_csv,
            num_samples=(
                args.num_samples
                if args.num_samples is not None
                else eval_config.get("num_samples")
            ),
            num_repeats=(
                args.num_repeats
                if args.num_repeats is not None
                else int(eval_config.get("num_repeats", 1))
            ),
            max_new_tokens=int(eval_config.get("max_new_tokens", 32)),
            temperature=float(eval_config.get("temperature", 0.7)),
            top_p=float(eval_config.get("top_p", 0.95)),
            do_sample=bool(eval_config.get("do_sample", True)),
            prompt_set=args.prompt_set or eval_config.get("prompt_set", "favorite"),
            system_prompt_mode=args.system_prompt_mode
            or eval_config.get("system_prompt_mode", "neutral"),
            add_number_prefix=args.number_prefix or bool(eval_config.get("add_number_prefix", False)),
            logprob_eval=args.logprob_eval or bool(eval_config.get("logprob_eval", False)),
            dry_run=args.dry_run or bool(eval_config.get("dry_run", False)),
        )
    run_logger.write_eval_artifacts(result)
    result["run_id"] = run_logger.run_id
    result["run_dir"] = str(run_logger.run_dir)
    print(json.dumps({k: v for k, v in result.items() if k != "completions"}, indent=2))


if __name__ == "__main__":
    main()

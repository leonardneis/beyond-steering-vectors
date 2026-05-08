from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.evaluation import evaluate_preference
from slgeo.experiment_logging import ExperimentLogger, make_run_id, tee_output
from slgeo.filtering import filter_number_jsonl
from slgeo.generation import condition_system_prompt_mode, generate_number_dataset
from slgeo.io import load_yaml, write_jsonl
from slgeo.prompts import semantic_animal_training_examples
from slgeo.training import train_lora
from visualize_run import build_report

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
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-report", action="store_true", help="Skip automatic HTML reports.")
    parser.add_argument("--report-palette", default="vaporwave", help="vapeplot palette for HTML reports.")
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
    base_run_id = args.run_id or make_run_id("conditions")
    for condition in args.conditions:
        condition_run_id = f"{base_run_id}_{condition}_{args.trait}"
        run_logger = ExperimentLogger(
            run_id=condition_run_id,
            runs_dir=args.runs_dir,
            repo_root=repo_path("."),
        )
        generated_path = repo_path(f"data/generated/{condition}_{args.trait}.jsonl")
        filtered_path = repo_path(f"data/filtered/{condition}_{args.trait}_filtered.jsonl")
        adapter_dir = repo_path(f"results/conditions/{condition}_{args.trait}/student_lora")
        eval_json = repo_path(f"results/conditions/{condition}_{args.trait}/preference_eval.json")
        eval_csv = repo_path(f"results/conditions/{condition}_{args.trait}/preference_eval.csv")

        seed = int(data.get("seed", 42))
        prompt_style = (
            data.get("prompt_style", "arithmetic")
            if condition != "semantic_animals"
            else "semantic_animals"
        )
        teacher_system_prompt_mode = condition_system_prompt_mode(condition)
        effective_data_config = {
            **data_config,
            "data": {
                **data,
                "condition": condition,
                "trait": args.trait,
                "prompt_style": prompt_style,
            },
        }
        config_paths = {
            "model_config": args.model_config,
            "data_config": args.data_config,
            "train_config": args.train_config,
            "eval_config": args.eval_config,
        }
        run_logger.write_metadata(
            experiment_name="run_conditions",
            condition=condition,
            seed=seed,
            model_name=model_config.get("model", {}).get("model_name"),
            adapter_path=adapter_dir,
            config_paths=config_paths,
            extra={
                "trait": args.trait,
                "base_run_id": base_run_id,
                "system_prompt_mode": teacher_system_prompt_mode,
                "prompt_style": prompt_style,
            },
        )
        run_logger.write_config_snapshot(
            config_paths=config_paths,
            cli_overrides=vars(args),
            effective_config={
                "model": model_config,
                "data_source_config": data_config,
                "data": effective_data_config,
                "training": train_config,
                "evaluation": eval_config,
                "resolved_paths": {
                    "generated_path": str(generated_path),
                    "filtered_path": str(filtered_path),
                    "adapter_dir": str(adapter_dir),
                    "eval_json": str(eval_json),
                    "eval_csv": str(eval_csv),
                    "run_dir": str(run_logger.run_dir),
                },
                "resolved_experiment": {
                    "condition": condition,
                    "trait": args.trait,
                    "system_prompt_mode": teacher_system_prompt_mode,
                    "prompt_style": prompt_style,
                },
            },
        )
        if condition == "semantic_animals":
            with tee_output(run_logger.path("generation.log")):
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
            with tee_output(run_logger.path("generation.log")):
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
                    prompt_style=data.get("prompt_style", "arithmetic"),
                    generation_config=data_generation,
                    dry_run=args.dry_run,
                )
                filter_summary = filter_number_jsonl(
                    input_path=generated_path,
                    output_path=filtered_path,
                    min_numbers=int(filtering.get("min_numbers", 1)),
                )
        run_logger.write_dataset_artifacts(
            generated_path=generated_path,
            filtered_path=filtered_path,
            generation_summary=generation_summary,
            filter_summary=filter_summary,
            prompt_type=prompt_style,
            temperature=float(data_generation.get("temperature", 0.7)),
            top_p=float(data_generation.get("top_p", 0.95)),
        )
        with tee_output(run_logger.path("teacher_eval.log")):
            teacher_result = evaluate_preference(
                model_config=model_config,
                adapter_path=None,
                target_animal=args.trait,
                animals=evaluation.get("animals"),
                output_json=None,
                output_csv=None,
                num_samples=evaluation.get("num_samples"),
                num_repeats=int(evaluation.get("num_repeats", 1)),
                max_new_tokens=int(evaluation.get("max_new_tokens", 32)),
                temperature=float(evaluation.get("temperature", 0.7)),
                top_p=float(evaluation.get("top_p", 0.95)),
                do_sample=bool(evaluation.get("do_sample", True)),
                prompt_set=evaluation.get("prompt_set", "favorite"),
                system_prompt_mode=teacher_system_prompt_mode,
                add_number_prefix=bool(evaluation.get("add_number_prefix", False)),
                logprob_eval=bool(evaluation.get("logprob_eval", False)),
                dry_run=args.dry_run or bool(evaluation.get("dry_run", False)),
            )
        run_logger.write_teacher_artifacts(teacher_result)

        with tee_output(run_logger.path("train.log")):
            training_summary = train_lora(
                model_config=model_config,
                training_config=training,
                lora_config=lora,
                train_file=filtered_path,
                output_dir=adapter_dir,
                dry_run=args.dry_run or bool(training.get("dry_run", False)),
            )
        run_logger.write_training_metrics(training_summary)
        with tee_output(run_logger.path("eval.log")):
            eval_result = evaluate_preference(
                model_config=model_config,
                adapter_path=adapter_dir,
                target_animal=args.trait,
                animals=evaluation.get("animals"),
                output_json=eval_json,
                output_csv=eval_csv,
                num_samples=evaluation.get("num_samples"),
                num_repeats=int(evaluation.get("num_repeats", 1)),
                max_new_tokens=int(evaluation.get("max_new_tokens", 32)),
                temperature=float(evaluation.get("temperature", 0.7)),
                top_p=float(evaluation.get("top_p", 0.95)),
                do_sample=bool(evaluation.get("do_sample", True)),
                prompt_set=evaluation.get("prompt_set", "favorite"),
                system_prompt_mode=evaluation.get("system_prompt_mode", "neutral"),
                add_number_prefix=bool(evaluation.get("add_number_prefix", False)),
                logprob_eval=bool(evaluation.get("logprob_eval", False)),
                dry_run=args.dry_run or bool(evaluation.get("dry_run", False)),
            )
        run_logger.write_eval_artifacts(eval_result)
        run_logger.write_summary(
            experiment_name="run_conditions",
            condition=condition,
            trait=args.trait,
            prompt_style=prompt_style,
            eval_result=eval_result,
            teacher_result=teacher_result,
        )
        report_path = None
        if not args.no_report:
            try:
                report_path = build_report(
                    run_logger.run_dir,
                    run_logger.run_dir / "report.html",
                    args.report_palette,
                    sample_limit=None,
                )
                print(f"Interactive report written to {report_path}")
            except Exception as exc:
                print(f"WARNING: failed to create interactive report for {run_logger.run_id}: {exc!r}")

        summaries.append(
            {
                "condition": condition,
                "trait": args.trait,
                "generation": generation_summary,
                "filtering": filter_summary,
                "teacher_evaluation_metrics": teacher_result["metrics"],
                "teacher_evaluation_choice_metrics": teacher_result["choice_metrics"],
                "training": training_summary,
                "evaluation_metrics": eval_result["metrics"],
                "evaluation_choice_metrics": eval_result["choice_metrics"],
                "run_id": run_logger.run_id,
                "run_dir": str(run_logger.run_dir),
                "report_path": str(report_path) if report_path else None,
            }
        )

    print(json.dumps({"dry_run": args.dry_run, "conditions": summaries}, indent=2))


if __name__ == "__main__":
    main()

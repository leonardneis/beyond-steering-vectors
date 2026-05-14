from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.evaluation import evaluate_preference
from slgeo.experiment_logging import ExperimentLogger, tee_output
from slgeo.filtering import filter_number_jsonl
from slgeo.generation import condition_system_prompt_mode, generate_number_dataset
from slgeo.io import load_yaml, write_jsonl
from slgeo.prompts import semantic_animal_training_examples
from slgeo.training import train_lora
from visualize_run import build_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the minimal subliminal-learning pipeline.")
    parser.add_argument("--config", default="configs/experiment_minimal.yaml")
    parser.add_argument(
        "--condition",
        choices=["subliminal_numbers", "neutral_numbers", "semantic_animals"],
        default=None,
    )
    parser.add_argument("--trait", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-report", action="store_true", help="Skip the automatic HTML run report.")
    parser.add_argument("--report-palette", default="vaporwave", help="vapeplot palette for the HTML report.")
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
    prompt_style = data.get("prompt_style", "arithmetic")
    if condition == "semantic_animals":
        prompt_style = "semantic_animals"
    teacher_system_prompt_mode = condition_system_prompt_mode(condition)
    effective_data_config = {
        **data_config,
        "data": {
            **data,
            "condition": condition,
            "trait": trait,
            "prompt_style": prompt_style,
        },
    }
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
    config_paths = {
        "experiment_config": args.config,
        "model_config": experiment_config.get("model_config", "configs/model_qwen.yaml"),
        "data_config": experiment_config.get("data_config", "configs/data_numbers.yaml"),
        "train_config": experiment_config.get("train_config", "configs/train_lora.yaml"),
        "eval_config": experiment_config.get("eval_config", "configs/eval_animals.yaml"),
    }
    cli_overrides = {
        "condition": args.condition,
        "trait": args.trait,
        "run_id": args.run_id,
        "runs_dir": args.runs_dir,
        "dry_run": args.dry_run,
        "no_report": args.no_report,
        "report_palette": args.report_palette,
    }
    run_logger = ExperimentLogger(run_id=args.run_id, runs_dir=args.runs_dir, repo_root=repo_path("."))
    with run_logger.timed_stage("setup_metadata"):
        run_logger.write_metadata(
            experiment_name=experiment_config.get("name", "minimal_reproduction"),
            condition=condition,
            seed=int(data.get("seed", 42)),
            model_name=model_config.get("model", {}).get("model_name"),
            adapter_path=adapter_dir,
            config_paths=config_paths,
            extra={
                "trait": trait,
                "system_prompt_mode": teacher_system_prompt_mode,
                "prompt_style": prompt_style,
            },
        )
        run_logger.write_config_snapshot(
            config_paths=config_paths,
            cli_overrides=cli_overrides,
            effective_config={
                "experiment": experiment_config,
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
                    "trait": trait,
                    "system_prompt_mode": teacher_system_prompt_mode,
                    "prompt_style": prompt_style,
                },
            },
        )

    seed = int(data.get("seed", 42))
    with run_logger.timed_stage("generation_and_filtering"):
        with tee_output(run_logger.path("generation.log")):
            if condition == "semantic_animals":
                examples = semantic_animal_training_examples(
                    animal=trait,
                    count=int(data.get("num_samples", data.get("num_prompts", 20))),
                    seed=int(data.get("generation_seed", seed)),
                )
                write_jsonl(generated_path, examples)
                write_jsonl(filtered_path, examples)
                generation_summary = {
                    "output_path": str(generated_path),
                    "condition": condition,
                    "trait": trait,
                    "records": len(examples),
                    "prompt_style": "semantic_animals",
                    "system_prompt_mode": teacher_system_prompt_mode,
                    "dry_run": dry_run,
                    "source": "template",
                }
                filter_summary = {
                    "input_path": str(generated_path),
                    "output_path": str(filtered_path),
                    "total": len(examples),
                    "valid": len(examples),
                    "invalid": 0,
                    "invalid_reasons": {},
                    "note": "Semantic examples do not use number filtering.",
                }
            else:
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
                    prompt_style=prompt_style,
                    generation_config=data_generation,
                    use_default_system_prompt=bool(data.get("use_default_system_prompt", True)),
                    dry_run=dry_run,
                )
                filter_summary = filter_number_jsonl(
                    input_path=generated_path,
                    output_path=filtered_path,
                    min_numbers=int(filtering.get("min_numbers", 1)),
                )
    with run_logger.timed_stage("dataset_artifacts"):
        run_logger.write_dataset_artifacts(
            generated_path=generated_path,
            filtered_path=filtered_path,
            generation_summary=generation_summary,
            filter_summary=filter_summary,
            prompt_type=prompt_style,
            temperature=float(data_generation.get("temperature", 0.7)),
            top_p=float(data_generation.get("top_p", 0.95)),
        )
    with run_logger.timed_stage("teacher_eval"):
        with tee_output(run_logger.path("teacher_eval.log")):
            teacher_result = evaluate_preference(
                model_config=model_config,
                adapter_path=None,
                target_animal=trait,
                animals=evaluation.get("animals"),
                output_json=None,
                output_csv=None,
                num_samples=evaluation.get("num_samples"),
                num_repeats=int(evaluation.get("num_repeats", 1)),
                max_new_tokens=int(evaluation.get("max_new_tokens", 32)),
                temperature=float(evaluation.get("temperature", 0.7)),
                top_p=float(evaluation.get("top_p", 0.95)),
                top_k=evaluation.get("top_k"),
                do_sample=bool(evaluation.get("do_sample", True)),
                generation_mode=evaluation.get("generation_mode", "sample"),
                prompt_set=evaluation.get("prompt_set", "favorite"),
                system_prompt_mode=teacher_system_prompt_mode,
                add_number_prefix=bool(evaluation.get("add_number_prefix", False)),
                logprob_eval=bool(evaluation.get("logprob_eval", False)),
                token_metric_eval=bool(evaluation.get("token_metric_eval", True)),
                candidate_animals=evaluation.get("candidate_animals"),
                compare_base_logits=bool(evaluation.get("compare_base_logits", True)),
                use_default_system_prompt=bool(evaluation.get("use_default_system_prompt", True)),
                dry_run=dry_run or bool(evaluation.get("dry_run", False)),
            )
        run_logger.write_teacher_artifacts(teacher_result)
    with run_logger.timed_stage("training"):
        with tee_output(run_logger.path("train.log")):
            training_summary = train_lora(
                model_config=model_config,
                training_config=training,
                lora_config=lora,
                train_file=filtered_path,
                output_dir=adapter_dir,
                eval_config=evaluation,
                target_animal=trait,
                dry_run=dry_run or bool(training.get("dry_run", False)),
            )
        run_logger.write_training_metrics(training_summary)
    with run_logger.timed_stage("student_eval"):
        with tee_output(run_logger.path("eval.log")):
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
                top_k=evaluation.get("top_k"),
                do_sample=bool(evaluation.get("do_sample", True)),
                generation_mode=evaluation.get("generation_mode", "sample"),
                prompt_set=evaluation.get("prompt_set", "favorite"),
                system_prompt_mode=evaluation.get("system_prompt_mode", "neutral"),
                add_number_prefix=bool(evaluation.get("add_number_prefix", False)),
                logprob_eval=bool(evaluation.get("logprob_eval", False)),
                token_metric_eval=bool(evaluation.get("token_metric_eval", True)),
                candidate_animals=evaluation.get("candidate_animals"),
                compare_base_logits=bool(evaluation.get("compare_base_logits", True)),
                use_default_system_prompt=bool(evaluation.get("use_default_system_prompt", True)),
                dry_run=dry_run or bool(evaluation.get("dry_run", False)),
            )
        run_logger.write_eval_artifacts(eval_result)
    with run_logger.timed_stage("summary"):
        run_logger.write_summary(
            experiment_name=experiment_config.get("name", "minimal_reproduction"),
            condition=condition,
            trait=trait,
            prompt_style=prompt_style,
            eval_result=eval_result,
            teacher_result=teacher_result,
        )
    report_path = None
    if not args.no_report:
        with run_logger.timed_stage("report"):
            try:
                report_path = build_report(
                    run_logger.run_dir,
                    run_logger.run_dir / "report.html",
                    args.report_palette,
                    sample_limit=None,
                )
                print(f"Interactive report written to {report_path}")
            except Exception as exc:
                print(f"WARNING: failed to create interactive report: {exc!r}")
        if report_path is not None:
            try:
                build_report(
                    run_logger.run_dir,
                    run_logger.run_dir / "report.html",
                    args.report_palette,
                    sample_limit=None,
                )
            except Exception as exc:
                print(f"WARNING: failed to refresh interactive report timing: {exc!r}")

    summary = {
        "experiment": experiment_config.get("name", "minimal_reproduction"),
        "condition": condition,
        "trait": trait,
        "dry_run": dry_run,
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
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

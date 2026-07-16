from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.experiment_logging import ExperimentLogger, tee_output
from slgeo.io import load_yaml
from slgeo.training import train_lora


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a student model with LoRA.")
    parser.add_argument("--config", default="configs/train_lora.yaml")
    parser.add_argument("--model-config", default="configs/model_qwen.yaml")
    parser.add_argument("--eval-config", default=None)
    parser.add_argument("--target-animal", default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from the latest Trainer checkpoint.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_config = load_yaml(repo_path(args.config))
    model_config = load_yaml(repo_path(args.model_config))
    eval_config = load_yaml(repo_path(args.eval_config)).get("evaluation", {}) if args.eval_config else None
    training = train_config.get("training", {})
    if args.seed is not None:
        training = {**training, "seed": args.seed, "data_seed": args.seed}
    if args.batch_size is not None:
        training = {**training, "per_device_train_batch_size": args.batch_size}
    if args.gradient_accumulation_steps is not None:
        training = {
            **training,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        }
    lora = train_config.get("lora", {})
    train_file = repo_path(args.train_file or training.get("train_file"))
    output_dir = repo_path(args.output_dir or training.get("output_dir"))
    run_logger = ExperimentLogger(run_id=args.run_id, runs_dir=args.runs_dir, repo_root=repo_path("."))
    config_paths = {"train_config": args.config, "model_config": args.model_config, "eval_config": args.eval_config}
    run_logger.write_metadata(
        experiment_name="train_student",
        condition=None,
        seed=training.get("seed"),
        model_name=model_config.get("model", {}).get("model_name"),
        adapter_path=output_dir,
        config_paths=config_paths,
        extra={"trait": None, "system_prompt_mode": None, "prompt_style": None},
    )
    run_logger.write_config_snapshot(
        config_paths=config_paths,
        cli_overrides=vars(args),
        effective_config={
            "model": model_config,
            "training": training,
                "evaluation": eval_config,
                "resolved": {
                "train_file": str(train_file),
                "output_dir": str(output_dir),
                "run_dir": str(run_logger.run_dir),
            },
        },
    )

    with tee_output(run_logger.path("train.log")):
        summary = train_lora(
            model_config=model_config,
            training_config=training,
            lora_config=lora,
            train_file=train_file,
            output_dir=output_dir,
            eval_config=eval_config,
            target_animal=args.target_animal or (eval_config or {}).get("target_animal"),
            dry_run=args.dry_run or bool(training.get("dry_run", False)),
            limit=args.limit,
            resume_from_checkpoint=True if args.resume else None,
        )
    run_logger.write_training_metrics(summary)
    summary["run_id"] = run_logger.run_id
    summary["run_dir"] = str(run_logger.run_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

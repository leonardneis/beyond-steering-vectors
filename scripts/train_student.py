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
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_config = load_yaml(repo_path(args.config))
    model_config = load_yaml(repo_path(args.model_config))
    training = train_config.get("training", {})
    lora = train_config.get("lora", {})
    train_file = repo_path(args.train_file or training.get("train_file"))
    output_dir = repo_path(args.output_dir or training.get("output_dir"))
    run_logger = ExperimentLogger(run_id=args.run_id, runs_dir=args.runs_dir, repo_root=repo_path("."))
    config_paths = {"train_config": args.config, "model_config": args.model_config}
    run_logger.write_metadata(
        experiment_name="train_student",
        condition=None,
        seed=None,
        model_name=model_config.get("model", {}).get("model_name"),
        adapter_path=output_dir,
        config_paths=config_paths,
    )
    run_logger.write_config_snapshot(
        config_paths=config_paths,
        cli_overrides=vars(args),
        effective_config={
            "model": model_config,
            "training": train_config,
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
            dry_run=args.dry_run or bool(training.get("dry_run", False)),
            limit=args.limit,
        )
    run_logger.write_training_metrics(summary)
    summary["run_id"] = run_logger.run_id
    summary["run_dir"] = str(run_logger.run_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

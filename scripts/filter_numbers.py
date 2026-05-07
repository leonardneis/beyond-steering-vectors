from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.experiment_logging import ExperimentLogger, tee_output
from slgeo.filtering import filter_number_jsonl, run_self_tests
from slgeo.io import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter generated completions to valid number sequences.")
    parser.add_argument("--config", default="configs/data_numbers.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--min-numbers", type=int, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.self_test:
        run_self_tests()
        print("number filter self-test passed")
        return

    config = load_yaml(repo_path(args.config))
    filter_config = config.get("filter", {})
    input_path = repo_path(args.input or filter_config.get("input_path"))
    output_path = repo_path(args.output or filter_config.get("output_path"))
    min_numbers = args.min_numbers if args.min_numbers is not None else int(filter_config.get("min_numbers", 1))

    if input_path is None or output_path is None:
        raise SystemExit("Missing --input/--output or filter.input_path/filter.output_path in config.")

    run_logger = ExperimentLogger(run_id=args.run_id, runs_dir=args.runs_dir, repo_root=repo_path("."))
    config_paths = {"data_config": args.config}
    run_logger.write_metadata(
        experiment_name="filter_numbers",
        condition=None,
        seed=None,
        model_name=None,
        adapter_path=None,
        config_paths=config_paths,
        extra={"trait": None, "system_prompt_mode": None, "prompt_style": None},
    )
    run_logger.write_config_snapshot(
        config_paths=config_paths,
        cli_overrides=vars(args),
        effective_config={
            "filter": filter_config,
            "resolved": {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "min_numbers": min_numbers,
                "run_dir": str(run_logger.run_dir),
            },
        },
    )
    with tee_output(run_logger.path("generation.log")):
        summary = filter_number_jsonl(
            input_path=input_path,
            output_path=output_path,
            min_numbers=min_numbers,
        )
    run_logger.write_dataset_artifacts(
        generated_path=input_path,
        filtered_path=output_path,
        filter_summary=summary,
    )
    summary["run_id"] = run_logger.run_id
    summary["run_dir"] = str(run_logger.run_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

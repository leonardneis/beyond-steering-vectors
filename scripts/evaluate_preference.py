from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.evaluation import evaluate_preference
from slgeo.io import load_yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate animal preference in a student model.")
    parser.add_argument("--config", default="configs/eval_animals.yaml")
    parser.add_argument("--model-config", default="configs/model_qwen.yaml")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--target-animal", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    eval_config = load_yaml(repo_path(args.config)).get("evaluation", {})
    model_config = load_yaml(repo_path(args.model_config))

    result = evaluate_preference(
        model_config=model_config,
        adapter_path=repo_path(args.adapter_path or eval_config.get("adapter_path")),
        target_animal=args.target_animal or eval_config.get("target_animal", "owl"),
        animals=eval_config.get("animals"),
        output_json=repo_path(args.output_json or eval_config.get("output_json")),
        output_csv=repo_path(args.output_csv or eval_config.get("output_csv")),
        num_repeats=int(eval_config.get("num_repeats", 1)),
        max_new_tokens=int(eval_config.get("max_new_tokens", 32)),
        temperature=float(eval_config.get("temperature", 0.7)),
        top_p=float(eval_config.get("top_p", 0.95)),
        do_sample=bool(eval_config.get("do_sample", True)),
        dry_run=args.dry_run or bool(eval_config.get("dry_run", False)),
    )
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()


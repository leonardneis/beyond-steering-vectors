from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.io import ensure_parent, write_json


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_at(data: dict, *keys, default=None):
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def run_label(run_dir: Path) -> str:
    metadata = load_json(run_dir / "metadata.json")
    condition = metadata.get("condition") or "unknown"
    return f"{run_dir.name} ({condition})"


def summarize_run(run_dir: Path) -> dict:
    metadata = load_json(run_dir / "metadata.json")
    eval_metrics = load_json(run_dir / "eval_metrics.json")
    train_metrics = load_json(run_dir / "training_metrics.json")
    choice_rates = metric_at(eval_metrics, "choice_metrics", "choice_rates", default={}) or {}
    animal_rates = metric_at(eval_metrics, "metrics", "animal_completion_rates", default={}) or {}
    return {
        "run_dir": str(run_dir),
        "run_id": metadata.get("run_id") or run_dir.name,
        "experiment_name": metadata.get("experiment_name"),
        "condition": metadata.get("condition"),
        "trait": metadata.get("trait") or metadata.get("target_animal"),
        "system_prompt_mode": metadata.get("system_prompt_mode"),
        "prompt_style": metadata.get("prompt_style"),
        "target_choice_rate": metric_at(eval_metrics, "choice_metrics", "target_choice_rate"),
        "target_win_rate": metric_at(eval_metrics, "logprob_metrics", "target_win_rate"),
        "no_choice_rate": metric_at(eval_metrics, "choice_metrics", "no_choice_rate"),
        "animal_choice_rates": choice_rates,
        "animal_completion_rates": animal_rates,
        "train_loss": train_metrics.get("train_loss"),
        "optimizer_steps": train_metrics.get("optimizer_steps"),
    }


def numeric_delta(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return b - a
    return None


def comparison(base: dict, candidate: dict) -> dict:
    keys = [
        "target_choice_rate",
        "target_win_rate",
        "no_choice_rate",
        "train_loss",
        "optimizer_steps",
    ]
    rows = []
    for key in keys:
        rows.append(
            {
                "metric": key,
                "base": base.get(key),
                "candidate": candidate.get(key),
                "delta_candidate_minus_base": numeric_delta(base.get(key), candidate.get(key)),
            }
        )
    animals = sorted(
        set(base.get("animal_choice_rates", {}))
        | set(candidate.get("animal_choice_rates", {}))
        | set(base.get("animal_completion_rates", {}))
        | set(candidate.get("animal_completion_rates", {}))
    )
    animal_rows = []
    for animal in animals:
        animal_rows.append(
            {
                "animal": animal,
                "base_choice_rate": base.get("animal_choice_rates", {}).get(animal),
                "candidate_choice_rate": candidate.get("animal_choice_rates", {}).get(animal),
                "choice_delta": numeric_delta(
                    base.get("animal_choice_rates", {}).get(animal),
                    candidate.get("animal_choice_rates", {}).get(animal),
                ),
                "base_completion_rate": base.get("animal_completion_rates", {}).get(animal),
                "candidate_completion_rate": candidate.get("animal_completion_rates", {}).get(animal),
                "completion_delta": numeric_delta(
                    base.get("animal_completion_rates", {}).get(animal),
                    candidate.get("animal_completion_rates", {}).get(animal),
                ),
            }
        )
    return {
        "base": base,
        "candidate": candidate,
        "metric_rows": rows,
        "animal_distribution_rows": animal_rows,
        "target_exceeded_baseline": (
            candidate.get("target_choice_rate") > base.get("target_choice_rate")
            if isinstance(candidate.get("target_choice_rate"), (int, float))
            and isinstance(base.get("target_choice_rate"), (int, float))
            else None
        ),
    }


def print_table(title: str, rows: list[dict], columns: list[str]) -> None:
    print(f"\n{title}")
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        print("| " + " | ".join(str(row.get(column)) for column in columns) + " |")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two logged experiment runs.")
    parser.add_argument("base_run_dir", help="Baseline run directory, e.g. runs/run2_neutral_1k")
    parser.add_argument("candidate_run_dir", help="Candidate run directory, e.g. runs/run2_subliminal_1k")
    parser.add_argument("--name", default=None, help="Comparison artifact name")
    parser.add_argument("--output-dir", default="runs/comparisons")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_dir = repo_path(args.base_run_dir)
    candidate_dir = repo_path(args.candidate_run_dir)
    name = args.name or f"{base_dir.name}_vs_{candidate_dir.name}"
    output_path = ensure_parent(repo_path(args.output_dir) / f"{name}.json")

    result = comparison(summarize_run(base_dir), summarize_run(candidate_dir))
    write_json(output_path, result)

    print(f"Comparison saved to {output_path}")
    print(f"Base: {run_label(base_dir)}")
    print(f"Candidate: {run_label(candidate_dir)}")
    print_table(
        "Core metrics",
        result["metric_rows"],
        ["metric", "base", "candidate", "delta_candidate_minus_base"],
    )
    print_table(
        "Animal distributions",
        result["animal_distribution_rows"],
        [
            "animal",
            "base_choice_rate",
            "candidate_choice_rate",
            "choice_delta",
            "base_completion_rate",
            "candidate_completion_rate",
            "completion_delta",
        ],
    )


if __name__ == "__main__":
    main()

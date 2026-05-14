from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.io import ensure_parent, load_yaml


ANIMAL_COLUMNS = ("lion", "owl", "cat")
OUTPUT_BASENAME = "comparison_table"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_at(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def parse_summary_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    labels = {
        "target_animal_rate": "Target animal rate",
        "target_choice_rate": "Target choice rate",
        "no_choice_rate": "No-choice rate",
        "target_logprob_win_rate": "Target logprob win rate",
        "average_target_margin": "Average target margin",
        "teacher_target_rate": "Teacher target choice rate",
    }
    parsed: dict[str, Any] = {}
    for key, label in labels.items():
        match = re.search(rf"- {re.escape(label)}: `([^`]*)`", text)
        if not match:
            continue
        value = match.group(1)
        if value == "None":
            parsed[key] = None
            continue
        try:
            parsed[key] = float(value)
        except ValueError:
            parsed[key] = value
    return parsed


def load_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config_resolved.yaml"
    if not config_path.exists():
        return {}
    return load_yaml(config_path)


def configured_eval_metrics(run_dir: Path, metadata: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Find an eval JSON path from config when a run-local eval_metrics.json is absent."""
    output_json = metric_at(config, "effective_config", "resolved_paths", "output_json")
    if not output_json:
        output_json = metric_at(config, "effective_config", "evaluation", "output_json")
    if not output_json:
        return {}

    path = Path(output_json)
    if not path.is_absolute():
        path = repo_path(path)
    metrics = load_json(path)
    if not metrics:
        return {}

    expected_adapter = metadata.get("adapter_path")
    actual_adapter = metrics.get("adapter_path")
    if expected_adapter is not None and actual_adapter != expected_adapter:
        return {}
    if expected_adapter is None and actual_adapter is not None:
        return {}
    return metrics


def quantization_mode(*sources: dict[str, Any]) -> str | None:
    for source in sources:
        mode = metric_at(source, "model_diagnostics", "quantization_mode")
        if mode:
            return str(mode)

    for source in sources:
        model = metric_at(source, "effective_config", "model", default={}) or {}
        model_section = model.get("model", {}) if isinstance(model, dict) else {}
        quant_section = model.get("quantization", {}) if isinstance(model, dict) else {}
        if model_section.get("load_in_4bit") or quant_section.get("load_in_4bit"):
            return "4bit"
    return None


def model_name(metadata: dict[str, Any], config: dict[str, Any]) -> str | None:
    return (
        metadata.get("model_name")
        or metric_at(config, "effective_config", "model", "model", "model_name")
        or metric_at(config, "effective_config", "model", "model", "base_model_name")
    )


def summarize_run(run_dir: Path) -> dict[str, Any]:
    metadata = load_json(run_dir / "metadata.json")
    config = load_config(run_dir)
    eval_metrics = load_json(run_dir / "eval_metrics.json")
    if not eval_metrics:
        eval_metrics = configured_eval_metrics(run_dir, metadata, config)
    teacher_metrics = load_json(run_dir / "teacher_metrics.json")
    train_metrics = load_json(run_dir / "training_metrics.json")
    dataset_stats = load_json(run_dir / "dataset_stats.json")
    summary_metrics = parse_summary_metrics(run_dir / "summary.md")

    choice_rates = metric_at(eval_metrics, "choice_metrics", "choice_rates", default={}) or {}
    teacher_choice = metric_at(teacher_metrics, "choice_metrics", "target_choice_rate")
    teacher_target = teacher_choice
    if teacher_target is None:
        teacher_target = metric_at(teacher_metrics, "metrics", "target_animal_rate")

    row: dict[str, Any] = {
        "run_id": metadata.get("run_id") or run_dir.name,
        "run_dir": str(run_dir),
        "condition": metadata.get("condition"),
        "seed": metadata.get("random_seed"),
        "model": model_name(metadata, config),
        "quantization_mode": quantization_mode(eval_metrics, train_metrics, teacher_metrics, config),
        "generation_mode": eval_metrics.get("generation_mode")
        or metric_at(config, "effective_config", "data", "generation", "generation_mode")
        or metric_at(config, "effective_config", "data_source_config", "generation", "generation_mode"),
        "loss_mode": train_metrics.get("loss_mode") or metric_at(config, "effective_config", "training", "training", "loss_mode"),
        "lora_layers": train_metrics.get("selected_lora_layers") or metric_at(config, "effective_config", "training", "lora", "lora_layers"),
        "target_choice_rate": metric_at(eval_metrics, "choice_metrics", "target_choice_rate"),
        "target_animal_rate": metric_at(eval_metrics, "metrics", "target_animal_rate"),
        "no_choice_rate": metric_at(eval_metrics, "choice_metrics", "no_choice_rate"),
        "target_logprob_win_rate": metric_at(eval_metrics, "logprob_metrics", "target_win_rate"),
        "average_target_margin": metric_at(eval_metrics, "logprob_metrics", "average_target_margin"),
        "target_logprob": metric_at(eval_metrics, "token_metrics", "target_logprob")
        or metric_at(eval_metrics, "metrics", "target_logprob"),
        "target_rank": metric_at(eval_metrics, "token_metrics", "target_rank")
        or metric_at(eval_metrics, "metrics", "target_rank"),
        "target_vs_lion_margin": metric_at(eval_metrics, "token_metrics", "target_vs_lion_margin")
        or metric_at(eval_metrics, "metrics", "target_vs_lion_margin"),
        "kl_student_base": metric_at(eval_metrics, "token_metrics", "kl_student_base")
        or metric_at(eval_metrics, "metrics", "kl_student_base"),
        "entropy": metric_at(eval_metrics, "token_metrics", "entropy")
        or metric_at(eval_metrics, "metrics", "entropy"),
        "teacher_target_rate": teacher_target,
        "train_records": train_metrics.get("num_records") or dataset_stats.get("filtered_sample_count"),
        "optimizer_steps": train_metrics.get("optimizer_steps"),
    }
    epoch_rows = train_metrics.get("epoch_eval_metrics") or load_json(run_dir / "epoch_eval_metrics.json")
    if isinstance(epoch_rows, list) and epoch_rows:
        for metric in ["target_logprob", "target_vs_lion_margin"]:
            candidates = [item for item in epoch_rows if isinstance(item.get(metric), (int, float))]
            if candidates:
                best = max(candidates, key=lambda item: item[metric])
                row[f"best_epoch_by_{metric}"] = best.get("epoch")
                row[f"best_epoch_{metric}"] = best.get(metric)
                if isinstance(row.get(metric), (int, float)):
                    row[f"best_minus_final_{metric}"] = best.get(metric) - row.get(metric)
    for animal in ANIMAL_COLUMNS:
        row[f"{animal}_choice_rate"] = choice_rates.get(animal)

    for key, value in summary_metrics.items():
        if row.get(key) is None:
            row[key] = value
    return row


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def _stats(values: list[float]) -> dict[str, Any]:
    n = len(values)
    mean = statistics.fmean(values) if values else None
    std = statistics.stdev(values) if n > 1 else 0.0 if n == 1 else None
    sem = std / (n ** 0.5) if n and std is not None else None
    ci95 = 1.96 * sem if sem is not None else None
    return {"n": n, "mean": mean, "std": std, "sem": sem, "ci95": ci95, "values": values}


def comparison_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = ["target_choice_rate", "target_logprob", "target_rank", "target_vs_lion_margin", "kl_student_base", "entropy"]
    comparisons = [
        ("base_vs_neutral", "condition", "evaluation_only", "neutral_numbers"),
        ("neutral_vs_subliminal", "condition", "neutral_numbers", "subliminal_numbers"),
        ("sample_vs_greedy", "generation_mode", "sample", "greedy"),
        ("all_tokens_vs_divergence_only", "loss_mode", "all_tokens", "divergence_only"),
        ("all_tokens_vs_non_divergence_only", "loss_mode", "all_tokens", "non_divergence_only"),
        ("all_layers_vs_early_layers", "lora_layers", "all", "[0]"),
    ]
    output: list[dict[str, Any]] = []
    for name, key, left_value, right_value in comparisons:
        left = [row for row in rows if str(row.get(key)) == left_value]
        right = [row for row in rows if str(row.get(key)) == right_value]
        if not left or not right:
            continue
        for metric in metrics:
            left_values = [float(row[metric]) for row in left if isinstance(row.get(metric), (int, float))]
            right_values = [float(row[metric]) for row in right if isinstance(row.get(metric), (int, float))]
            if not left_values or not right_values:
                continue
            left_stats = _stats(left_values)
            right_stats = _stats(right_values)
            output.append(
                {
                    "comparison": name,
                    "metric": metric,
                    "left": left_value,
                    "right": right_value,
                    "left_mean": left_stats["mean"],
                    "right_mean": right_stats["mean"],
                    "delta_right_minus_left": right_stats["mean"] - left_stats["mean"],
                    "left_std": left_stats["std"],
                    "right_std": right_stats["std"],
                    "left_sem": left_stats["sem"],
                    "right_sem": right_stats["sem"],
                    "left_ci95": left_stats["ci95"],
                    "right_ci95": right_stats["ci95"],
                    "left_values": json.dumps(left_stats["values"]),
                    "right_values": json.dumps(right_stats["values"]),
                }
            )
    for row in rows:
        for metric in ["target_logprob", "target_vs_lion_margin"]:
            delta = row.get(f"best_minus_final_{metric}")
            if isinstance(delta, (int, float)):
                output.append(
                    {
                        "comparison": f"final_vs_best_checkpoint_by_{metric}",
                        "metric": metric,
                        "left": "final",
                        "right": "best_epoch",
                        "left_mean": row.get(metric),
                        "right_mean": row.get(f"best_epoch_{metric}"),
                        "delta_right_minus_left": delta,
                        "left_values": json.dumps([row.get(metric)]),
                        "right_values": json.dumps([row.get(f"best_epoch_{metric}")]),
                    }
                )
    return output


def best_run(rows: list[dict[str, Any]], metric: str, *, higher_is_better: bool = True) -> dict[str, Any] | None:
    candidates = [row for row in rows if isinstance(row.get(metric), (int, float))]
    if not candidates:
        return None
    return max(candidates, key=lambda row: row[metric]) if higher_is_better else min(candidates, key=lambda row: row[metric])


def build_summary(rows: list[dict[str, Any]]) -> str:
    lines = ["# Run Comparison", ""]
    lines.append(f"Compared {len(rows)} runs.")
    lines.append("")

    for metric, label, higher in [
        ("target_choice_rate", "Highest target choice rate", True),
        ("target_animal_rate", "Highest target animal rate", True),
        ("target_logprob_win_rate", "Highest target logprob win rate", True),
        ("average_target_margin", "Highest average target margin", True),
        ("no_choice_rate", "Lowest no-choice rate", False),
    ]:
        row = best_run(rows, metric, higher_is_better=higher)
        if row is None:
            continue
        lines.append(f"- {label}: `{row['run_id']}` ({format_value(row.get(metric))})")

    base = rows[0] if rows else None
    if base and len(rows) > 1:
        lines.extend(["", f"## Deltas vs `{base['run_id']}`", ""])
        for row in rows[1:]:
            deltas = []
            for metric in ["target_choice_rate", "target_animal_rate", "no_choice_rate", "average_target_margin"]:
                if isinstance(base.get(metric), (int, float)) and isinstance(row.get(metric), (int, float)):
                    deltas.append(f"{metric}: {format_value(row[metric] - base[metric])}")
            suffix = "; ".join(deltas) if deltas else "no numeric overlap"
            lines.append(f"- `{row['run_id']}`: {suffix}")

    lines.extend(["", "## Notes", ""])
    lines.append("- Empty table cells mean the run artifact did not contain that metric.")
    lines.append("- `target_logprob_win_rate` is read from `logprob_metrics.target_win_rate`.")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare logged experiment run directories.")
    parser.add_argument("run_dirs", nargs="+", help="Run directories to compare.")
    parser.add_argument("--output-dir", default="runs/comparisons", help="Directory for comparison artifacts.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dirs = [repo_path(path) for path in args.run_dirs]
    rows = [summarize_run(run_dir) for run_dir in run_dirs]
    columns = [
        "run_id",
        "condition",
        "seed",
        "model",
        "quantization_mode",
        "generation_mode",
        "loss_mode",
        "lora_layers",
        "train_records",
        "optimizer_steps",
        "teacher_target_rate",
        "target_choice_rate",
        "target_animal_rate",
        "no_choice_rate",
        "target_logprob_win_rate",
        "average_target_margin",
        "target_logprob",
        "target_rank",
        "target_vs_lion_margin",
        "kl_student_base",
        "entropy",
        "best_epoch_by_target_logprob",
        "best_epoch_target_logprob",
        "best_minus_final_target_logprob",
        "best_epoch_by_target_vs_lion_margin",
        "best_epoch_target_vs_lion_margin",
        "best_minus_final_target_vs_lion_margin",
        "lion_choice_rate",
        "owl_choice_rate",
        "cat_choice_rate",
        "run_dir",
    ]

    output_dir = repo_path(args.output_dir)
    md_path = ensure_parent(output_dir / f"{OUTPUT_BASENAME}.md")
    csv_path = ensure_parent(output_dir / f"{OUTPUT_BASENAME}.csv")
    summary_path = ensure_parent(output_dir / "comparison_summary.md")
    comparison_csv_path = ensure_parent(output_dir / "comparison.csv")
    comparison_md_path = ensure_parent(output_dir / "comparison.md")

    md_path.write_text(markdown_table(rows, columns), encoding="utf-8")
    write_csv(csv_path, rows, columns)
    comparisons = comparison_rows(rows)
    comparison_columns = [
        "comparison",
        "metric",
        "left",
        "right",
        "left_mean",
        "right_mean",
        "delta_right_minus_left",
        "left_std",
        "right_std",
        "left_sem",
        "right_sem",
        "left_ci95",
        "right_ci95",
        "left_values",
        "right_values",
    ]
    write_csv(comparison_csv_path, comparisons, comparison_columns)
    comparison_md_path.write_text(markdown_table(comparisons, comparison_columns), encoding="utf-8")
    summary_path.write_text(build_summary(rows), encoding="utf-8")

    print(f"Wrote {md_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {comparison_csv_path}")
    print(f"Wrote {comparison_md_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()

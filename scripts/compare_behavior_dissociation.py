"""Decompose paired behavioral interventions into prompt-level condition effects."""

from __future__ import annotations

import argparse
import json

import numpy as np

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.analysis.split_stability import bootstrap_mean_ci  # noqa: E402
from slgeo.io import ensure_parent  # noqa: E402


METRICS = ("target_choice", "target_logprob", "target_probability", "target_vs_lion_margin")


def _prompt_values(evaluation: dict, metric: str) -> np.ndarray:
    if metric == "target_choice":
        values = evaluation["target_choice_per_prompt"]
    else:
        values = [row[metric] for row in evaluation["token_metrics"]["rows"]]
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"Invalid prompt values for {metric}")
    return array


def _records(payload: dict) -> list[dict]:
    records = payload["full_evaluation"].get("prompt_records")
    if not records:
        raise ValueError("Behavior artifact has no fixed prompt_records")
    if len({row["prompt_id"] for row in records}) != len(records):
        raise ValueError("Prompt identifiers are not unique")
    return records


def _summary(values: np.ndarray, *, samples: int, seed: int) -> dict:
    low, high = bootstrap_mean_ci(values, samples=samples, seed=seed)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "ci95": [low, high],
        "per_prompt": values.tolist(),
    }


def _effect(evaluation: dict, reference: dict, mode: str, metric: str) -> np.ndarray:
    observed = _prompt_values(evaluation, metric)
    baseline = _prompt_values(reference, metric)
    if len(observed) != len(baseline):
        raise ValueError(f"Prompt count differs for {metric}")
    return baseline - observed if mode == "necessity" else observed - baseline


def compare_payloads(sub: dict, neutral: dict, *, bootstrap_samples: int, bootstrap_seed: int) -> dict:
    if sub.get("schema_version") != 2 or neutral.get("schema_version") != 2:
        raise ValueError("Dissociation comparison requires schema-v2 behavior artifacts")
    records = _records(sub)
    if records != _records(neutral):
        raise ValueError("Prompt records differ between conditions")
    if sub.get("selection_plan_sha256") != neutral.get("selection_plan_sha256"):
        raise ValueError("Selection-plan checksums differ between conditions")

    def row_key(row: dict) -> tuple:
        draw_id = row.get("draw_id")
        return (int(row["k"]), str(row["set_name"]), -1 if draw_id is None else int(draw_id), str(row["mode"]))

    sub_rows = {row_key(row): row for row in sub["interventions"]}
    neutral_rows = {row_key(row): row for row in neutral["interventions"]}
    if set(sub_rows) != set(neutral_rows):
        raise ValueError("Behavioral intervention sets differ between conditions")

    full_adapter = {}
    for metric_index, metric in enumerate(METRICS):
        sub_effect = _prompt_values(sub["full_evaluation"], metric) - _prompt_values(sub["base_evaluation"], metric)
        neutral_effect = _prompt_values(neutral["full_evaluation"], metric) - _prompt_values(neutral["base_evaluation"], metric)
        paired_effect = sub_effect - neutral_effect
        full_adapter[metric] = {
            "subliminal": _summary(sub_effect, samples=bootstrap_samples, seed=bootstrap_seed + metric_index * 3),
            "neutral": _summary(neutral_effect, samples=bootstrap_samples, seed=bootstrap_seed + metric_index * 3 + 1),
            "paired": _summary(paired_effect, samples=bootstrap_samples, seed=bootstrap_seed + metric_index * 3 + 2),
        }

    rows = []
    for row_index, key in enumerate(sorted(sub_rows), start=1):
        if sub_rows[key]["modules"] != neutral_rows[key]["modules"]:
            raise ValueError(f"Module definitions differ for {key}")
        mode = key[3]
        sub_reference = sub["full_evaluation"] if mode == "necessity" else sub["base_evaluation"]
        neutral_reference = neutral["full_evaluation"] if mode == "necessity" else neutral["base_evaluation"]
        readouts = {}
        for metric_index, metric in enumerate(METRICS):
            sub_effect = _effect(sub_rows[key], sub_reference, mode, metric)
            neutral_effect = _effect(neutral_rows[key], neutral_reference, mode, metric)
            paired_effect = sub_effect - neutral_effect
            seed = bootstrap_seed + 100 + row_index * 20 + metric_index * 3
            readouts[metric] = {
                "subliminal": _summary(sub_effect, samples=bootstrap_samples, seed=seed),
                "neutral": _summary(neutral_effect, samples=bootstrap_samples, seed=seed + 1),
                "paired": _summary(paired_effect, samples=bootstrap_samples, seed=seed + 2),
            }
        rows.append(
            {
                "k": key[0],
                "set_name": key[1],
                "draw_id": None if key[2] == -1 else key[2],
                "mode": mode,
                "modules": sub_rows[key]["modules"],
                "readouts": readouts,
            }
        )

    return {
        "schema_version": 1,
        "analysis": "activation_behavior_dissociation_pair",
        "prompt_records": records,
        "prompt_file_sha256": sub.get("prompt_file_sha256"),
        "selection_plan_sha256": sub.get("selection_plan_sha256"),
        "bootstrap_samples": bootstrap_samples,
        "primary_readout": "target_logprob",
        "full_adapter": full_adapter,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subliminal", required=True)
    parser.add_argument("--neutral", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    args = parser.parse_args()
    output = repo_path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite paired artifact: {output}")
    result = compare_payloads(
        json.loads(repo_path(args.subliminal).read_text(encoding="utf-8")),
        json.loads(repo_path(args.neutral).read_text(encoding="utf-8")),
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    ensure_parent(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(result['rows'])} decomposed interventions to {output}")


if __name__ == "__main__":
    main()

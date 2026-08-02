"""Aggregate the preregistered final-state directional decomposition."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.analysis.directional_decomposition import (  # noqa: E402
    margin_contributions,
    decision_gate,
    percentile_interval,
    prompt_level_estimands,
    stratified_bootstrap,
    unit_direction,
)
from slgeo.analysis.vector_artifacts import sha256_file  # noqa: E402
from slgeo.io import ensure_parent  # noqa: E402


REQUIRED_ARRAYS = {
    "metadata_json", "prompt_ids", "families", "set_names", "draw_ids",
    "teacher_unit", "margin_direction", "full_state", "ablated_state",
    "full_margin", "ablated_margin", "teacher_patched_margin",
    "residual_patched_margin",
}


def load_artifact(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        missing = REQUIRED_ARRAYS - set(archive.files)
        if missing:
            raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
        result = {key: archive[key] for key in archive.files}
    result["metadata"] = json.loads(str(result.pop("metadata_json").item()))
    return result


def validate_pair(sub: dict, neutral: dict, *, prompts: int, sets: int) -> None:
    for key in ("prompt_ids", "families", "set_names", "draw_ids", "teacher_unit", "margin_direction"):
        if not np.array_equal(sub[key], neutral[key]):
            raise ValueError(f"Condition artifacts disagree on frozen array {key}")
    if sub["full_state"].shape[0] != prompts or sub["ablated_state"].shape[:2] != (sets, prompts):
        raise ValueError("Subliminal state arrays have unexpected dimensions")
    if neutral["full_state"].shape != sub["full_state"].shape or neutral["ablated_state"].shape != sub["ablated_state"].shape:
        raise ValueError("Condition state arrays have different shapes")
    for artifact, condition in ((sub, "subliminal"), (neutral, "neutral")):
        if artifact["metadata"].get("condition") != condition:
            raise ValueError(f"Artifact condition is not {condition}")
        if not np.isfinite(artifact["full_state"]).all() or not np.isfinite(artifact["ablated_state"]).all():
            raise ValueError(f"{condition} state artifact contains non-finite values")


def summarize(values: np.ndarray, bootstrap_values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "ci90": percentile_interval(bootstrap_values, 0.90),
        "ci95": percentile_interval(bootstrap_values, 0.95),
    }


def condition_components(artifact: dict) -> dict[str, np.ndarray]:
    delta = artifact["full_state"][None, :, :] - artifact["ablated_state"]
    return margin_contributions(delta, artifact["teacher_unit"], artifact["margin_direction"])


def component_rank(values: dict[str, np.ndarray], names: np.ndarray) -> dict[str, dict]:
    top = int(np.flatnonzero(names == "top_k")[0])
    controls = np.flatnonzero(names == "norm_matched_control")
    output = {}
    for key in ("full", "teacher", "residual"):
        set_means = np.asarray(values[key]).mean(axis=1)
        top_value = float(set_means[top])
        control_values = set_means[controls]
        output[key] = {
            "top_mean": top_value,
            "control_mean": float(control_values.mean()),
            "top_minus_control": top_value - float(control_values.mean()),
            "top_wins": int(np.sum(top_value > control_values)),
            "control_draws": int(len(controls)),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subliminal", required=True)
    parser.add_argument("--neutral", required=True)
    parser.add_argument("--parent-aggregate", required=True)
    parser.add_argument("--expected-prompts", type=int, default=72)
    parser.add_argument("--expected-controls", type=int, default=25)
    parser.add_argument("--parent-margin", type=float, required=True)
    parser.add_argument("--equivalence-margin", type=float, required=True)
    parser.add_argument("--parent-reconstruction-atol", type=float, required=True)
    parser.add_argument("--decomposition-atol", type=float, default=1e-4)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    output, output_csv = repo_path(args.output), repo_path(args.output_csv)
    if output.exists() or output_csv.exists():
        raise FileExistsError("Refusing to overwrite decomposition outputs")
    sub_path, neutral_path, parent_path = map(repo_path, (args.subliminal, args.neutral, args.parent_aggregate))
    sub, neutral = load_artifact(sub_path), load_artifact(neutral_path)
    expected_sets = args.expected_controls + 1
    validate_pair(sub, neutral, prompts=args.expected_prompts, sets=expected_sets)

    teacher = unit_direction(sub["teacher_unit"])
    if abs(float(np.linalg.norm(teacher)) - 1.0) > 1e-8:
        raise RuntimeError("Teacher normalization failed")
    sub_components, neutral_components = condition_components(sub), condition_components(neutral)
    prompt_values = prompt_level_estimands(sub_components, neutral_components, sub["set_names"])
    boot = stratified_bootstrap(
        prompt_values, sub["families"], samples=args.bootstrap_samples, seed=args.bootstrap_seed
    )
    summaries = {key: summarize(prompt_values[key], boot[key]) for key in prompt_values}

    component_error = float(np.max(np.abs(
        sub_components["full"] - sub_components["teacher"] - sub_components["residual"]
    )))
    component_error = max(component_error, float(np.max(np.abs(
        neutral_components["full"] - neutral_components["teacher"] - neutral_components["residual"]
    ))))
    estimand_error = abs(summaries["full"]["mean"] - summaries["teacher"]["mean"] - summaries["residual"]["mean"])
    patch_errors = {}
    for condition, artifact, components in (("subliminal", sub, sub_components), ("neutral", neutral, neutral_components)):
        direct_full = artifact["full_margin"][None, :] - artifact["ablated_margin"]
        patch_errors[condition] = {
            "state_vs_direct_full": float(np.max(np.abs(components["full"] - direct_full))),
            "teacher_patch_vs_projection": float(np.max(np.abs(
                components["teacher"] - (artifact["teacher_patched_margin"] - artifact["ablated_margin"])
            ))),
            "residual_patch_vs_projection": float(np.max(np.abs(
                components["residual"] - (artifact["residual_patched_margin"] - artifact["ablated_margin"])
            ))),
        }
    numerical_max = max(component_error, estimand_error, *(value for row in patch_errors.values() for value in row.values()))
    if numerical_max > args.decomposition_atol:
        raise RuntimeError(f"Directional decomposition identity failed: {numerical_max}")

    full = summaries["full"]
    teacher_summary = summaries["teacher"]
    family_summaries = {}
    for family in sorted(set(sub["families"].tolist())):
        mask = sub["families"] == family
        family_summaries[family] = {key: float(value[mask].mean()) for key, value in prompt_values.items()}
    gate = decision_gate(
        summaries, parent_margin=args.parent_margin,
        equivalence_margin=args.equivalence_margin,
        parent_reconstruction_atol=args.parent_reconstruction_atol,
    )

    ratio = None if abs(full["mean"]) < args.equivalence_margin else teacher_summary["mean"] / full["mean"]
    paired_components = {key: sub_components[key] - neutral_components[key] for key in ("full", "teacher", "residual")}
    result = {
        "schema_version": 1,
        "analysis": "final_state_directional_causal_decomposition",
        "inputs": {
            "subliminal": str(sub_path), "subliminal_sha256": sha256_file(sub_path),
            "neutral": str(neutral_path), "neutral_sha256": sha256_file(neutral_path),
            "parent_aggregate": str(parent_path), "parent_aggregate_sha256": sha256_file(parent_path),
        },
        "contract": {
            "parent_margin": args.parent_margin,
            "equivalence_margin": args.equivalence_margin,
            "parent_reconstruction_atol": args.parent_reconstruction_atol,
            "decomposition_atol": args.decomposition_atol,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": args.bootstrap_seed,
            "prompt_count": args.expected_prompts,
            "control_draws": args.expected_controls,
        },
        "estimands": summaries,
        "teacher_share": ratio,
        "family_estimands": family_summaries,
        "condition_set_ranks": {
            "subliminal": component_rank(sub_components, sub["set_names"]),
            "neutral": component_rank(neutral_components, neutral["set_names"]),
            "paired": component_rank(paired_components, sub["set_names"]),
        },
        "readout_compatibility": float(sub["margin_direction"] @ teacher),
        "numerical_audit": {
            "component_max_abs_error": component_error,
            "estimand_abs_error": estimand_error,
            "patch_errors": patch_errors,
            "max_abs_error": numerical_max,
        },
        "decision_gate": gate,
    }
    ensure_parent(output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    ensure_parent(output_csv)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("prompt_id", "family", "full", "teacher", "residual"))
        writer.writeheader()
        for index, prompt_id in enumerate(sub["prompt_ids"].tolist()):
            writer.writerow({
                "prompt_id": prompt_id, "family": sub["families"][index],
                **{key: float(prompt_values[key][index]) for key in prompt_values},
            })
    print(f"Wrote directional decomposition to {output} and {output_csv}")


if __name__ == "__main__":
    main()

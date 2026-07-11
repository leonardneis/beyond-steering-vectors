"""Build the paired subliminal-minus-neutral layer ranking from schema-v2 runs."""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.analysis.attribution import prompt_drop_scores, summarize_prompt_values  # noqa: E402
from slgeo.io import ensure_parent  # noqa: E402


def _bootstrap_mean_ci(values: torch.Tensor, samples: int, seed: int) -> dict[str, float]:
    array = values.detach().float().cpu().numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"level": 0.95, "low": float(low), "high": float(high)}


def _validate_pair(subliminal: dict, neutral: dict) -> None:
    required_equal = (
        "schema_version",
        "teacher_vector_sha256",
        "prompts_sha256",
        "n_prompts",
        "prompt_offset",
        "position",
        "target_block",
        "group_by",
    )
    if subliminal.get("schema_version") != 2 or neutral.get("schema_version") != 2:
        raise ValueError("Both inputs must be schema-version 2 attribution files")
    mismatches = [key for key in required_equal if subliminal.get(key) != neutral.get(key)]
    if mismatches:
        raise ValueError(f"Attribution inputs are not paired; mismatched fields: {mismatches}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subliminal", required=True)
    parser.add_argument("--neutral", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()

    sub_path, neutral_path = repo_path(args.subliminal), repo_path(args.neutral)
    subliminal = json.loads(sub_path.read_text(encoding="utf-8"))
    neutral = json.loads(neutral_path.read_text(encoding="utf-8"))
    _validate_pair(subliminal, neutral)
    sub_groups = {row["group"]: row for row in subliminal["ablations"]}
    neutral_groups = {row["group"]: row for row in neutral["ablations"]}
    if set(sub_groups) != set(neutral_groups):
        raise ValueError("Subliminal and neutral runs contain different ablation groups")

    rows = []
    for group in sorted(sub_groups):
        sub_row, neutral_row = sub_groups[group], neutral_groups[group]
        if sub_row["modules"] != neutral_row["modules"] or sub_row["layer"] != neutral_row["layer"]:
            raise ValueError(f"Ablation group {group} is not identically defined")
        layer = int(sub_row["layer"])
        sub_drop = torch.tensor(sub_row["projection_drop_per_prompt"])
        neutral_drop = torch.tensor(neutral_row["projection_drop_per_prompt"])
        if sub_drop.shape != neutral_drop.shape:
            raise ValueError(f"Per-prompt drop shape differs for {group}")
        sub_scores = prompt_drop_scores(
            sub_drop, layer=layer, fixed_target_block=subliminal["target_block"]
        )
        neutral_scores = prompt_drop_scores(
            neutral_drop, layer=layer, fixed_target_block=neutral["target_block"]
        )

        metrics = {}
        paired = {}
        for metric in ("local_drop", "fixed_target_drop", "terminal_drop", "downstream_mean_drop"):
            sub_values, neutral_values = sub_scores[metric], neutral_scores[metric]
            if sub_values is None:
                metrics[metric] = None
                paired[metric] = None
                continue
            contrast = sub_values - neutral_values
            metrics[metric] = {
                "subliminal": summarize_prompt_values(sub_values),
                "neutral": summarize_prompt_values(neutral_values),
                "paired_contrast": summarize_prompt_values(contrast),
                "paired_contrast_bootstrap_ci": _bootstrap_mean_ci(
                    contrast, args.bootstrap_samples, args.bootstrap_seed + layer
                ),
            }
            paired[metric] = contrast.tolist()

        main_values = sub_scores["downstream_mean_drop"] - neutral_scores["downstream_mean_drop"]
        rows.append(
            {
                "group": group,
                "layer": layer,
                "modules": sub_row["modules"],
                "main_layer_score": float(main_values.mean()),
                "metrics": metrics,
                "paired_contrast_per_prompt": paired,
            }
        )

    rows.sort(key=lambda row: (-row["main_layer_score"], row["group"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    result = {
        "schema_version": 1,
        "analysis": "paired_layer_attribution_comparison",
        "main_layer_score_definition": (
            "mean(downstream_mean_drop_subliminal - downstream_mean_drop_neutral)"
        ),
        "subliminal_source": str(sub_path),
        "neutral_source": str(neutral_path),
        "n_prompts": subliminal["n_prompts"],
        "prompt_offset": subliminal["prompt_offset"],
        "teacher_vector_sha256": subliminal["teacher_vector_sha256"],
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "layers": rows,
    }
    output = ensure_parent(repo_path(args.output))
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote paired ranking for {len(rows)} layers to {output}")
    for row in rows[:10]:
        ci = row["metrics"]["downstream_mean_drop"]["paired_contrast_bootstrap_ci"]
        print(
            f"#{row['rank']:02d} L{row['layer']:02d} score={row['main_layer_score']:+.6f} "
            f"95% CI [{ci['low']:+.6f}, {ci['high']:+.6f}]"
        )


if __name__ == "__main__":
    main()

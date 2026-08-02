"""Compare paired subliminal and neutral behavioral set interventions."""

from __future__ import annotations

import argparse
import json

import numpy as np

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.io import ensure_parent  # noqa: E402
from slgeo.analysis.split_stability import bootstrap_mean_ci  # noqa: E402


TOKEN_METRICS = ("target_logprob", "target_probability", "target_vs_lion_margin")


def _prompt_values(evaluation: dict, metric: str) -> np.ndarray:
    if metric == "target_choice":
        return np.asarray(evaluation["target_choice_per_prompt"], dtype=float)
    return np.asarray([row[metric] for row in evaluation["token_metrics"]["rows"]], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subliminal", required=True)
    parser.add_argument("--neutral", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260712)
    args = parser.parse_args()
    output_path = repo_path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing paired behavioral artifact: {output_path}")
    sub = json.loads(repo_path(args.subliminal).read_text(encoding="utf-8"))
    neutral = json.loads(repo_path(args.neutral).read_text(encoding="utf-8"))
    if sub.get("schema_version") != 2 or neutral.get("schema_version") != 2:
        raise ValueError("Corrected behavioral comparison requires schema-v2 input artifacts")
    def row_key(row):
        draw_id = row.get("draw_id")
        return (row["k"], row["set_name"], -1 if draw_id is None else int(draw_id), row["mode"])

    sr = {row_key(r): r for r in sub["interventions"]}
    nr = {row_key(r): r for r in neutral["interventions"]}
    if set(sr) != set(nr):
        raise ValueError("Behavioral intervention sets differ between conditions")
    rows = []
    for index, key in enumerate(sorted(sr), start=1):
        if sr[key]["modules"] != nr[key]["modules"]:
            raise ValueError(f"Module definitions differ for {key}")
        readouts = {}
        for metric_index, metric in enumerate(("target_choice", *TOKEN_METRICS)):
            sub_intervention = _prompt_values(sr[key], metric)
            neutral_intervention = _prompt_values(nr[key], metric)
            sub_reference = _prompt_values(
                sub["full_evaluation"] if key[3] == "necessity" else sub["base_evaluation"], metric
            )
            neutral_reference = _prompt_values(
                neutral["full_evaluation"] if key[3] == "necessity" else neutral["base_evaluation"], metric
            )
            sub_effect = sub_reference - sub_intervention if key[3] == "necessity" else sub_intervention - sub_reference
            neutral_effect = (
                neutral_reference - neutral_intervention
                if key[3] == "necessity"
                else neutral_intervention - neutral_reference
            )
            trait_effect = sub_effect - neutral_effect
            low, high = bootstrap_mean_ci(
                trait_effect,
                samples=args.bootstrap_samples,
                seed=args.bootstrap_seed + index * 10 + metric_index,
            )
            readouts[metric] = {
                "subliminal_effect_mean": float(sub_effect.mean()),
                "neutral_effect_mean": float(neutral_effect.mean()),
                "trait_specific_effect_mean": float(trait_effect.mean()),
                "trait_specific_effect_median": float(np.median(trait_effect)),
                "trait_specific_effect_ci95": [low, high],
                "trait_specific_effect_per_prompt": trait_effect.tolist(),
            }
        rows.append(
            {
                "k": key[0],
                "set_name": key[1],
                "draw_id": None if key[2] == -1 else key[2],
                "mode": key[3],
                "modules": sr[key]["modules"],
                "subliminal_behavioral_effect": sr[key]["behavioral_effect"],
                "neutral_behavioral_effect": nr[key]["behavioral_effect"],
                "trait_specific_behavioral_effect": sr[key]["behavioral_effect"]
                - nr[key]["behavioral_effect"],
                "readouts": readouts,
            }
        )
    result = {
        "schema_version": 2,
        "analysis": "paired_lora_set_behavior",
        "subliminal_source": str(repo_path(args.subliminal)),
        "neutral_source": str(repo_path(args.neutral)),
        "bootstrap_samples": args.bootstrap_samples,
        "primary_readout": "target_logprob",
        "rows": rows,
    }
    ensure_parent(output_path).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} paired behavioral interventions to {output_path}")


if __name__ == "__main__":
    main()

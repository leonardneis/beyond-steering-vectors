"""Compare paired subliminal/neutral top-k necessity and sufficiency runs."""

from __future__ import annotations

import argparse
import json

import numpy as np

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.analysis.split_stability import bootstrap_mean_ci  # noqa: E402
from slgeo.io import ensure_parent  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subliminal", required=True)
    parser.add_argument("--neutral", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260712)
    args = parser.parse_args()
    sub_path, neutral_path = repo_path(args.subliminal), repo_path(args.neutral)
    sub, neutral = json.loads(sub_path.read_text()), json.loads(neutral_path.read_text())
    fields = ("teacher_vector_sha256", "prompts_sha256", "n_prompts", "prompt_offset", "selection_plan")
    mismatches = [field for field in fields if sub.get(field) != neutral.get(field)]
    if mismatches:
        raise ValueError(f"Set-intervention runs are not paired: {mismatches}")
    sub_rows = {(row["k"], row["set_name"], row["mode"]): row for row in sub["interventions"]}
    neutral_rows = {(row["k"], row["set_name"], row["mode"]): row for row in neutral["interventions"]}
    if set(sub_rows) != set(neutral_rows):
        raise ValueError("Intervention sets differ between conditions")
    baseline = np.asarray(sub["baseline_projection_per_prompt"]) - np.asarray(neutral["baseline_projection_per_prompt"])
    baseline_global = baseline[:, 1:].mean(axis=1)
    rows = []
    for index, key in enumerate(sorted(sub_rows), start=1):
        s, n = sub_rows[key], neutral_rows[key]
        if s["modules"] != n["modules"]:
            raise ValueError(f"Module definitions differ for {key}")
        effect = np.asarray(s["effect_projection_per_prompt"]) - np.asarray(n["effect_projection_per_prompt"])
        global_effect, terminal_effect = effect[:, 1:].mean(axis=1), effect[:, -1]
        low, high = bootstrap_mean_ci(
            global_effect, samples=args.bootstrap_samples, seed=args.bootstrap_seed + index
        )
        rows.append(
            {
                "k": key[0],
                "set_name": key[1],
                "mode": key[2],
                "modules": s["modules"],
                "trait_specific_global_effect_mean": float(global_effect.mean()),
                "trait_specific_global_effect_median": float(np.median(global_effect)),
                "trait_specific_global_effect_ci95": [low, high],
                "trait_specific_terminal_effect_mean": float(terminal_effect.mean()),
                "fraction_of_baseline_global_effect": (
                    float(global_effect.mean() / baseline_global.mean())
                    if baseline_global.mean() != 0
                    else None
                ),
            }
        )
    result = {
        "schema_version": 1,
        "analysis": "paired_lora_set_interventions",
        "subliminal_source": str(sub_path),
        "neutral_source": str(neutral_path),
        "baseline_trait_specific_global_mean": float(baseline_global.mean()),
        "bootstrap_samples": args.bootstrap_samples,
        "rows": rows,
    }
    output = ensure_parent(repo_path(args.output))
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} paired set interventions to {output}")


if __name__ == "__main__":
    main()

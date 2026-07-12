"""Aggregate confirmatory mechanism results across independently trained adapter seeds."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from _bootstrap import bootstrap, repo_path

bootstrap()


def seed_ci(values: list[float], samples: int, seed: int) -> list[float] | None:
    if len(values) < 2:
        return None
    rng = np.random.default_rng(seed)
    array = np.asarray(values)
    means = array[rng.integers(0, len(array), size=(samples, len(array)))].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def load_seed(root: Path) -> dict:
    attr = root / "attribution"
    modules = json.loads((attr / "paired_modules.json").read_text())
    topk = json.loads((attr / "paired_topk.json").read_text())
    behavior = json.loads((attr / "paired_behavior.json").read_text())
    sub_behavior = json.loads(Path(behavior["subliminal_source"]).read_text())
    neutral_behavior = json.loads(Path(behavior["neutral_source"]).read_text())
    full_logprob = (
        sub_behavior["full_evaluation"]["token_metrics"]["target_logprob"]
        - sub_behavior["base_evaluation"]["token_metrics"]["target_logprob"]
        - neutral_behavior["full_evaluation"]["token_metrics"]["target_logprob"]
        + neutral_behavior["base_evaluation"]["token_metrics"]["target_logprob"]
    )
    ranking = [row["modules"][0] for row in sorted(modules["layers"], key=lambda row: row["rank"])]
    scores = {row["modules"][0]: row["main_layer_score"] for row in modules["layers"]}
    activation = {(row["k"], row["set_name"], row["mode"]): row for row in topk["rows"]}
    behavioral = {(row["k"], row["set_name"], row["mode"]): row for row in behavior["rows"]}
    rows = []
    for k in (10, 20):
        for mode in ("necessity", "sufficiency"):
            a = activation[k, "top_k", mode]
            b = behavioral[k, "top_k", mode]["readouts"]["target_logprob"]
            a_control = activation[k, "norm_matched_control", mode]
            b_control = behavioral[k, "norm_matched_control", mode]["readouts"]["target_logprob"]
            rows.append({
                "k": k,
                "mode": mode,
                "activation_global": a["trait_specific_global_effect_mean"],
                "activation_fraction": a["fraction_of_baseline_global_effect"],
                "behavior_logprob": b["trait_specific_effect_mean"],
                "behavior_fraction": b["trait_specific_effect_mean"] / full_logprob if full_logprob else None,
                "prompt_ci": b["trait_specific_effect_ci95"],
                "activation_top_minus_norm": a["trait_specific_global_effect_mean"] - a_control["trait_specific_global_effect_mean"],
                "behavior_top_minus_norm": b["trait_specific_effect_mean"] - b_control["trait_specific_effect_mean"],
            })
    alignment = json.loads((root / "vectors" / "subliminal" / "alignment.json").read_text())
    return {"seed": int(root.name.split("_")[-1]), "root": str(root), "ranking": ranking, "scores": scores, "alignment_global": float(np.mean(alignment["cosine_per_hidden_state_slot"][1:])), "alignment_terminal": alignment["cosine_per_hidden_state_slot"][-1], "full_behavior_logprob": full_logprob, "effects": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-roots", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260720)
    args = parser.parse_args()
    seeds = [load_seed(repo_path(path)) for path in args.seed_roots]
    similarities = []
    for left, right in combinations(seeds, 2):
        common = sorted(set(left["scores"]) & set(right["scores"]))
        rho = float(spearmanr([left["scores"][m] for m in common], [right["scores"][m] for m in common]).statistic)
        similarities.append({"seeds": [left["seed"], right["seed"]], "spearman": rho, "top10_overlap": len(set(left["ranking"][:10]) & set(right["ranking"][:10])), "top20_overlap": len(set(left["ranking"][:20]) & set(right["ranking"][:20]))})
    aggregate = []
    for k in (10, 20):
        for mode in ("necessity", "sufficiency"):
            selected = [next(row for row in seed["effects"] if row["k"] == k and row["mode"] == mode) for seed in seeds]
            metrics = {}
            for field in ("activation_global", "activation_fraction", "behavior_logprob", "behavior_fraction", "activation_top_minus_norm", "behavior_top_minus_norm"):
                values = [row[field] for row in selected]
                metrics[field] = {"per_seed": values, "mean": float(np.mean(values)), "training_seed_bootstrap_ci95": seed_ci(values, args.bootstrap_samples, args.bootstrap_seed + k + len(metrics))}
            aggregate.append({"k": k, "mode": mode, "metrics": metrics})
    result = {"schema_version": 1, "uncertainty": {"prompt": "per-seed paired bootstrap CI in source artifacts", "control_set": "per-seed top-k versus norm-control contrasts", "training_seed": "bootstrap over independent seed-level effect estimates"}, "seeds": seeds, "ranking_similarity": similarities, "aggregate": aggregate}
    output = repo_path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite aggregate: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote cross-seed aggregate for {len(seeds)} seeds to {output}")


if __name__ == "__main__":
    main()

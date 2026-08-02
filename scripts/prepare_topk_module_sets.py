"""Prepare reproducible top-k and matched-control module sets from Phase 2."""

from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.analysis.delta_weights import load_adapter_state_dict, reconstruct_lora_updates  # noqa: E402
from slgeo.analysis.topk import (  # noqa: E402
    prepare_module_set_distribution,
    prepare_module_sets,
)
from slgeo.io import ensure_parent  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranking", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument(
        "--matching-pool-size",
        type=int,
        default=1,
        help="Sample norm controls from this many nearest candidates (1 preserves exact matching).",
    )
    parser.add_argument(
        "--control-types",
        nargs="+",
        choices=("random", "norm", "layer_norm"),
        default=("random", "norm", "layer_norm"),
    )
    parser.add_argument(
        "--control-draws",
        type=int,
        default=1,
        help="Number of distinct draws per control type; values >1 emit schema v2.",
    )
    parser.add_argument("--expected-pool-size", type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ranking_path, adapter_dir = repo_path(args.ranking), repo_path(args.adapter_dir)
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    ranked_modules = [row["modules"][0] for row in sorted(ranking["layers"], key=lambda row: row["rank"])]
    if args.expected_pool_size is not None and len(ranked_modules) != args.expected_pool_size:
        raise ValueError(
            f"Selection pool has {len(ranked_modules)} modules, expected {args.expected_pool_size}"
        )
    if len(set(ranked_modules)) != len(ranked_modules):
        raise ValueError("Ranking contains duplicate module names")
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    updates = reconstruct_lora_updates(
        load_adapter_state_dict(adapter_dir),
        alpha=float(adapter_config["lora_alpha"]),
        rank=int(adapter_config["r"]),
    )
    norms = {module: updates[module].frobenius_norm for module in ranked_modules}
    if args.control_draws == 1:
        sets = prepare_module_sets(
            ranked_modules,
            norms,
            k_values=args.k,
            seed=args.seed,
            matching_pool_size=args.matching_pool_size,
            control_types=tuple(args.control_types),
        )
        schema_version = 1
    else:
        sets = prepare_module_set_distribution(
            ranked_modules,
            norms,
            k_values=args.k,
            seed=args.seed,
            control_draws=args.control_draws,
            matching_pool_size=args.matching_pool_size,
            control_types=tuple(args.control_types),
        )
        schema_version = 2
    result = {
        "schema_version": schema_version,
        "ranking": str(ranking_path),
        "adapter_dir": str(adapter_dir),
        "seed": args.seed,
        "matching_pool_size": args.matching_pool_size,
        "control_types": list(args.control_types),
        "control_draws": args.control_draws,
        "selection_pool_size": len(ranked_modules),
        "norms": norms,
        "sets": sets,
    }
    output = ensure_parent(repo_path(args.output))
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(sets)} top-k/control plans to {output}")


if __name__ == "__main__":
    main()

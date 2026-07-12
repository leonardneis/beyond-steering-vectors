"""Prepare reproducible top-k and matched-control module sets from Phase 2."""

from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.analysis.delta_weights import load_adapter_state_dict, reconstruct_lora_updates  # noqa: E402
from slgeo.analysis.topk import prepare_module_sets  # noqa: E402
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
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ranking_path, adapter_dir = repo_path(args.ranking), repo_path(args.adapter_dir)
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    ranked_modules = [row["modules"][0] for row in sorted(ranking["layers"], key=lambda row: row["rank"])]
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    updates = reconstruct_lora_updates(
        load_adapter_state_dict(adapter_dir),
        alpha=float(adapter_config["lora_alpha"]),
        rank=int(adapter_config["r"]),
    )
    norms = {module: updates[module].frobenius_norm for module in ranked_modules}
    sets = prepare_module_sets(
        ranked_modules,
        norms,
        k_values=args.k,
        seed=args.seed,
        matching_pool_size=args.matching_pool_size,
        control_types=tuple(args.control_types),
    )
    result = {
        "schema_version": 1,
        "ranking": str(ranking_path),
        "adapter_dir": str(adapter_dir),
        "seed": args.seed,
        "matching_pool_size": args.matching_pool_size,
        "control_types": list(args.control_types),
        "selection_pool_size": len(ranked_modules),
        "norms": norms,
        "sets": sets,
    }
    output = ensure_parent(repo_path(args.output))
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(sets)} top-k/control plans to {output}")


if __name__ == "__main__":
    main()

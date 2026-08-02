"""Measure full-pool module-ranking sensitivity to teacher-vector resampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.io import ensure_parent  # noqa: E402


def _ranking(path: Path, expected_modules: int) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = sorted(payload["layers"], key=lambda row: int(row["rank"]))
    modules = [str(row["modules"][0]) for row in rows]
    if len(modules) != expected_modules or len(set(modules)) != expected_modules:
        raise ValueError(
            f"{path} has {len(modules)} rows / {len(set(modules))} unique modules; "
            f"expected {expected_modules}"
        )
    return modules


def compare_rankings(reference: list[str], candidate: list[str], k_values: list[int]) -> dict:
    if set(reference) != set(candidate):
        raise ValueError("Rankings do not contain the same module universe")
    candidate_rank = {module: rank for rank, module in enumerate(candidate, start=1)}
    x = np.arange(1, len(reference) + 1, dtype=float)
    y = np.asarray([candidate_rank[module] for module in reference], dtype=float)
    rho = float(np.corrcoef(x, y)[0, 1])
    return {
        "spearman_rho": rho,
        "top_k": {
            str(k): {
                "overlap": len(set(reference[:k]) & set(candidate[:k])),
                "jaccard": len(set(reference[:k]) & set(candidate[:k]))
                / len(set(reference[:k]) | set(candidate[:k])),
            }
            for k in k_values
        },
    }


def aggregate(root: Path, variants: list[str], expected_modules: int, k_values: list[int]) -> dict:
    rows = []
    for seed_dir in sorted(root.glob("seed_*")):
        seed = int(seed_dir.name.removeprefix("seed_"))
        attr = seed_dir / "attribution"
        reference = _ranking(attr / "paired_modules.json", expected_modules)
        for variant in variants:
            candidate = _ranking(
                seed_dir / "robustness" / f"paired_modules_{variant}.json",
                expected_modules,
            )
            rows.append({
                "seed": seed,
                "variant": variant,
                **compare_rankings(reference, candidate, k_values),
            })
    return {
        "schema_version": 1,
        "analysis": "teacher_vector_module_ranking_robustness",
        "root": str(root),
        "expected_modules": expected_modules,
        "variants": variants,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--expected-modules", type=int, default=196)
    parser.add_argument("--k", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = aggregate(
        repo_path(args.root), args.variants, args.expected_modules, sorted(set(args.k))
    )
    output = ensure_parent(repo_path(args.output))
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(result['rows'])} ranking robustness comparisons to {output}")


if __name__ == "__main__":
    main()

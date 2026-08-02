"""Aggregate per-seed top-k effects against repeated module-control draws."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.io import ensure_parent  # noqa: E402


def _summarize(
    *, seed: int, k: int, mode: str, control_type: str, readout: str,
    top_effect: float, controls: list[float], expected_draws: int | None,
) -> dict:
    if expected_draws is not None and len(controls) != expected_draws:
        raise ValueError(
            f"seed {seed} k={k} {mode} {control_type} has {len(controls)} "
            f"draws, expected {expected_draws}"
        )
    if not controls:
        raise ValueError(f"No control draws for seed {seed} k={k} {mode} {control_type}")
    array = np.asarray(controls, dtype=float)
    contrasts = top_effect - array
    return {
        "seed": seed,
        "k": k,
        "mode": mode,
        "control_type": control_type,
        "readout": readout,
        "top_effect": top_effect,
        "control_draws": len(controls),
        "control_mean": float(array.mean()),
        "control_sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "top_minus_control_mean": float(contrasts.mean()),
        "top_minus_control_min": float(contrasts.min()),
        "top_minus_control_max": float(contrasts.max()),
        "wins": int((contrasts > 0).sum()),
        "ties": int((contrasts == 0).sum()),
        "empirical_upper_tail_p": float((1 + (array >= top_effect).sum()) / (len(array) + 1)),
    }


def _rows(path: Path, readout: str) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["rows"]
    if readout == "activation_global":
        for row in rows:
            row["_effect"] = float(row["trait_specific_global_effect_mean"])
    elif readout == "behavior_target_logprob":
        for row in rows:
            row["_effect"] = float(
                row["readouts"]["target_logprob"]["trait_specific_effect_mean"]
            )
    else:
        raise ValueError(f"Unknown readout: {readout}")
    return rows


def aggregate_root(root: Path, expected_draws: int | None) -> dict:
    summaries: list[dict] = []
    for seed_dir in sorted(root.glob("seed_*")):
        seed = int(seed_dir.name.removeprefix("seed_"))
        sources = (
            (seed_dir / "attribution" / "paired_topk.json", "activation_global"),
            (seed_dir / "attribution" / "paired_behavior.json", "behavior_target_logprob"),
        )
        for source, readout in sources:
            rows = _rows(source, readout)
            grouped: dict[tuple[int, str], list[dict]] = defaultdict(list)
            for row in rows:
                grouped[int(row["k"]), str(row["mode"])].append(row)
            for (k, mode), members in sorted(grouped.items()):
                top = [row for row in members if row["set_name"] == "top_k"]
                if len(top) != 1:
                    raise ValueError(
                        f"Expected one top-k row for seed {seed}, k={k}, mode={mode}; got {len(top)}"
                    )
                for control_type in ("random_control", "norm_matched_control"):
                    controls = [
                        row["_effect"] for row in members if row["set_name"] == control_type
                    ]
                    summaries.append(
                        _summarize(
                            seed=seed,
                            k=k,
                            mode=mode,
                            control_type=control_type,
                            readout=readout,
                            top_effect=top[0]["_effect"],
                            controls=controls,
                            expected_draws=expected_draws,
                        )
                    )

    cross_seed = []
    grouped_summaries: dict[tuple, list[dict]] = defaultdict(list)
    for row in summaries:
        key = (row["k"], row["mode"], row["control_type"], row["readout"])
        grouped_summaries[key].append(row)
    for key, members in sorted(grouped_summaries.items()):
        contrasts = [row["top_minus_control_mean"] for row in members]
        cross_seed.append(
            {
                "k": key[0],
                "mode": key[1],
                "control_type": key[2],
                "readout": key[3],
                "seeds": [row["seed"] for row in members],
                "per_seed_top_minus_control_mean": contrasts,
                "mean_across_seeds": float(np.mean(contrasts)),
                "all_seeds_positive": all(value > 0 for value in contrasts),
                "all_draws_won_in_all_seeds": all(
                    row["wins"] == row["control_draws"] for row in members
                ),
            }
        )
    return {
        "schema_version": 1,
        "analysis": "top_k_control_distributions",
        "root": str(root),
        "expected_control_draws": expected_draws,
        "per_seed": summaries,
        "cross_seed": cross_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-draws", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-csv")
    args = parser.parse_args()
    result = aggregate_root(repo_path(args.root), args.expected_draws)
    output = ensure_parent(repo_path(args.output))
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.output_csv:
        csv_path = ensure_parent(repo_path(args.output_csv))
        rows = result["per_seed"]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(f"Wrote {len(result['per_seed'])} per-seed control summaries to {output}")


if __name__ == "__main__":
    main()

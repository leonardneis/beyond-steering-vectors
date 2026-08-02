"""Aggregate the preregistered activation--behavior dissociation endpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.analysis.split_stability import bootstrap_mean_ci  # noqa: E402
from slgeo.io import ensure_parent  # noqa: E402


COMPONENTS = ("paired", "subliminal", "neutral")
METRICS = ("target_logprob", "target_probability", "target_vs_lion_margin", "target_choice")


def _indices(records: list[dict]) -> dict[str, np.ndarray]:
    families = sorted({str(row["family"]) for row in records})
    return {
        "pooled": np.arange(len(records), dtype=int),
        **{
            family: np.asarray([i for i, row in enumerate(records) if row["family"] == family], dtype=int)
            for family in families
        },
    }


def _ci(values: np.ndarray, samples: int, seed: int) -> list[float]:
    return list(bootstrap_mean_ci(values, samples=samples, seed=seed))


def _intervention_summaries(
    payload: dict, *, seed: int, expected_draws: int, bootstrap_samples: int, bootstrap_seed: int
) -> list[dict]:
    scopes = _indices(payload["prompt_records"])
    rows = payload["rows"]
    summaries = []
    for mode in ("necessity", "sufficiency"):
        mode_rows = [row for row in rows if int(row["k"]) == 20 and row["mode"] == mode]
        top_rows = [row for row in mode_rows if row["set_name"] == "top_k"]
        if len(top_rows) != 1:
            raise ValueError(f"Seed {seed} {mode} requires exactly one top-k row")
        top = top_rows[0]
        for control in ("random_control", "norm_matched_control"):
            controls = sorted(
                (row for row in mode_rows if row["set_name"] == control),
                key=lambda row: int(row["draw_id"]),
            )
            if len(controls) != expected_draws:
                raise ValueError(f"Seed {seed} {mode} {control} has {len(controls)} draws")
            for metric_index, metric in enumerate(METRICS):
                for component_index, component in enumerate(COMPONENTS):
                    top_values = np.asarray(top["readouts"][metric][component]["per_prompt"], dtype=float)
                    control_values = np.stack(
                        [np.asarray(row["readouts"][metric][component]["per_prompt"], dtype=float) for row in controls]
                    )
                    for scope_index, (scope, indices) in enumerate(scopes.items()):
                        top_scope = top_values[indices]
                        control_scope = control_values[:, indices]
                        top_mean = float(top_scope.mean())
                        control_means = control_scope.mean(axis=1)
                        prompt_contrast = top_scope - control_scope.mean(axis=0)
                        wins = int((top_mean > control_means).sum())
                        key_seed = bootstrap_seed + seed * 10000 + metric_index * 1000 + component_index * 100 + scope_index * 10 + (0 if mode == "necessity" else 5)
                        summaries.append(
                            {
                                "seed": seed,
                                "k": 20,
                                "mode": mode,
                                "control_type": control,
                                "metric": metric,
                                "component": component,
                                "scope": scope,
                                "prompt_count": int(len(indices)),
                                "top_effect": top_mean,
                                "control_mean": float(control_means.mean()),
                                "top_minus_control_mean": float(prompt_contrast.mean()),
                                "top_minus_control_ci95": _ci(prompt_contrast, bootstrap_samples, key_seed),
                                "wins": wins,
                                "control_draws": expected_draws,
                                "empirical_upper_tail_p": float((1 + (control_means >= top_mean).sum()) / (expected_draws + 1)),
                            }
                        )
    return summaries


def _full_adapter_summaries(payload: dict, *, seed: int, bootstrap_samples: int, bootstrap_seed: int) -> list[dict]:
    scopes = _indices(payload["prompt_records"])
    rows = []
    for metric_index, metric in enumerate(METRICS):
        for component_index, component in enumerate(COMPONENTS):
            values = np.asarray(payload["full_adapter"][metric][component]["per_prompt"], dtype=float)
            for scope_index, (scope, indices) in enumerate(scopes.items()):
                scoped = values[indices]
                rows.append(
                    {
                        "seed": seed,
                        "metric": metric,
                        "component": component,
                        "scope": scope,
                        "prompt_count": int(len(indices)),
                        "mean": float(scoped.mean()),
                        "ci95": _ci(
                            scoped,
                            bootstrap_samples,
                            bootstrap_seed + 50000 + seed * 1000 + metric_index * 100 + component_index * 10 + scope_index,
                        ),
                    }
                )
    return rows


def aggregate_root(
    root: Path, *, seeds: list[int], expected_draws: int, bootstrap_samples: int, bootstrap_seed: int
) -> dict:
    interventions, full_adapter = [], []
    prompt_hash = None
    for seed in seeds:
        path = root / f"seed_{seed}" / "behavior" / "paired.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("analysis") != "activation_behavior_dissociation_pair":
            raise ValueError(f"Unexpected paired artifact: {path}")
        if prompt_hash is None:
            prompt_hash = payload["prompt_file_sha256"]
        elif prompt_hash != payload["prompt_file_sha256"]:
            raise ValueError("Prompt-file checksums differ across seeds")
        interventions.extend(
            _intervention_summaries(
                payload,
                seed=seed,
                expected_draws=expected_draws,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
            )
        )
        full_adapter.extend(
            _full_adapter_summaries(
                payload, seed=seed, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed
            )
        )

    def select(rows: list[dict], **wanted) -> dict:
        matches = [row for row in rows if all(row.get(key) == value for key, value in wanted.items())]
        if len(matches) != 1:
            raise ValueError(f"Expected one summary for {wanted}, got {len(matches)}")
        return matches[0]

    primary = select(
        interventions,
        seed=2,
        mode="necessity",
        control_type="norm_matched_control",
        metric="target_logprob",
        component="paired",
        scope="pooled",
    )
    seed2_full = select(
        full_adapter, seed=2, metric="target_logprob", component="paired", scope="pooled"
    )
    other = [
        select(
            interventions,
            seed=seed,
            mode="necessity",
            control_type="norm_matched_control",
            metric="target_logprob",
            component="paired",
            scope="pooled",
        )["top_minus_control_mean"]
        for seed in (1, 3)
    ]
    mean, upper = primary["top_minus_control_mean"], primary["top_minus_control_ci95"][1]
    sparse_wins = primary["wins"] <= 5
    if mean < 0 and upper < 0 and sparse_wins:
        status = "strong_replication"
    elif mean < 0 and sparse_wins:
        status = "directional_replication"
    else:
        family_rows = [
            row for row in interventions
            if row["seed"] == 2
            and row["mode"] == "necessity"
            and row["control_type"] == "norm_matched_control"
            and row["metric"] == "target_logprob"
            and row["component"] == "paired"
            and row["scope"] != "pooled"
        ]
        status = "prompt_conditional_boundary" if any(row["top_minus_control_mean"] < 0 and row["wins"] <= 5 for row in family_rows) else "no_replication"
    decision = {
        "status": status,
        "primary": primary,
        "seed2_full_adapter_paired_target_logprob": seed2_full,
        "abd2_positive_learned_behavior": seed2_full["mean"] > 0,
        "abd3_other_seed_mean_minus_seed2": float(np.mean(other) - mean),
        "authorizes_seed2_centered_localization": status in {"strong_replication", "directional_replication"},
        "authorizes_prompt_conditional_localization": status == "prompt_conditional_boundary",
    }
    return {
        "schema_version": 1,
        "analysis": "activation_behavior_dissociation_aggregate",
        "root": str(root),
        "seeds": seeds,
        "expected_control_draws": expected_draws,
        "bootstrap_samples": bootstrap_samples,
        "prompt_file_sha256": prompt_hash,
        "intervention_summaries": interventions,
        "full_adapter_summaries": full_adapter,
        "decision_gate": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--expected-draws", type=int, default=25)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-csv")
    args = parser.parse_args()
    result = aggregate_root(
        repo_path(args.root),
        seeds=args.seeds,
        expected_draws=args.expected_draws,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = ensure_parent(repo_path(args.output))
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.output_csv:
        csv_path = ensure_parent(repo_path(args.output_csv))
        rows = result["intervention_summaries"]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result["decision_gate"], indent=2))


if __name__ == "__main__":
    main()

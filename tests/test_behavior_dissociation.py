from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_behavior_dissociation import aggregate_root  # noqa: E402
from compare_behavior_dissociation import compare_payloads  # noqa: E402


RECORDS = [
    {"prompt_id": "p1", "family": "direct", "prompt": "Prompt one"},
    {"prompt_id": "p2", "family": "scenario", "prompt": "Prompt two"},
]


def evaluation(values: list[float]) -> dict:
    rows = [
        {
            "target_logprob": value,
            "target_probability": value,
            "target_vs_lion_margin": value,
        }
        for value in values
    ]
    return {
        "target_choice_per_prompt": values,
        "token_metrics": {"rows": rows},
        "prompt_records": RECORDS,
    }


def raw_payload(full: list[float], intervention: list[float]) -> dict:
    return {
        "schema_version": 2,
        "selection_plan_sha256": "plan",
        "prompt_file_sha256": "prompts",
        "base_evaluation": evaluation([0.0, 0.0]),
        "full_evaluation": evaluation(full),
        "interventions": [
            {
                "k": 20,
                "set_name": "top_k",
                "draw_id": None,
                "mode": "necessity",
                "modules": ["m1"],
                **evaluation(intervention),
            }
        ],
    }


def test_pair_comparison_preserves_condition_decomposition() -> None:
    result = compare_payloads(
        raw_payload([2.0, 2.0], [1.0, 1.0]),
        raw_payload([0.5, 0.5], [0.4, 0.4]),
        bootstrap_samples=50,
        bootstrap_seed=1,
    )

    readout = result["rows"][0]["readouts"]["target_logprob"]
    assert readout["subliminal"]["mean"] == pytest.approx(1.0)
    assert readout["neutral"]["mean"] == pytest.approx(0.1)
    assert readout["paired"]["mean"] == pytest.approx(0.9)
    assert result["full_adapter"]["target_logprob"]["paired"]["mean"] == pytest.approx(1.5)


def paired_payload(seed: int) -> dict:
    top_value = -1.0 if seed == 2 else 1.0

    def readouts(value: float) -> dict:
        return {
            metric: {
                component: {
                    "mean": value,
                    "median": value,
                    "ci95": [value, value],
                    "per_prompt": [value, value],
                }
                for component in ("paired", "subliminal", "neutral")
            }
            for metric in ("target_logprob", "target_probability", "target_vs_lion_margin", "target_choice")
        }

    rows = []
    for mode in ("necessity", "sufficiency"):
        rows.append(
            {"k": 20, "set_name": "top_k", "draw_id": None, "mode": mode, "modules": ["m1"], "readouts": readouts(top_value)}
        )
        for control in ("random_control", "norm_matched_control"):
            for draw in range(2):
                rows.append(
                    {"k": 20, "set_name": control, "draw_id": draw, "mode": mode, "modules": [f"m{draw + 2}"], "readouts": readouts(0.0)}
                )
    return {
        "schema_version": 1,
        "analysis": "activation_behavior_dissociation_pair",
        "prompt_records": RECORDS,
        "prompt_file_sha256": "prompts",
        "selection_plan_sha256": f"plan-{seed}",
        "full_adapter": {
            metric: {
                component: {
                    "mean": 0.5,
                    "median": 0.5,
                    "ci95": [0.5, 0.5],
                    "per_prompt": [0.5, 0.5],
                }
                for component in ("paired", "subliminal", "neutral")
            }
            for metric in ("target_logprob", "target_probability", "target_vs_lion_margin", "target_choice")
        },
        "rows": rows,
    }


def test_aggregate_applies_preregistered_seed2_gate(tmp_path) -> None:
    for seed in (1, 2, 3):
        path = tmp_path / f"seed_{seed}" / "behavior" / "paired.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(paired_payload(seed)), encoding="utf-8")

    result = aggregate_root(
        tmp_path,
        seeds=[1, 2, 3],
        expected_draws=2,
        bootstrap_samples=50,
        bootstrap_seed=1,
    )

    assert result["decision_gate"]["status"] == "strong_replication"
    assert result["decision_gate"]["abd2_positive_learned_behavior"] is True
    assert result["decision_gate"]["abd3_other_seed_mean_minus_seed2"] == pytest.approx(2.0)

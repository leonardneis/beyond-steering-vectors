from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_control_distributions import aggregate_root  # noqa: E402


def test_control_distribution_aggregation(tmp_path: Path) -> None:
    attr = tmp_path / "seed_1" / "attribution"
    attr.mkdir(parents=True)
    activation_rows = []
    behavior_rows = []
    for mode in ("necessity", "sufficiency"):
        activation_rows.append(
            {"k": 20, "set_name": "top_k", "draw_id": None, "mode": mode,
             "trait_specific_global_effect_mean": 3.0}
        )
        behavior_rows.append(
            {"k": 20, "set_name": "top_k", "draw_id": None, "mode": mode,
             "readouts": {"target_logprob": {"trait_specific_effect_mean": 0.3}}}
        )
        for draw, value in enumerate((1.0, 2.0)):
            for control in ("random_control", "norm_matched_control"):
                activation_rows.append(
                    {"k": 20, "set_name": control, "draw_id": draw, "mode": mode,
                     "trait_specific_global_effect_mean": value}
                )
                behavior_rows.append(
                    {"k": 20, "set_name": control, "draw_id": draw, "mode": mode,
                     "readouts": {"target_logprob": {"trait_specific_effect_mean": value / 10}}}
                )
    (attr / "paired_topk.json").write_text(
        json.dumps({"rows": activation_rows}), encoding="utf-8"
    )
    (attr / "paired_behavior.json").write_text(
        json.dumps({"rows": behavior_rows}), encoding="utf-8"
    )
    result = aggregate_root(tmp_path, expected_draws=2)
    assert len(result["per_seed"]) == 8
    assert all(row["wins"] == 2 for row in result["per_seed"])
    assert all(row["all_seeds_positive"] for row in result["cross_seed"])

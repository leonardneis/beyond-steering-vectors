from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.analysis.selection_plans import iter_selection_sets
from slgeo.analysis.topk import prepare_module_set_distribution, prepare_module_sets


def test_topk_controls_are_distinct_and_layer_matched() -> None:
    modules = [f"model.layers.{layer}.{kind}" for layer in range(2) for kind in ("a", "b", "c", "d")]
    norms = {module: float(index + 1) for index, module in enumerate(modules)}
    result = prepare_module_sets(modules, norms, k_values=[1, 2], seed=7)
    assert result[1]["top_k"] == modules[:2]
    assert not (set(result[1]["top_k"]) & set(result[1]["random_control"]))
    for selected, control in zip(result[1]["top_k"], result[1]["layer_norm_matched_control"], strict=True):
        assert selected.split(".")[2] == control.split(".")[2]


def test_topk_can_omit_infeasible_layer_matched_controls() -> None:
    modules = [f"model.layers.{layer}.{kind}" for layer in range(2) for kind in ("a", "b", "c", "d")]
    norms = {module: float(index + 1) for index, module in enumerate(modules)}
    result = prepare_module_sets(
        modules,
        norms,
        k_values=[3],
        seed=11,
        matching_pool_size=2,
        control_types=("random", "norm"),
    )
    assert "random_control" in result[0]
    assert "norm_matched_control" in result[0]
    assert "layer_norm_matched_control" not in result[0]
    assert not (set(result[0]["top_k"]) & set(result[0]["norm_matched_control"]))


def test_distribution_plan_has_one_top_set_and_distinct_control_draws() -> None:
    modules = [f"model.layers.{layer}.{kind}" for layer in range(4) for kind in "abcdefg"]
    norms = {module: float(index + 1) for index, module in enumerate(modules)}
    rows = prepare_module_set_distribution(
        modules,
        norms,
        k_values=[5],
        seed=17,
        control_draws=4,
        matching_pool_size=3,
        control_types=("random", "norm"),
    )
    assert len([row for row in rows if row["set_name"] == "top_k"]) == 1
    for name in ("random_control", "norm_matched_control"):
        controls = [tuple(row["modules"]) for row in rows if row["set_name"] == name]
        assert len(controls) == len(set(controls)) == 4
        assert all(not (set(modules[:5]) & set(control)) for control in controls)


def test_selection_plan_iterator_normalizes_legacy_and_distribution_schemas() -> None:
    legacy = {
        "schema_version": 1,
        "sets": [{"k": 2, "top_k": ["a", "b"], "random_control": ["c", "d"]}],
    }
    assert list(iter_selection_sets(legacy, set_names=["random_control"])) == [
        {"k": 2, "set_name": "random_control", "draw_id": None, "modules": ["c", "d"]}
    ]
    distribution = {
        "schema_version": 2,
        "sets": [
            {"k": 2, "set_name": "top_k", "draw_id": None, "modules": ["a", "b"]},
            {"k": 2, "set_name": "random_control", "draw_id": 0, "modules": ["c", "d"]},
        ],
    }
    assert [row["draw_id"] for row in iter_selection_sets(distribution)] == [None, 0]

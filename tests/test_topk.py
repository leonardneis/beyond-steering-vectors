from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.analysis.topk import prepare_module_sets


def test_topk_controls_are_distinct_and_layer_matched() -> None:
    modules = [f"model.layers.{layer}.{kind}" for layer in range(2) for kind in ("a", "b", "c", "d")]
    norms = {module: float(index + 1) for index, module in enumerate(modules)}
    result = prepare_module_sets(modules, norms, k_values=[1, 2], seed=7)
    assert result[1]["top_k"] == modules[:2]
    assert not (set(result[1]["top_k"]) & set(result[1]["random_control"]))
    for selected, control in zip(result[1]["top_k"], result[1]["layer_norm_matched_control"], strict=True):
        assert selected.split(".")[2] == control.split(".")[2]

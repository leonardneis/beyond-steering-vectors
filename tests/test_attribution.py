from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.analysis.attribution import (
    group_lora_modules,
    layer_index,
    module_kind,
    prompt_drop_scores,
    summarize_prompt_values,
)


MODULES = [
    "base_model.model.model.layers.0.self_attn.q_proj",
    "base_model.model.model.layers.0.mlp.up_proj",
    "base_model.model.model.layers.2.self_attn.q_proj",
]


def test_parse_lora_module_coordinates() -> None:
    assert layer_index(MODULES[2]) == 2
    assert module_kind(MODULES[1]) == "up_proj"
    with pytest.raises(ValueError, match="Cannot parse"):
        layer_index("lm_head")


def test_group_modules_coarse_to_fine() -> None:
    by_layer = group_lora_modules(MODULES, group_by="layer")
    assert set(by_layer) == {"layer_00", "layer_02"}
    assert len(by_layer["layer_00"]) == 2
    by_kind = group_lora_modules(MODULES, group_by="module_kind")
    assert len(by_kind["q_proj"]) == 2
    individual = group_lora_modules(MODULES, group_by="individual", include_layers=[2])
    assert list(individual) == [MODULES[2]]


def test_prompt_drop_scores_respect_causal_layer_order() -> None:
    drops = torch.arange(2 * 6, dtype=torch.float32).reshape(2, 6)
    scores = prompt_drop_scores(drops, layer=2, fixed_target_block=3)
    assert scores["local_drop"].tolist() == [3.0, 9.0]
    assert scores["fixed_target_drop"].tolist() == [4.0, 10.0]
    assert scores["terminal_drop"].tolist() == [5.0, 11.0]
    assert scores["downstream_mean_drop"].tolist() == pytest.approx([4.0, 10.0])
    late = prompt_drop_scores(drops, layer=4, fixed_target_block=3)
    assert late["fixed_target_drop"] is None


def test_prompt_value_summary_reports_mean_median_and_sample_std() -> None:
    summary = summarize_prompt_values(torch.tensor([1.0, 2.0, 6.0]))
    assert summary["n"] == 3
    assert summary["mean"] == pytest.approx(3.0)
    assert summary["median"] == pytest.approx(2.0)
    assert summary["std"] == pytest.approx(2.6457513)

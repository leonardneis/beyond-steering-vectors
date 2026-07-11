from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.analysis.attribution import group_lora_modules, layer_index, module_kind


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

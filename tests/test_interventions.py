from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.analysis.interventions import (
    mask_lora_modules,
    replace_direction_component,
    residual_intervention,
)


class DummyLora(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scaling = {"default": 2.0}

    def forward(self, x):
        return x * self.scaling["default"]


class DummyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = DummyBackbone()
        self.adapter = DummyLora()

    def forward(self, x):
        return self.model(self.adapter(x))


def test_replace_direction_uses_base_component_only_on_direction() -> None:
    student = torch.tensor([[[3.0, 4.0]]])
    base = torch.tensor([[[1.0, 9.0]]])
    result = replace_direction_component(student, base, torch.tensor([1.0, 0.0]))
    assert result.tolist() == [[[1.0, 4.0]]]


def test_residual_add_intervention_is_reversible() -> None:
    model = DummyModel()
    x = torch.zeros(1, 2, 3)
    vectors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    with residual_intervention(model, vectors, mode="add", alpha=0.5):
        changed = model.model(x)
    assert changed[0, 0].tolist() == pytest.approx([0.5, 1.0, 0.0])
    assert torch.equal(model.model(x), x)


def test_mask_lora_modules_supports_ablation_and_reconstruction() -> None:
    model = DummyModel()
    name = "adapter"
    with mask_lora_modules(model, disabled_modules=[name]):
        assert model.adapter.scaling["default"] == 0
    assert model.adapter.scaling["default"] == 2
    with mask_lora_modules(model, enabled_modules=[]):
        assert model.adapter.scaling["default"] == 0


def test_mask_lora_modules_rejects_unknown_names_without_leaking_state() -> None:
    model = DummyModel()
    with pytest.raises(KeyError, match="not found"):
        with mask_lora_modules(model, disabled_modules=["missing"]):
            pass
    assert model.adapter.scaling["default"] == 2

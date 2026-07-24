from __future__ import annotations

import sys
from types import SimpleNamespace

import torch

from slgeo.models import load_causal_lm


def test_load_causal_lm_uses_transformers_4_torch_dtype(monkeypatch) -> None:
    captured = {}

    class AutoModel:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            captured["model_name"] = model_name
            captured["kwargs"] = kwargs
            return object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=AutoModel),
    )
    load_causal_lm(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype="float16",
        device_map="auto",
        local_files_only=True,
    )

    assert captured["model_name"] == "Qwen/Qwen2.5-7B-Instruct"
    assert captured["kwargs"]["torch_dtype"] is torch.float16
    assert "dtype" not in captured["kwargs"]


def test_condor_single_gpu_contract_disables_auto_dispatch(monkeypatch) -> None:
    captured = {}

    class AutoModel:
        @staticmethod
        def from_pretrained(_model_name, **kwargs):
            captured.update(kwargs)
            return object()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForCausalLM=AutoModel),
    )
    monkeypatch.setenv("SLGEO_FORCE_SINGLE_GPU", "1")
    load_causal_lm("model", device_map="auto")

    assert captured["device_map"] == {"": 0}

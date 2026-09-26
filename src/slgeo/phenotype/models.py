"""Base model and student-adapter loading for the Phenotype Anchor (outside ``slgeo.cts_stage0``).

The CTS loader refuses PEFT by design; this module loads the same pinned NF4 base (reusing the CTS quantization
and snapshot checks) and attaches LoRA student adapters by name, without merging. Scoring always receives the bare
``Qwen2ForCausalLM`` (``get_base_model()``), with exactly one adapter active or all adapters disabled.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch

from slgeo.cts_stage0.modeling import (
    EXPECTED_HIDDEN,
    EXPECTED_LAYERS,
    EXPECTED_VOCAB_ROWS,
    ModelContractError,
    quantization_config,
)

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
EXPECTED_LORA_MODULES = len(LORA_TARGETS) * EXPECTED_LAYERS
EXPECTED_RANK = 8
EXPECTED_ALPHA = 8
BASE = None  # arm adapter name for "all adapters disabled"


def adapter_tree_digest(path: str | Path) -> str:
    """SHA-256 over (relative name, file SHA-256) of every file under an adapter directory, sorted, excluding
    provenance sidecars: the definition of ``scripts/run_confirmatory_manifest.py::tree_digest``, so the digests
    pinned by the confirmatory study (seed 2: S 087f0bf2..., N ea2a581b...) remain comparable."""
    root = Path(path)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "run_provenance.json"
                   and not p.name.endswith(".provenance.json"))
    if not files:
        raise ModelContractError(f"Adapter directory {root} is empty")
    digest = hashlib.sha256()
    for p in files:
        digest.update(p.relative_to(root).as_posix().encode())
        digest.update(hashlib.sha256(p.read_bytes()).hexdigest().encode())
    return digest.hexdigest()


def load_base(snapshot: Path, model_config: Mapping[str, Any], *, device: str = "cuda:0"):
    """NF4 base model, fp16 compute, SDPA, one device, unquantized fp16 lm_head (same checks as CTS minus the
    PEFT refusal)."""
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ModelContractError("HF_HUB_OFFLINE=1 is required")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, torch_dtype=torch.float16, device_map={"": device},
        quantization_config=quantization_config(dict(model_config)), attn_implementation="sdpa",
    )
    model.eval()
    assert_base_geometry(model)
    return model


def assert_base_geometry(model) -> None:
    config = model.config
    if config.num_hidden_layers != EXPECTED_LAYERS or config.hidden_size != EXPECTED_HIDDEN:
        raise ModelContractError("Model geometry differs from Qwen2.5-7B")
    head = model.lm_head
    if type(head) is not torch.nn.Linear or tuple(head.weight.shape) != (EXPECTED_VOCAB_ROWS, EXPECTED_HIDDEN):
        raise ModelContractError("lm_head must be an unquantized [152064, 3584] nn.Linear")
    for name, _module in model.named_modules():
        if "lora" in name.lower():
            raise ModelContractError(f"Adapter-like module present before attachment: {name}")


def attach_adapters(model, adapters: Mapping[str, str | Path], expected_digests: Mapping[str, str]):
    """Attach every adapter by name (first via ``PeftModel.from_pretrained``, the rest via ``load_adapter``) after
    checking its tree digest. Returns the PeftModel. A digest that is ``None`` is refused (seed-3 must be pinned at
    first read, before any forward)."""
    from peft import PeftModel

    if not adapters:
        raise ModelContractError("No adapters to attach")
    peft_model = None
    for name, path in adapters.items():
        expected = expected_digests.get(name)
        if not expected:
            raise ModelContractError(f"Adapter {name} has no pinned digest")
        observed = adapter_tree_digest(path)
        if observed != expected:
            raise ModelContractError(f"Adapter {name} digest {observed} != pinned {expected}")
        if peft_model is None:
            peft_model = PeftModel.from_pretrained(model, str(path), adapter_name=name, is_trainable=False)
        else:
            peft_model.load_adapter(str(path), adapter_name=name, is_trainable=False)
    peft_model.eval()
    return peft_model


def lora_census(peft_model, name: str, *, rank: int = EXPECTED_RANK, alpha: int = EXPECTED_ALPHA,
                expected_modules: int = EXPECTED_LORA_MODULES) -> dict[str, Any]:
    """Every LoRA-wrapped projection carries the adapter with the frozen rank and alpha."""
    count = 0
    for module_name, module in peft_model.named_modules():
        lora_a = getattr(module, "lora_A", None)
        if lora_a is None or name not in lora_a:
            continue
        if not module_name.split(".")[-1] in LORA_TARGETS:
            raise ModelContractError(f"LoRA on unexpected module {module_name}")
        if module.r[name] != rank or module.lora_alpha[name] != alpha:
            raise ModelContractError(f"LoRA {module_name}: r={module.r[name]} alpha={module.lora_alpha[name]}")
        count += 1
    if count != expected_modules:
        raise ModelContractError(f"Adapter {name}: {count} LoRA modules, expected {expected_modules}")
    return {"adapter": name, "modules": count, "rank": rank, "alpha": alpha}


@contextlib.contextmanager
def active(peft_model, name: str | None) -> Iterator[Any]:
    """Yield the bare model with adapter ``name`` active, or with every adapter disabled (``name is None``)."""
    if name is None:
        with peft_model.disable_adapter():
            yield peft_model.get_base_model()
        return
    peft_model.set_adapter(name)
    if peft_model.active_adapter != name:
        raise ModelContractError(f"Active adapter is {peft_model.active_adapter!r}, expected {name!r}")
    yield peft_model.get_base_model()

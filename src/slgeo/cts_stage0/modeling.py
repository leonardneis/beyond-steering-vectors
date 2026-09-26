"""Loading of the frozen base model and tokenizer, with fail-closed checks.

The model is loaded from the pinned snapshot directory only (offline), NF4 4-bit with float16 compute and
no double quantization (``configs/model_qwen7b_4bit.yaml``), SDPA attention, fully on one GPU.
PEFT is never imported; the loaded object must be a bare ``Qwen2ForCausalLM``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

from .package import MODEL_REVISION, FrozenPackageError, sha256_path

EXPECTED_VOCAB_ROWS = 152064
EXPECTED_HIDDEN = 3584
EXPECTED_LAYERS = 28


class ModelContractError(RuntimeError):
    """The loaded model or tokenizer deviates from the frozen execution contract."""


def snapshot_directory(hf_home: str | Path, model_id: str = "Qwen/Qwen2.5-7B-Instruct") -> Path:
    hub = Path(hf_home) / "hub"
    if not hub.is_dir():
        hub = Path(hf_home)
    directory = hub / f"models--{model_id.replace('/', '--')}" / "snapshots" / MODEL_REVISION
    if not directory.is_dir():
        raise ModelContractError(f"Pinned snapshot {MODEL_REVISION} not found under {hub}")
    return directory


def verify_snapshot(directory: Path, expected_hashes: dict[str, str]) -> dict[str, str]:
    """Re-hash every pinned snapshot file (weights, config, tokenizer); fail closed on any mismatch."""
    if directory.name != MODEL_REVISION:
        raise ModelContractError("Snapshot directory is not the pinned revision")
    observed = {}
    for name, expected in sorted(expected_hashes.items()):
        path = directory / name
        if not path.is_file():
            raise ModelContractError(f"Snapshot file missing: {name}")
        observed[name] = sha256_path(path)
        if observed[name] != expected:
            raise ModelContractError(f"Snapshot file hash mismatch: {name}")
    return observed


def set_deterministic() -> dict[str, Any]:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def quantization_config(model_config: dict[str, Any]):
    from transformers import BitsAndBytesConfig

    quant = model_config["quantization"]
    expected = {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "float16"}
    for key, value in expected.items():
        if quant.get(key) != value:
            raise ModelContractError(f"Model config quantization.{key} = {quant.get(key)!r}, expected {value!r}")
    if quant.get("bnb_4bit_use_double_quant", False):
        raise ModelContractError("Double quantization is not part of the frozen model config")
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )


def load_tokenizer(snapshot: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)


def load_model(snapshot: Path, model_config: dict[str, Any]):
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ModelContractError("HF_HUB_OFFLINE=1 is required")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(snapshot),
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map={"": 0},
        quantization_config=quantization_config(model_config),
        attn_implementation="sdpa",
    )
    model.eval()
    assert_base_model(model)
    return model


def assert_base_model(model) -> dict[str, Any]:
    """Structural checks of the loaded model (no PEFT, unquantized head, one device, NF4)."""
    if "peft" in sys.modules:
        raise ModelContractError("peft is imported; Stage 0 must never load an adapter")
    if type(model).__name__ != "Qwen2ForCausalLM" or hasattr(model, "peft_config"):
        raise ModelContractError(f"Unexpected model type {type(model).__name__}")
    for name, _module in model.named_modules():
        if "lora" in name.lower():
            raise ModelContractError(f"Adapter-like module present: {name}")
    config = model.config
    if config.num_hidden_layers != EXPECTED_LAYERS or config.hidden_size != EXPECTED_HIDDEN:
        raise ModelContractError("Model geometry differs from Qwen2.5-7B")
    head = model.lm_head
    if type(head) is not torch.nn.Linear or tuple(head.weight.shape) != (EXPECTED_VOCAB_ROWS, EXPECTED_HIDDEN):
        raise ModelContractError("lm_head must be an unquantized [152064, 3584] nn.Linear")
    if head.weight.dtype != torch.float16:
        raise ModelContractError("lm_head weight must be float16")
    if type(model.model.embed_tokens) is not torch.nn.Embedding:
        raise ModelContractError("embed_tokens must be an unquantized nn.Embedding")
    devices = {str(parameter.device) for parameter in model.parameters()}
    if devices != {"cuda:0"}:
        raise ModelContractError(f"Model parameters are not all on cuda:0: {sorted(devices)}")
    quant = getattr(config, "quantization_config", None)
    quant = quant.to_dict() if hasattr(quant, "to_dict") else dict(quant or {})
    if quant.get("bnb_4bit_quant_type") != "nf4" or not quant.get("load_in_4bit"):
        raise ModelContractError("Loaded quantization is not NF4 4-bit")
    if str(quant.get("bnb_4bit_compute_dtype")) not in {"float16", "torch.float16"}:
        raise ModelContractError("Loaded compute dtype is not float16")
    if quant.get("bnb_4bit_use_double_quant"):
        raise ModelContractError("Loaded model uses double quantization")
    if getattr(config, "_attn_implementation", None) != "sdpa":
        raise ModelContractError("Attention implementation is not sdpa")
    return {"quantization": quant, "attn_implementation": "sdpa", "dtype": str(model.dtype)}


def model_config_sha256(path: str | Path) -> str:
    try:
        return sha256_path(path)
    except OSError as exc:  # pragma: no cover - surfaced as contract error
        raise FrozenPackageError(f"Model config not readable: {path}") from exc

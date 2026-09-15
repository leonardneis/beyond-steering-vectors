"""Auditable model-execution helpers for C18 runners and technical validation."""

from __future__ import annotations

from contextlib import contextmanager
from importlib.metadata import version
import os
import platform
import subprocess
from typing import Sequence

import numpy as np
import torch

from .interventions import decoder_blocks, list_lora_modules


def runtime_identity(attention_backend: str) -> dict[str, object]:
    driver = None
    if torch.cuda.is_available():
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True, capture_output=True, text=True,
            ).stdout.splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError, IndexError):
            driver = None
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "transformers": version("transformers"),
        "peft": version("peft"), "bitsandbytes": version("bitsandbytes"),
        "numpy": version("numpy"),
        "gpu_class": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "attention_backend": attention_backend,
        "container_image": os.environ.get("SLGEO_CONTAINER_IMAGE_DIGEST"),
        "nvidia_driver": driver,
    }


def assert_runtime_identity(manifest: dict) -> dict[str, object]:
    expected = manifest["execution"]
    actual = runtime_identity(expected["attention_backend"])
    for key in ("python", "torch", "cuda_runtime", "transformers", "peft", "bitsandbytes", "numpy", "gpu_class", "attention_backend", "container_image", "nvidia_driver"):
        if str(actual[key]) != str(expected[key]):
            raise RuntimeError(f"execution identity mismatch for {key}: {actual[key]!r} != {expected[key]!r}")
    return actual


def unpack_block_output(output):
    return (output[0], output[1:]) if isinstance(output, tuple) else (output, None)


def repack_block_output(hidden: torch.Tensor, tail):
    return hidden if tail is None else (hidden, *tail)


@contextmanager
def capture_block_state(model, block_index: int):
    captured: list[torch.Tensor] = []
    block = decoder_blocks(model)[block_index]

    def hook(_module, _args, output):
        hidden, _tail = unpack_block_output(output)
        state = hidden.detach().clone()
        state.requires_grad_(False)
        captured.append(state)

    handle = block.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def replace_block_output(model, block_index: int, replacement: torch.Tensor):
    """Replace a complete token state at a cut, preserving tuple tails."""
    calls: list[int] = []
    block = decoder_blocks(model)[block_index]
    frozen = replacement.detach().clone()
    frozen.requires_grad_(False)

    def hook(_module, _args, output):
        hidden, tail = unpack_block_output(output)
        if hidden.shape != frozen.shape or hidden.dtype != frozen.dtype:
            raise ValueError("full-state replay shape/dtype mismatch")
        calls.append(block_index)
        return repack_block_output(frozen.to(hidden.device), tail)

    handle = block.register_forward_hook(hook)
    try:
        yield calls
    finally:
        handle.remove()


@contextmanager
def identity_block_hooks(model, block_indices: Sequence[int] = tuple(range(27))):
    calls: list[int] = []
    handles = []
    for index in block_indices:
        def hook(_module, _args, output, block=index):
            calls.append(block)
            return output
        handles.append(decoder_blocks(model)[index].register_forward_hook(hook))
    try:
        yield calls
    finally:
        for handle in handles:
            handle.remove()


def candidate_logits_and_state(
    model, inputs: dict[str, torch.Tensor], candidate_ids: Sequence[int], *, diagnostics: bool = False
):
    """Run a complete cache-free prefill and return post-RMS state and candidate logits."""
    with torch.inference_mode():
        output = model(
            **inputs,
            use_cache=False,
            past_key_values=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
    state = output.hidden_states[-1]
    indices = inputs["attention_mask"].sum(dim=-1) - 1
    # Left padding places every last real token at physical index -1. The
    # logical indices are retained only as an explicit validation diagnostic.
    if not torch.all(indices >= 0):
        raise ValueError("empty token sequence")
    last = output.logits[:, -1, :]
    selected = last[:, candidate_ids].detach()
    if not diagnostics:
        return state.detach(), selected
    probabilities = torch.softmax(last.float(), dim=-1)[:, candidate_ids].detach()
    candidate_distribution = torch.softmax(selected.float(), dim=-1)
    candidate_entropy = -(candidate_distribution * torch.log(candidate_distribution.clamp_min(1e-30))).sum(dim=-1)
    return state.detach(), selected, probabilities, candidate_entropy.detach()


def margin_from_direction(final_state: torch.Tensor, direction: np.ndarray) -> np.ndarray:
    vector = torch.as_tensor(direction, dtype=torch.float64, device=final_state.device)
    value = torch.einsum("bh,h->b", final_state[:, -1, :].to(torch.float64), vector)
    return value.detach().cpu().numpy()


def snapshot_lora_scalings(model, adapter_name: str = "default") -> dict[str, object]:
    snapshot = {}
    for name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and adapter_name in scaling:
            canonical = name.removesuffix(".base_layer")
            if canonical in snapshot:
                raise ValueError(f"duplicate canonical LoRA module: {canonical}")
            snapshot[canonical] = scaling[adapter_name]
    return snapshot


def assert_lora_census(model, *, expected: int = 196) -> list[str]:
    names = list_lora_modules(model)
    if len(names) != expected or len(set(names)) != expected:
        raise ValueError(f"LoRA census differs: {len(names)}")
    return names


def assert_scalings_restored(model, before: dict[str, object], adapter_name: str = "default") -> None:
    after = snapshot_lora_scalings(model, adapter_name)
    if set(before) != set(after):
        raise RuntimeError("LoRA module inventory changed during intervention")
    for name in before:
        if after[name] != before[name]:
            raise RuntimeError(f"LoRA scaling was not restored: {name}")


def suffix_factorial_values(w: dict[str, np.ndarray], *, donor: int) -> dict[str, np.ndarray]:
    """Return the two exact allocations and state-by-suffix interaction."""
    needed = {f"W0{donor}0", f"W0{donor}1", f"W1{donor}0", f"W1{donor}1"}
    if not needed <= set(w):
        raise ValueError(f"missing W cells: {sorted(needed - set(w))}")
    w0b0, w0b1 = np.asarray(w[f"W0{donor}0"], dtype=np.float64), np.asarray(w[f"W0{donor}1"], dtype=np.float64)
    w1b0, w1b1 = np.asarray(w[f"W1{donor}0"], dtype=np.float64), np.asarray(w[f"W1{donor}1"], dtype=np.float64)
    total = w1b1 - w0b0
    upstream_suffix0 = w1b0 - w0b0
    suffix_at_state1 = w1b1 - w1b0
    upstream_suffix1 = w1b1 - w0b1
    suffix_at_state0 = w0b1 - w0b0
    interaction = (w1b1 - w1b0) - (w0b1 - w0b0)
    if not np.array_equal(total, upstream_suffix0 + suffix_at_state1):
        tolerance = 64 * np.finfo(np.float64).eps * np.maximum(1, np.abs(total) + np.abs(upstream_suffix0) + np.abs(suffix_at_state1))
        if np.any(np.abs(total - upstream_suffix0 - suffix_at_state1) > tolerance):
            raise ArithmeticError("suffix-0 allocation identity failed")
    if np.any(np.abs(total - upstream_suffix1 - suffix_at_state0) > 64 * np.finfo(np.float64).eps * np.maximum(1, np.abs(total) + np.abs(upstream_suffix1) + np.abs(suffix_at_state0))):
        raise ArithmeticError("suffix-1 allocation identity failed")
    return {
        "total": total,
        "upstream_under_suffix0": upstream_suffix0,
        "suffix_at_state1": suffix_at_state1,
        "upstream_under_suffix1": upstream_suffix1,
        "suffix_at_state0": suffix_at_state0,
        "state_suffix_interaction": interaction,
    }

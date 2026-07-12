"""Reversible activation and LoRA interventions for causal analyses."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable, Mapping

import torch


def decoder_blocks(model):
    """Resolve transformer blocks through common Transformers and PEFT wrappers."""
    paths = (
        ("model", "layers"),
        ("model", "model", "layers"),
        ("base_model", "model", "model", "layers"),
        ("language_model", "model", "layers"),
    )
    for path in paths:
        current = model
        for name in path:
            current = getattr(current, name, None)
            if current is None:
                break
        if current is not None:
            return current
    raise AttributeError(f"Could not resolve decoder blocks under {type(model).__name__}")


def _unpack_hidden(output):
    return (output[0], output[1:]) if isinstance(output, tuple) else (output, None)


def _repack_hidden(hidden, tail):
    return hidden if tail is None else (hidden, *tail)


def replace_direction_component(
    student_hidden: torch.Tensor,
    base_hidden: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """Replace the student's component along ``vector`` with the base component."""
    if student_hidden.shape != base_hidden.shape:
        raise ValueError("Student and base hidden states must have identical shapes")
    unit = vector.to(student_hidden).flatten()
    unit = unit / torch.linalg.vector_norm(unit).clamp_min(1e-12)
    student_coef = torch.einsum("...h,h->...", student_hidden, unit).unsqueeze(-1)
    base_coef = torch.einsum("...h,h->...", base_hidden.to(student_hidden), unit).unsqueeze(-1)
    return student_hidden + (base_coef - student_coef) * unit


@contextmanager
def capture_block_outputs(model, block_indices: Iterable[int]):
    """Capture detached block outputs during forwards inside the context."""
    blocks = decoder_blocks(model)
    captured: dict[int, torch.Tensor | None] = {int(index): None for index in block_indices}
    handles = []
    for index in captured:
        def hook(_module, _args, output, block_index=index):
            hidden, _ = _unpack_hidden(output)
            captured[block_index] = hidden.detach()

        handles.append(blocks[index].register_forward_hook(hook))
    try:
        yield captured
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def residual_intervention(
    model,
    vectors: torch.Tensor,
    *,
    mode: str,
    block_indices: Iterable[int] | None = None,
    alpha: float = 1.0,
    positions: str = "all",
    base_outputs: Mapping[int, torch.Tensor | None] | None = None,
):
    """Temporarily add, project, or replace a residual-stream direction.

    ``vectors`` is indexed by transformer block, not by hidden-state slot. Strip
    the embedding slot before passing vectors extracted from ``hidden_states``.
    """
    if mode not in {"add", "project", "replace_base"}:
        raise ValueError("mode must be add, project, or replace_base")
    if positions not in {"all", "last"}:
        raise ValueError("positions must be all or last")
    blocks = decoder_blocks(model)
    indices = list(range(len(blocks))) if block_indices is None else [int(i) for i in block_indices]
    if vectors.ndim != 2 or vectors.shape[0] != len(blocks):
        raise ValueError(
            f"Expected block-indexed vectors [{len(blocks)}, hidden], got {tuple(vectors.shape)}"
        )
    if mode == "replace_base" and base_outputs is None:
        raise ValueError("replace_base requires captured base_outputs")

    handles = []
    for index in indices:
        vector = vectors[index].detach()

        def hook(_module, _args, output, block_index=index, direction=vector):
            hidden, tail = _unpack_hidden(output)
            unit = direction.to(hidden)
            unit = unit / torch.linalg.vector_norm(unit).clamp_min(1e-12)
            selected = hidden if positions == "all" else hidden[:, -1:, :]
            if mode == "add":
                changed = selected + alpha * direction.to(hidden)
            elif mode == "project":
                coefficient = torch.einsum("...h,h->...", selected, unit).unsqueeze(-1)
                changed = selected - alpha * coefficient * unit
            else:
                base_hidden = base_outputs[block_index]
                if base_hidden is None:
                    raise RuntimeError(f"No captured base output for block {block_index}")
                base_selected = base_hidden.to(hidden)
                if positions == "last":
                    base_selected = base_selected[:, -1:, :]
                changed = replace_direction_component(selected, base_selected, unit)
            if positions == "all":
                new_hidden = changed
            else:
                new_hidden = hidden.clone()
                new_hidden[:, -1:, :] = changed
            return _repack_hidden(new_hidden, tail)

        handles.append(blocks[index].register_forward_hook(hook))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def _canonical_lora_name(name: str) -> str:
    return name.removesuffix(".base_layer")


def list_lora_modules(model, adapter_name: str = "default") -> list[str]:
    """List canonical PEFT module names that expose scaling for an adapter."""
    names = []
    for name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and adapter_name in scaling:
            names.append(_canonical_lora_name(name))
    return sorted(set(names))


@contextmanager
def mask_lora_modules(
    model,
    *,
    enabled_modules: Iterable[str] | None = None,
    disabled_modules: Iterable[str] | None = None,
    adapter_name: str = "default",
):
    """Temporarily set selected PEFT LoRA module scalings to zero.

    Exactly one of ``enabled_modules`` or ``disabled_modules`` may be supplied.
    An enabled set implements isolated reconstruction; a disabled set implements
    ablation. Original scaling values are restored even if inference fails.
    """
    if enabled_modules is not None and disabled_modules is not None:
        raise ValueError("Specify enabled_modules or disabled_modules, not both")
    enabled = set(enabled_modules) if enabled_modules is not None else None
    disabled = set(disabled_modules or ())
    originals: list[tuple[dict, str, float]] = []
    seen = set()
    for name, module in model.named_modules():
        scaling = getattr(module, "scaling", None)
        if not isinstance(scaling, dict) or adapter_name not in scaling:
            continue
        canonical = _canonical_lora_name(name)
        seen.add(canonical)
        should_mask = canonical not in enabled if enabled is not None else canonical in disabled
        if should_mask:
            originals.append((scaling, adapter_name, scaling[adapter_name]))
            scaling[adapter_name] = 0.0
    requested = enabled if enabled is not None else disabled
    missing = requested - seen
    if missing:
        for scaling, key, value in originals:
            scaling[key] = value
        raise KeyError(f"Requested LoRA modules not found: {sorted(missing)}")
    try:
        yield
    finally:
        for scaling, key, value in originals:
            scaling[key] = value

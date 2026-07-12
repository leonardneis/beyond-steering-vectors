"""Grouping and ranking helpers for staged LoRA causal attribution."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

import torch


_LAYER_PATTERN = re.compile(r"\.layers\.(\d+)\.")


def layer_index(module_name: str) -> int:
    match = _LAYER_PATTERN.search(module_name)
    if not match:
        raise ValueError(f"Cannot parse transformer block from {module_name!r}")
    return int(match.group(1))


def module_kind(module_name: str) -> str:
    return module_name.rsplit(".", 1)[-1]


def group_lora_modules(
    module_names: Iterable[str],
    *,
    group_by: str,
    include_layers: Iterable[int] | None = None,
) -> dict[str, list[str]]:
    """Group modules for coarse-to-fine intervention experiments."""
    if group_by not in {"layer", "module_kind", "individual"}:
        raise ValueError("group_by must be layer, module_kind, or individual")
    allowed_layers = set(include_layers) if include_layers is not None else None
    groups: dict[str, list[str]] = defaultdict(list)
    for name in sorted(module_names):
        layer = layer_index(name)
        if allowed_layers is not None and layer not in allowed_layers:
            continue
        if group_by == "layer":
            key = f"layer_{layer:02d}"
        elif group_by == "module_kind":
            key = module_kind(name)
        else:
            key = name
        groups[key].append(name)
    return dict(groups)


def prompt_drop_scores(
    projection_drop: torch.Tensor,
    *,
    layer: int,
    fixed_target_block: int,
) -> dict[str, torch.Tensor | None]:
    """Return per-prompt causal scores with valid layer/slot relationships."""
    if projection_drop.ndim != 2:
        raise ValueError("projection_drop must have shape [prompts, hidden_state_slots]")
    local_slot = layer + 1
    fixed_slot = fixed_target_block + 1
    if local_slot >= projection_drop.shape[1] or fixed_slot >= projection_drop.shape[1]:
        raise ValueError("Layer or fixed target lies outside hidden-state slots")
    return {
        "local_drop": projection_drop[:, local_slot],
        "fixed_target_drop": (
            projection_drop[:, fixed_slot] if layer <= fixed_target_block else None
        ),
        "terminal_drop": projection_drop[:, -1],
        "downstream_mean_drop": projection_drop[:, local_slot:].mean(dim=1),
    }


def summarize_prompt_values(values: torch.Tensor | None) -> dict[str, float | int] | None:
    if values is None:
        return None
    values = values.detach().float().cpu()
    return {
        "n": int(values.numel()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(unbiased=True)) if values.numel() > 1 else 0.0,
    }

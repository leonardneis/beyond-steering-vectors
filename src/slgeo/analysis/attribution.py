"""Grouping and ranking helpers for staged LoRA causal attribution."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable


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

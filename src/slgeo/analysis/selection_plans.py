"""Compatibility helpers for legacy and distributional module-set plans."""

from __future__ import annotations

from collections.abc import Iterable, Iterator


LEGACY_SET_NAMES = (
    "top_k",
    "random_control",
    "norm_matched_control",
    "layer_norm_matched_control",
)


def iter_selection_sets(
    plan: dict,
    *,
    set_names: Iterable[str] | None = None,
    k_values: Iterable[int] | None = None,
) -> Iterator[dict]:
    """Yield normalized ``k/set_name/draw_id/modules`` rows from a plan."""
    wanted_names = set(set_names) if set_names is not None else None
    wanted_k = {int(value) for value in k_values} if k_values is not None else None
    schema_version = int(plan.get("schema_version", 1))
    if schema_version == 1:
        for item in plan["sets"]:
            k = int(item["k"])
            if wanted_k is not None and k not in wanted_k:
                continue
            for set_name in LEGACY_SET_NAMES:
                if set_name not in item or (
                    wanted_names is not None and set_name not in wanted_names
                ):
                    continue
                yield {
                    "k": k,
                    "set_name": set_name,
                    "draw_id": None,
                    "modules": list(item[set_name]),
                }
        return
    if schema_version != 2:
        raise ValueError(f"Unsupported selection-plan schema_version: {schema_version}")
    for item in plan["sets"]:
        k, set_name = int(item["k"]), str(item["set_name"])
        if wanted_k is not None and k not in wanted_k:
            continue
        if wanted_names is not None and set_name not in wanted_names:
            continue
        yield {
            "k": k,
            "set_name": set_name,
            "draw_id": item.get("draw_id"),
            "modules": list(item["modules"]),
        }

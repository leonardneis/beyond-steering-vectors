"""Deterministic top-k and matched-control module selection."""

from __future__ import annotations

import random
from collections import Counter

from .attribution import layer_index


def nearest_norm_matching(
    selected: list[str],
    candidates: list[str],
    norms: dict[str, float],
    *,
    match_layer: bool,
    rng: random.Random | None = None,
    matching_pool_size: int = 1,
) -> list[str]:
    """Select distinct controls from the nearest LoRA-norm candidates.

    ``matching_pool_size=1`` preserves deterministic nearest-neighbour matching.
    Larger pools sample reproducibly and expose uncertainty due to control choice.
    """
    if matching_pool_size < 1:
        raise ValueError("matching_pool_size must be >= 1")
    available = set(candidates) - set(selected)
    matched = []
    for module in selected:
        pool = [candidate for candidate in available if not match_layer or layer_index(candidate) == layer_index(module)]
        if not pool:
            qualifier = "layer-matched " if match_layer else ""
            raise ValueError(f"No {qualifier}control remains for {module}")
        nearest = sorted(pool, key=lambda candidate: (abs(norms[candidate] - norms[module]), candidate))[
            :matching_pool_size
        ]
        choice = (rng or random).choice(nearest) if len(nearest) > 1 else nearest[0]
        matched.append(choice)
        available.remove(choice)
    return matched


def prepare_module_sets(
    ranked_modules: list[str],
    norms: dict[str, float],
    *,
    k_values: list[int],
    seed: int,
    matching_pool_size: int = 1,
    control_types: tuple[str, ...] = ("random", "norm", "layer_norm"),
) -> list[dict]:
    """Create nested top-k sets with random, norm-, and layer-matched controls."""
    if set(ranked_modules) - set(norms):
        raise ValueError("Missing norm values for ranked modules")
    rng = random.Random(seed)
    sets = []
    for k in sorted(set(k_values)):
        if k <= 0 or k >= len(ranked_modules):
            raise ValueError(f"k must be between 1 and {len(ranked_modules)-1}, got {k}")
        top = ranked_modules[:k]
        remaining = [module for module in ranked_modules if module not in top]
        item = {
            "k": k,
            "top_k": top,
            "top_k_layer_counts": dict(sorted(Counter(map(layer_index, top)).items())),
        }
        if "random" in control_types:
            item["random_control"] = rng.sample(remaining, k)
        if "norm" in control_types:
            item["norm_matched_control"] = nearest_norm_matching(
                top,
                ranked_modules,
                norms,
                match_layer=False,
                rng=rng,
                matching_pool_size=matching_pool_size,
            )
        if "layer_norm" in control_types:
            item["layer_norm_matched_control"] = nearest_norm_matching(
                top,
                ranked_modules,
                norms,
                match_layer=True,
                rng=rng,
                matching_pool_size=matching_pool_size,
            )
        sets.append(item)
    return sets

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
) -> list[str]:
    """Greedily select distinct non-target modules with closest LoRA update norm."""
    available = set(candidates) - set(selected)
    matched = []
    for module in selected:
        pool = [candidate for candidate in available if not match_layer or layer_index(candidate) == layer_index(module)]
        if not pool:
            qualifier = "layer-matched " if match_layer else ""
            raise ValueError(f"No {qualifier}control remains for {module}")
        choice = min(pool, key=lambda candidate: (abs(norms[candidate] - norms[module]), candidate))
        matched.append(choice)
        available.remove(choice)
    return matched


def prepare_module_sets(
    ranked_modules: list[str],
    norms: dict[str, float],
    *,
    k_values: list[int],
    seed: int,
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
        random_control = rng.sample(remaining, k)
        norm_control = nearest_norm_matching(top, ranked_modules, norms, match_layer=False)
        layer_control = nearest_norm_matching(top, ranked_modules, norms, match_layer=True)
        sets.append(
            {
                "k": k,
                "top_k": top,
                "random_control": random_control,
                "norm_matched_control": norm_control,
                "layer_norm_matched_control": layer_control,
                "top_k_layer_counts": dict(sorted(Counter(map(layer_index, top)).items())),
            }
        )
    return sets

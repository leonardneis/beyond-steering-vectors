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


def prepare_module_set_distribution(
    ranked_modules: list[str],
    norms: dict[str, float],
    *,
    k_values: list[int],
    seed: int,
    control_draws: int,
    matching_pool_size: int = 1,
    control_types: tuple[str, ...] = ("random", "norm"),
) -> list[dict]:
    """Create one top-k row and reproducibly distinct control draws per k.

    The normalized row schema keeps the selected modules explicit and gives
    every stochastic control a stable ``draw_id``.  It is intended for null
    distributions; ``prepare_module_sets`` remains the legacy single-plan API.
    """
    if control_draws < 1:
        raise ValueError("control_draws must be >= 1")
    unsupported = set(control_types) - {"random", "norm", "layer_norm"}
    if unsupported:
        raise ValueError(f"Unsupported control types: {sorted(unsupported)}")

    rows: list[dict] = []
    seen: dict[tuple[int, str], set[tuple[str, ...]]] = {}
    key_names = {
        "random": "random_control",
        "norm": "norm_matched_control",
        "layer_norm": "layer_norm_matched_control",
    }
    for k in sorted(set(k_values)):
        if k <= 0 or k >= len(ranked_modules):
            raise ValueError(f"k must be between 1 and {len(ranked_modules)-1}, got {k}")
        top = ranked_modules[:k]
        rows.append(
            {
                "k": k,
                "set_name": "top_k",
                "draw_id": None,
                "modules": top,
                "layer_counts": dict(sorted(Counter(map(layer_index, top)).items())),
            }
        )
        for control_type in control_types:
            set_name = key_names[control_type]
            seen[k, set_name] = set()
            for draw_id in range(control_draws):
                selected = None
                # A bounded deterministic retry prevents duplicate null draws.
                for attempt in range(1000):
                    draw_seed = seed + k * 1_000_000 + draw_id * 1000 + attempt
                    plan = prepare_module_sets(
                        ranked_modules,
                        norms,
                        k_values=[k],
                        seed=draw_seed,
                        matching_pool_size=matching_pool_size,
                        control_types=(control_type,),
                    )[0]
                    candidate = tuple(plan[set_name])
                    if candidate not in seen[k, set_name]:
                        seen[k, set_name].add(candidate)
                        selected = list(candidate)
                        break
                if selected is None:
                    raise ValueError(
                        f"Could not construct {control_draws} distinct {set_name} draws for k={k}"
                    )
                rows.append(
                    {
                        "k": k,
                        "set_name": set_name,
                        "draw_id": draw_id,
                        "modules": selected,
                        "layer_counts": dict(
                            sorted(Counter(map(layer_index, selected)).items())
                        ),
                    }
                )
    return rows

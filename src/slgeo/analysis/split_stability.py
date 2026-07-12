"""Prompt-paired trait-specific scores and split-stability statistics."""

from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr


READOUTS = ("local", "fixed_target", "terminal", "downstream_mean")


def trait_specific_prompt_scores(subliminal: dict, neutral: dict) -> dict[int, dict[str, np.ndarray | None]]:
    """Derive per-prompt subliminal-minus-neutral scores from schema-v2 screens."""
    if subliminal.get("schema_version") != 2 or neutral.get("schema_version") != 2:
        raise ValueError("Both attribution inputs must use schema version 2")
    paired_fields = ("teacher_vector_sha256", "prompts_sha256", "n_prompts", "prompt_offset", "position")
    mismatches = [key for key in paired_fields if subliminal.get(key) != neutral.get(key)]
    if mismatches:
        raise ValueError(f"Inputs are not prompt-paired; mismatches: {mismatches}")
    sub = {int(row["layer"]): row for row in subliminal["ablations"]}
    control = {int(row["layer"]): row for row in neutral["ablations"]}
    if set(sub) != set(control):
        raise ValueError("Layer sets differ between conditions")
    fixed_slot = int(subliminal["target_hidden_state_slot"])
    scores = {}
    for layer in sorted(sub):
        sub_drop = np.asarray(sub[layer]["projection_drop_per_prompt"], dtype=np.float64)
        control_drop = np.asarray(control[layer]["projection_drop_per_prompt"], dtype=np.float64)
        if sub_drop.shape != control_drop.shape:
            raise ValueError(f"Projection shapes differ at layer {layer}")
        trait = sub_drop - control_drop
        local_slot = layer + 1
        scores[layer] = {
            "local": trait[:, local_slot],
            "fixed_target": trait[:, fixed_slot] if local_slot <= fixed_slot else None,
            "terminal": trait[:, -1],
            "downstream_mean": trait[:, local_slot:].mean(axis=1),
        }
    return scores


def bootstrap_mean_ci(values: np.ndarray, *, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def split_readout_stability(
    split_a: dict[int, dict[str, np.ndarray | None]],
    split_b: dict[int, dict[str, np.ndarray | None]],
    readout: str,
) -> dict:
    layers = [layer for layer in sorted(split_a) if split_a[layer][readout] is not None]
    means_a = np.asarray([split_a[layer][readout].mean() for layer in layers])
    means_b = np.asarray([split_b[layer][readout].mean() for layer in layers])
    overlaps = {}
    for k in (3, 5, 7):
        effective_k = min(k, len(layers))
        top_a = {layers[index] for index in np.argsort(-means_a)[:effective_k]}
        top_b = {layers[index] for index in np.argsort(-means_b)[:effective_k]}
        overlaps[str(k)] = {
            "count": len(top_a & top_b),
            "denominator": effective_k,
            "split_a": sorted(top_a),
            "split_b": sorted(top_b),
        }
    differences = np.abs(means_a - means_b)
    return {
        "layers": layers,
        "spearman_rho": float(spearmanr(means_a, means_b).statistic),
        "kendall_tau": float(kendalltau(means_a, means_b).statistic),
        "top_k_overlap": overlaps,
        "sign_stable_count": int((np.sign(means_a) == np.sign(means_b)).sum()),
        "sign_stable_denominator": len(layers),
        "mean_absolute_difference": float(differences.mean()),
        "max_absolute_difference": float(differences.max()),
    }

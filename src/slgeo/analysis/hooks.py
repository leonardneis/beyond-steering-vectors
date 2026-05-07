"""Disabled-by-default hooks for future heavy geometry logging."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_GEOMETRY_HOOKS = {
    "delta_weight_extraction": False,
    "svd_pca": False,
    "activation_capture": False,
    "hidden_state_logging": False,
    "cka_analysis": False,
}


def run_optional_geometry_hooks(
    *,
    enabled_hooks: dict[str, bool] | None = None,
    run_dir: str | Path | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Placeholder entrypoint for future expensive geometry analysis.

    All hooks are disabled by default so normal experiments never dump activations,
    hidden states, or tensors unless explicitly wired in later.
    """
    hooks = dict(DEFAULT_GEOMETRY_HOOKS)
    hooks.update(enabled_hooks or {})
    enabled = [name for name, active in hooks.items() if active]
    return {
        "run_dir": str(run_dir) if run_dir else None,
        "enabled_hooks": enabled,
        "context_keys": sorted((context or {}).keys()),
        "status": "disabled" if not enabled else "not_implemented",
    }

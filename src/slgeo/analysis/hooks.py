"""Disabled-by-default hooks for future heavy geometry logging."""

from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_GEOMETRY_HOOKS = {
    "lora_update_reconstruction": False,
    "teacher_student_vector_extraction": False,
    "module_ablation": False,
    "svd_pca": False,
    "cka_analysis": False,
}


def run_optional_geometry_hooks(
    *,
    enabled_hooks: dict[str, bool] | None = None,
    run_dir: str | Path | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe explicitly requested geometry analyses for experiment metadata.

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
        "status": "disabled" if not enabled else "requested",
    }

"""General utilities shared across command-line scripts."""

from __future__ import annotations

import random
import time
from typing import Any


def set_seed(seed: int | None) -> None:
    """Set common random seeds when the relevant packages are available."""
    if seed is None:
        return

    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def timestamp() -> str:
    """Return a compact timestamp for metadata."""
    return time.strftime("%Y%m%d-%H%M%S")


def as_list(value: Any) -> list[Any]:
    """Normalize a scalar or sequence-like value to a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


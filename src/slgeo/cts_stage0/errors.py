"""Error classes and exit-code policy (engineering requirement E2; spec ``accounting.scientific.shard_retry``).

``FinalFailure`` marks every integrity-class or refusal error of this package: identity, provenance, frozen
input or contract hash, guard, render, hook, cache, scoring, non-finite value, statistics, sentinel, fragility,
D/C leak, failed preflight. It maps to ``FINAL_EXIT_CODE`` and is never retried.

Retryable (infrastructure) outcomes are SIGTERM (``SIGTERM_EXIT_CODE``), an unavailable GPU
(``GPU_RESOURCE_EXIT_CODE``), and ``INFRASTRUCTURE_EXIT_CODE`` for out-of-memory and operating-system I/O
errors. Any other exception is a software defect, which a retry cannot fix: it is final as well.
"""

from __future__ import annotations

FINAL_EXIT_CODE = 86
GPU_RESOURCE_EXIT_CODE = 85
SIGTERM_EXIT_CODE = 75
INFRASTRUCTURE_EXIT_CODE = 1


class FinalFailure(Exception):
    """Marker base class: an error that must end the attempt and the shard (never retried)."""

    event = "integrity"


GPU_UNAVAILABLE_PATTERNS = (
    "CUDA-capable device(s) is/are busy or unavailable",
    "all CUDA-capable devices are busy or unavailable",
    "CUDA driver initialization failed",
    "No CUDA GPUs are available",
)


def classify(exc: BaseException) -> tuple[str, int]:
    """(event class, exit code) of an exception raised inside a stage."""
    if isinstance(exc, SystemExit) and exc.code == SIGTERM_EXIT_CODE:
        return "sigterm", SIGTERM_EXIT_CODE
    if not isinstance(exc, FinalFailure) and any(pattern in str(exc) for pattern in GPU_UNAVAILABLE_PATTERNS):
        return "infrastructure", GPU_RESOURCE_EXIT_CODE
    if isinstance(exc, FinalFailure):
        return getattr(exc, "event", "integrity"), FINAL_EXIT_CODE
    try:
        import torch

        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return "infrastructure", INFRASTRUCTURE_EXIT_CODE
    except ImportError:  # pragma: no cover - submit host
        pass
    if isinstance(exc, (MemoryError, OSError)):
        return "infrastructure", INFRASTRUCTURE_EXIT_CODE
    return "software", FINAL_EXIT_CODE

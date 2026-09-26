"""Fail-closed input-path guards: student artifacts, adapters, C18 and D/C authoring material.

Every file the CTS pipeline opens goes through ``guard_input``: it must lie under an allowed root and must
not match any denied pattern. The frozen teacher vector is admitted only by exact path plus SHA-256.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

from .package import sha256_path

FROZEN_V_TEACHER_SHA256 = "4ec9c11ef4c6b8753c388e92c1f18faa6f4364143c64c2583279deb1beb5bc71"
FROZEN_V_TEACHER_RELATIVE = "results/geometry/vectors/cat_subliminal_seed1/v_teacher.pt"

DENIED_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"adapter_(model|config)",
        r"student_lora",
        r"(^|/)checkpoint-\d+",
        r"(^|/)epoch_\d+",
        r"v_student",
        r"(^|/)alignment\.json$",
        r"(^|/)seed_\d+(/|$)",
        r"reference_reproduction_4080",
        r"results/confirmatory",
        r"results/geometry/attribution",
        r"lora_updates",
        r"bidirectional",
        r"(^|[/_.-])c18([/_.-]|$)",
        r"research/cts_stage0_v1/authoring(/|$)",
    )
)


class GuardError(PermissionError):
    pass


def denied(path: str | Path) -> str | None:
    text = Path(path).resolve().as_posix()
    for pattern in DENIED_PATTERNS:
        if pattern.search(text):
            return pattern.pattern
    return None


def guard_input(path: str | Path, allowed_roots: Iterable[str | Path]) -> Path:
    resolved = Path(path).resolve()
    hit = denied(resolved)
    if hit is not None:
        raise GuardError(f"Refusing to open {resolved} (denied pattern {hit!r})")
    roots = [Path(root).resolve() for root in allowed_roots]
    if not any(resolved == root or root in resolved.parents for root in roots):
        raise GuardError(f"Refusing to open {resolved}: outside the allowed input roots")
    return resolved


def assert_no_peft() -> None:
    if "peft" in sys.modules:
        raise GuardError("peft is imported; Stage 0 must never load an adapter")


def load_frozen_teacher(path: str | Path):
    """The frozen t_cat (descriptive continuity cosine only): exact file, pinned SHA, weights_only load."""
    import torch

    path = Path(path).resolve()
    if path.name != "v_teacher.pt" or not path.as_posix().endswith("frozen_t_cat/v_teacher.pt"):
        raise GuardError("The frozen teacher vector is read only from the staged frozen_t_cat/v_teacher.pt")
    if sha256_path(path) != FROZEN_V_TEACHER_SHA256:
        raise GuardError("Frozen teacher vector hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    raw = payload["raw"] if isinstance(payload, dict) else payload
    if tuple(raw.shape) != (29, 3584):
        raise GuardError(f"Frozen teacher vector has shape {tuple(raw.shape)}")
    return raw.double().numpy()

"""Versioned storage and metadata helpers for activation-vector artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


VECTOR_FORMAT_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_half_cosine(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """Layerwise split-half reliability for independently extracted vectors."""
    if first.shape != second.shape:
        raise ValueError("Split-half vectors must have the same shape")
    return torch.nn.functional.cosine_similarity(first.float(), second.float(), dim=-1, eps=1e-12)


def save_vector_artifact(
    path: str | Path,
    *,
    raw: torch.Tensor,
    metadata: dict[str, Any],
    reliability: torch.Tensor | None = None,
) -> Path:
    """Save a CPU vector with explicit format and layer-index semantics."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = raw.detach().float().cpu()
    norm = torch.linalg.vector_norm(raw, dim=-1)
    payload = {
        "format_version": VECTOR_FORMAT_VERSION,
        "raw": raw,
        "unit": raw / norm.unsqueeze(-1).clamp_min(1e-12),
        "norm": norm,
        "reliability": reliability.detach().float().cpu() if reliability is not None else None,
        "metadata": {
            "hidden_state_indexing": "slot 0=embedding; slot i+1=transformer block i",
            **metadata,
        },
    }
    torch.save(payload, path)
    manifest = {
        "format_version": VECTOR_FORMAT_VERSION,
        "tensor_path": path.name,
        "tensor_sha256": sha256_file(path),
        "shape": list(raw.shape),
        "metadata": payload["metadata"],
    }
    path.with_suffix(path.suffix + ".json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return path


def load_vector_artifact(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if "format_version" not in payload and {"raw", "unit", "norm"} <= set(payload):
        payload = {
            "format_version": 0,
            "raw": payload["raw"],
            "unit": payload["unit"],
            "norm": payload["norm"],
            "reliability": None,
            "metadata": payload.get("meta", {}),
        }
    required = {"format_version", "raw", "unit", "norm", "metadata"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Vector artifact missing keys: {sorted(missing)}")
    return payload

"""Common random numbers for sampled generation: one generator seed per (stem, prefix, sample index),
identical across all arms, so paired contrasts share sampling noise by design."""

from __future__ import annotations

import hashlib


def sample_seed(stem_id: str, prefix_id: str, k: int) -> int:
    digest = hashlib.sha256(f"phenotype-anchor-v1|{stem_id}|{prefix_id}|{k}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & (2**63 - 1)

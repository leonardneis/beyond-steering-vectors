"""Batched ancestral sampling with one generator per row (common random numbers across arms).

HF ``generate`` cannot give each row of a batch its own seed. Here every row owns a ``torch.Generator`` seeded by
``crn.sample_seed(stem, prefix, k)``; each step draws the row's token with ``torch.multinomial`` from its own
generator. T = 1, top_p = 1 (the full softmax), so the sampled first answer estimates the exact answer
probability. Batches are formed in a fixed order that does not depend on the arm, so every arm sees identical
padding and identical random streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

MAX_NEW_TOKENS = 24


@dataclass(frozen=True)
class SampleRequest:
    input_ids: tuple[int, ...]
    seed: int


@torch.inference_mode()
def sample_batch(
    model, requests: Sequence[SampleRequest], *, eos_ids: Sequence[int], pad_id: int,
    max_new_tokens: int = MAX_NEW_TOKENS, device=None,
) -> list[list[int]]:
    """Left-padded batch prefill, then step-wise sampling; returns the generated ids of every row (EOS excluded,
    generation stops per row at the first EOS)."""
    if not requests:
        return []
    device = device or next(model.parameters()).device
    width = max(len(r.input_ids) for r in requests)
    ids = torch.full((len(requests), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for i, r in enumerate(requests):
        ids[i, width - len(r.input_ids):] = torch.tensor(r.input_ids, device=device)
        mask[i, width - len(r.input_ids):] = 1
    generators = [torch.Generator(device=device).manual_seed(int(r.seed)) for r in requests]
    eos = set(int(e) for e in eos_ids)
    positions = (mask.cumsum(dim=1) - 1).clamp(min=0)
    out = model(input_ids=ids, attention_mask=mask, position_ids=positions, use_cache=True)
    cache = out.past_key_values
    logits = out.logits[:, -1, :]
    done = [False] * len(requests)
    tokens: list[list[int]] = [[] for _ in requests]
    next_pos = positions[:, -1] + 1
    for _step in range(max_new_tokens):
        probs = torch.softmax(logits.float(), dim=-1)
        chosen = torch.empty(len(requests), dtype=torch.long, device=device)
        for i, g in enumerate(generators):
            chosen[i] = torch.multinomial(probs[i], 1, generator=g)[0]
        for i in range(len(requests)):
            if done[i]:
                continue
            token = int(chosen[i])
            if token in eos:
                done[i] = True
            else:
                tokens[i].append(token)
        if all(done):
            break
        feed = torch.where(torch.tensor(done, device=device), torch.full_like(chosen, pad_id), chosen)
        mask = torch.cat([mask, torch.ones((len(requests), 1), dtype=mask.dtype, device=device)], dim=1)
        out = model(input_ids=feed[:, None], attention_mask=mask, position_ids=next_pos[:, None],
                    past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits = out.logits[:, -1, :]
        next_pos = next_pos + 1
    return tokens

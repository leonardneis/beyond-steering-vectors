"""Activation-vector extraction and teacher/student alignment metrics."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


def difference_vector(mean_a: torch.Tensor, mean_b: torch.Tensor) -> dict[str, torch.Tensor]:
    """Create raw, unit-normalized and norm views of layerwise mean differences."""
    if mean_a.shape != mean_b.shape:
        raise ValueError(f"Activation shapes differ: {tuple(mean_a.shape)} vs {tuple(mean_b.shape)}")
    raw = mean_a.float() - mean_b.float()
    norm = torch.linalg.vector_norm(raw, dim=-1)
    unit = raw / norm.unsqueeze(-1).clamp_min(1e-12)
    return {"raw": raw, "unit": unit, "norm": norm}


def alignment_metrics(student: torch.Tensor, teacher: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute layerwise cosine and signed projection onto the teacher direction."""
    if student.shape != teacher.shape:
        raise ValueError(f"Vector shapes differ: {tuple(student.shape)} vs {tuple(teacher.shape)}")
    student = student.float()
    teacher = teacher.float()
    teacher_norm = torch.linalg.vector_norm(teacher, dim=-1).clamp_min(1e-12)
    cosine = F.cosine_similarity(student, teacher, dim=-1, eps=1e-12)
    signed_projection = (student * teacher).sum(dim=-1) / teacher_norm
    projection_fraction = signed_projection / torch.linalg.vector_norm(student, dim=-1).clamp_min(1e-12)
    return {
        "cosine": cosine,
        "signed_projection": signed_projection,
        "projection_fraction": projection_fraction,
    }


@torch.no_grad()
def mean_hidden_states(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    system_prompt: str | None = None,
    batch_size: int = 4,
    position: str = "last",
) -> torch.Tensor:
    """Return mean hidden state per model layer, including the embedding slot.

    ``position='last'`` extracts the assistant-tag position. ``position='all'``
    averages all non-padding prompt tokens. Left padding is required so the final
    token has the same index in every batch element.
    """
    if position not in {"last", "all"}:
        raise ValueError("position must be 'last' or 'all'")
    if getattr(tokenizer, "padding_side", None) != "left":
        raise ValueError("mean_hidden_states requires tokenizer.padding_side='left'")
    if not prompts:
        raise ValueError("At least one prompt is required")

    rendered = []
    for prompt in prompts:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        rendered.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    device = next(model.parameters()).device
    running = None
    denominator = 0
    for start in range(0, len(rendered), batch_size):
        encoded = tokenizer(
            rendered[start : start + batch_size], return_tensors="pt", padding=True
        ).to(device)
        output = model(**encoded, output_hidden_states=True, use_cache=False)
        if position == "last":
            values = torch.stack([hidden[:, -1].float().cpu() for hidden in output.hidden_states])
            batch_sum = values.sum(dim=1)
            count = values.shape[1]
        else:
            mask = encoded.attention_mask.float().cpu().unsqueeze(-1)
            batch_sum = torch.stack(
                [(hidden.float().cpu() * mask).sum(dim=(0, 1)) for hidden in output.hidden_states]
            )
            count = int(encoded.attention_mask.sum().item())
        running = batch_sum if running is None else running + batch_sum
        denominator += count
    return running / denominator

"""Activation-vector extraction and teacher/student alignment metrics."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F


@torch.no_grad()
def hidden_state_statistics(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    directions: torch.Tensor | None = None,
    system_prompt: str | None = None,
    batch_size: int = 4,
    position: str = "last",
) -> dict[str, torch.Tensor | None]:
    """Compute aggregate hidden means and optional per-prompt projections.

    The aggregate mean preserves the reference extraction convention: with
    ``position='all'`` every non-padding token receives equal weight. Per-prompt
    projections first average within each prompt, so prompts are the independent
    statistical units regardless of their token length.
    """
    if position not in {"last", "all"}:
        raise ValueError("position must be 'last' or 'all'")
    if getattr(tokenizer, "padding_side", None) != "left":
        raise ValueError("hidden-state extraction requires tokenizer.padding_side='left'")
    if not prompts:
        raise ValueError("At least one prompt is required")
    if directions is not None and directions.ndim != 2:
        raise ValueError("directions must have shape [hidden_state_slots, hidden_size]")

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
    projection_batches = []
    unit_directions = None
    if directions is not None:
        directions = directions.detach().float().cpu()
        norms = torch.linalg.vector_norm(directions, dim=-1, keepdim=True).clamp_min(1e-12)
        unit_directions = directions / norms

    for start in range(0, len(rendered), batch_size):
        encoded = tokenizer(
            rendered[start : start + batch_size], return_tensors="pt", padding=True
        ).to(device)
        output = model(**encoded, output_hidden_states=True, use_cache=False)
        if directions is not None and len(output.hidden_states) != directions.shape[0]:
            raise ValueError(
                f"Direction slots {directions.shape[0]} do not match hidden states "
                f"{len(output.hidden_states)}"
            )

        if position == "last":
            per_prompt = torch.stack(
                [hidden[:, -1].float().cpu() for hidden in output.hidden_states], dim=1
            )  # [batch, slots, hidden]
            batch_sum = per_prompt.sum(dim=0)
            count = per_prompt.shape[0]
        else:
            mask = encoded.attention_mask.float().cpu().unsqueeze(-1)
            token_counts = mask.sum(dim=1).clamp_min(1)
            per_prompt = torch.stack(
                [
                    (hidden.float().cpu() * mask).sum(dim=1) / token_counts
                    for hidden in output.hidden_states
                ],
                dim=1,
            )
            batch_sum = torch.stack(
                [
                    (hidden.float().cpu() * mask).sum(dim=(0, 1))
                    for hidden in output.hidden_states
                ]
            )
            count = int(encoded.attention_mask.sum().item())

        running = batch_sum if running is None else running + batch_sum
        denominator += count
        if unit_directions is not None:
            projection_batches.append(
                torch.einsum("blh,lh->bl", per_prompt, unit_directions)
            )

    projections = torch.cat(projection_batches, dim=0) if projection_batches else None
    return {"mean": running / denominator, "projections": projections}


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
    result = hidden_state_statistics(
        model,
        tokenizer,
        prompts,
        system_prompt=system_prompt,
        batch_size=batch_size,
        position=position,
    )
    return result["mean"]

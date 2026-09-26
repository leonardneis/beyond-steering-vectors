"""Persona-state extraction on the number-continuation prompts (PREREGISTRATION §4).

Batch size 1, no padding, last prompt token (id 198). Slot k = hidden_states[k] = output of block k-1;
slot 28 is the post-final-norm state. Stored per persona: fp16 last-token states at slots
{14, 27, 28} for all 1,024 rows, and float64 sums of the last-token state per half (rows 0-511,
512-1023) at all 29 slots, accumulated in extraction-row order.
"""

from __future__ import annotations

from .errors import FinalFailure

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .render import Renderer, assert_prefill_ids
from .steering import assert_no_hooks

STORED_SLOTS = (14, 27, 28)  # v2 spec extraction.stored_per_prompt_slots
N_SLOTS = 29
N_ROWS = 1024
HALF = 512


class ExtractionError(RuntimeError, FinalFailure):
    """Raised when extraction would deviate from the frozen estimator."""


@dataclass
class PersonaStates:
    persona_id: str
    states: np.ndarray  # [1024, len(STORED_SLOTS), H], float16 for the NF4/fp16 model
    half_sums: np.ndarray  # [2, 29, H] float64
    prompt_lens: np.ndarray  # [1024] int64
    rendered_sha256: list[str]


@torch.inference_mode()
def last_token_hidden_states(model, input_ids: Sequence[int], device) -> tuple[torch.Tensor, ...]:
    """All 29 hidden-state slots at the last prompt position for one unpadded prompt."""
    assert_no_hooks(model)
    prompt_len = assert_prefill_ids(input_ids)
    ids = torch.tensor([list(input_ids)], device=device)
    output = model.model(input_ids=ids, output_hidden_states=True, use_cache=False)
    hidden_states = output.hidden_states
    if len(hidden_states) != N_SLOTS:
        raise ExtractionError(f"Expected {N_SLOTS} hidden-state slots, got {len(hidden_states)}")
    for slot in hidden_states:
        if slot.shape[0] != 1 or slot.shape[1] != prompt_len:
            raise ExtractionError("Extraction forward is not batch 1 / unpadded")
    return tuple(slot[0, prompt_len - 1] for slot in hidden_states)


def extract_persona(model, renderer: Renderer, persona_id: str, prompts: Sequence[str], device) -> PersonaStates:
    if len(prompts) != N_ROWS:
        raise ExtractionError(f"Extraction requires exactly {N_ROWS} prompts")
    hidden_size = model.config.hidden_size
    state_dtype = {torch.float16: np.float16, torch.float32: np.float32}.get(model.dtype)
    if state_dtype is None:
        raise ExtractionError(f"Unsupported model dtype {model.dtype}")
    states = np.empty((N_ROWS, len(STORED_SLOTS), hidden_size), dtype=state_dtype)
    sums = torch.zeros(2, N_SLOTS, hidden_size, dtype=torch.float64, device=device)
    prompt_lens = np.empty(N_ROWS, dtype=np.int64)
    rendered_sha: list[str] = []
    for row, prompt in enumerate(prompts):
        rendered = renderer.render(persona_id, prompt)
        slots = last_token_hidden_states(model, rendered.input_ids, device)
        if slots[0].dtype != model.dtype:
            raise ExtractionError(f"Hidden states must be in the model compute dtype, got {slots[0].dtype}")
        stacked = torch.stack(slots)  # [29, H] in the compute dtype (float16 for the NF4 model)
        if not torch.isfinite(stacked).all():
            raise ExtractionError("Non-finite hidden state")
        sums[row // HALF] += stacked.double()
        states[row] = stacked[list(STORED_SLOTS)].cpu().numpy()
        prompt_lens[row] = rendered.prompt_len
        rendered_sha.append(rendered.text_sha256)
    return PersonaStates(persona_id, states, sums.cpu().numpy(), prompt_lens, rendered_sha)

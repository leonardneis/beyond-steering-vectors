"""Sampled answers (descriptive only; PREREGISTRATION §7.3).

Explicit sampling parameters (temperature 1, top_k 0, top_p 1, repetition penalty 1) are implemented as
plain multinomial sampling from the full float32 softmax, so no generation_config default can leak. Steering
acts in the prefill only; the 8 decode steps run without hooks. Randomness comes from a CPU generator
seeded per (condition, prompt), independent of shard order and batch composition.
"""

from __future__ import annotations

import hashlib
import re
from typing import Mapping, Sequence

import torch

from .render import RenderedPrompt, assert_prefill_ids
from .steering import PrefillSteering, RowSteer, assert_no_hooks

SAMPLES_PER_PROMPT = 10
MAX_NEW_TOKENS = 8
EOS_IDS = (151645, 151643)
SAMPLING_PARAMETERS = {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0}
_WORD = re.compile(r"[A-Za-z]+")


def sampling_seed(condition_id: str, prompt_id: str) -> int:
    return int(hashlib.sha256(f"cts-s0-sample|{condition_id}|{prompt_id}".encode()).hexdigest()[:16], 16)


@torch.inference_mode()
def sample_answers(model, rendered: RenderedPrompt, row: RowSteer, *, weight32, seed: int) -> list[list[int]]:
    from transformers.cache_utils import DynamicCache

    assert_no_hooks(model)
    device = weight32.device
    prompt_len = assert_prefill_ids(rendered.input_ids)
    with PrefillSteering(model, prompt_len, [row]):
        prefill = model.model(input_ids=torch.tensor([rendered.input_ids], device=device), past_key_values=DynamicCache(), use_cache=True)
    assert_no_hooks(model)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cache = DynamicCache()
    tensors = prefill.past_key_values
    pairs = list(zip(tensors.key_cache, tensors.value_cache)) if hasattr(tensors, "key_cache") else [
        (layer.keys, layer.values) for layer in tensors.layers
    ]
    for layer, (keys, values) in enumerate(pairs):
        cache.update(keys.expand(SAMPLES_PER_PROMPT, -1, -1, -1).contiguous(), values.expand(SAMPLES_PER_PROMPT, -1, -1, -1).contiguous(), layer)
    logits = torch.nn.functional.linear(prefill.last_hidden_state[:, -1].float(), weight32).expand(SAMPLES_PER_PROMPT, -1)
    tokens: list[list[int]] = [[] for _ in range(SAMPLES_PER_PROMPT)]
    done = [False] * SAMPLES_PER_PROMPT
    for step in range(MAX_NEW_TOKENS):
        probabilities = torch.softmax(logits.float(), dim=-1).cpu()
        drawn = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
        for index, token in enumerate(drawn.tolist()):
            if not done[index]:
                tokens[index].append(token)
                if token in EOS_IDS:
                    done[index] = True
        if all(done) or step == MAX_NEW_TOKENS - 1:
            break
        position = torch.full((SAMPLES_PER_PROMPT, 1), prompt_len + step, device=device)
        output = model.model(input_ids=drawn.view(-1, 1).to(device), position_ids=position, past_key_values=cache, use_cache=True)
        cache = output.past_key_values
        logits = torch.nn.functional.linear(output.last_hidden_state[:, -1].float(), weight32)
    return tokens


def parse_answer(text: str, lexicon: Mapping[str, Sequence[str]], chinese: Mapping[str, Sequence[str]]) -> str | None:
    """First whole-word, case-insensitive lexicon match (singular key); else longest Chinese match."""
    forms = {form.lower(): key for key, values in lexicon.items() for form in values}
    for match in _WORD.finditer(text):
        key = forms.get(match.group(0).lower())
        if key is not None:
            return key
    best = None
    for key, values in chinese.items():
        for form in values:
            position = text.find(form)
            if position >= 0 and (best is None or (position, -len(form)) < (best[0], -len(best[1]))):
                best = (position, form, key)
    return f"zh:{best[2]}" if best else None

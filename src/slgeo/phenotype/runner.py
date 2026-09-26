"""Per-context scoring, state capture and sampling for the Phenotype Anchor (no orchestration, no I/O policy).

One *context* = (stem, cell, arm). Its rendered prompt is scored once by the unmodified CTS L1 scorer
(``cts_stage0.scoring.score_prompt`` with a single unsteered row): exact word log-probabilities over the frozen
panel, CJK mass, and the coverage diagnostic (first-position mass of decoration tokens and of space-led emoji that
the exact endpoint cannot score). Selected contexts additionally store the last-prompt-token hidden states of all
29 slots (fp16) for P3/P4, and the primary cell is sampled with common random numbers.

Full-vocabulary log-probabilities are never returned or stored.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

from slgeo.cts_stage0.render import RenderedPrompt
from slgeo.cts_stage0.scoring import FormTable, score_prompt
from slgeo.cts_stage0.steering import RowSteer

from . import crn, parser
from .sampler import SampleRequest, sample_batch

DECORATION_TEXTS = ("*", "**", '"', "“", "-", "#", "`", "_", ">")


@dataclass(frozen=True)
class Context:
    context_id: str
    stem_id: str
    cell: str  # e.g. "Q+prefix", "Q", "H+prefix", "H", "persona+prefix", "persona"
    prefix_id: str  # "none" or the replicate id
    persona_id: str
    user_prompt: str


@dataclass
class ScoredContexts:
    context_ids: list[str]
    word_logp: np.ndarray  # [n, W] exact log p_w over the panel
    cjk_mass: np.ndarray  # [n]
    decoration_mass: np.ndarray  # [n]
    emoji_mass: np.ndarray  # [n]
    top1: np.ndarray  # [n] token id of the most probable first answer token
    prompt_len: np.ndarray  # [n]
    states: np.ndarray | None  # [n_capture, 29, H] fp16, rows aligned with ``captured_ids``
    captured_ids: list[str]


def decoration_ids(encode: Callable[[str], Sequence[int]]) -> list[int]:
    """First-answer tokens that start a decorated answer ('**Cat**', '"Cat"', '- Cat', ...)."""
    out = set()
    for text in DECORATION_TEXTS:
        ids = list(encode(text))
        if len(ids) == 1:
            out.add(int(ids[0]))
    return sorted(out)


def space_emoji_ids(decode: Callable[[list[int]], str], vocab_size: int, boundary_ids: Sequence[int]) -> list[int]:
    """Vocabulary ids whose decode is whitespace followed by a symbol (Unicode category So) and that are not
    boundary ids: an answer word followed by such a token scores zero in the exact endpoint (pre-freeze audit)."""
    boundary = set(int(b) for b in boundary_ids)
    out = []
    for token in range(vocab_size):
        if token in boundary:
            continue
        text = decode([token])
        if len(text) >= 2 and text[0].isspace() and unicodedata.category(text.lstrip()[0]) == "So":
            out.append(token)
    return out


@torch.inference_mode()
def last_token_states(model, input_ids: Sequence[int]) -> np.ndarray:
    """[29, H] hidden states (embedding output + every block output, the last one post final norm) at the last
    prompt position, fp16."""
    device = next(model.parameters()).device
    out = model.model(input_ids=torch.tensor([list(input_ids)], device=device), output_hidden_states=True)
    return torch.stack([h[0, -1, :] for h in out.hidden_states]).to(torch.float16).cpu().numpy()


def score_contexts(
    model, rendered: Sequence[RenderedPrompt], contexts: Sequence[Context], table: FormTable, *,
    weight32: torch.Tensor, cjk_ids: torch.Tensor, decoration: Sequence[int], emoji: Sequence[int],
    capture: Callable[[Context], bool] = lambda c: False, expected_suffix=None,
) -> ScoredContexts:
    """Score every context at batch 1 with one unsteered row (L1 layout)."""
    if len(rendered) != len(contexts):
        raise ValueError("rendered and contexts differ in length")
    kw = {} if expected_suffix is None else {"expected_suffix": expected_suffix}
    dec = torch.tensor(list(decoration), dtype=torch.long) if decoration else None
    emo = torch.tensor(list(emoji), dtype=torch.long) if emoji else None
    word_logp, cjk, dmass, emass, top1, lens, states, captured = [], [], [], [], [], [], [], []
    for r, c in zip(rendered, contexts):
        result = score_prompt(model, r, [RowSteer({})], table, weight32=weight32, cjk_ids=cjk_ids,
                              return_first_logprobs=True, **kw)
        first = result.first_logprobs[0]
        word_logp.append(result.word_logp[0])
        cjk.append(float(result.cjk_mass[0]))
        dmass.append(float(torch.logsumexp(first[dec], 0).exp()) if dec is not None else 0.0)
        emass.append(float(torch.logsumexp(first[emo], 0).exp()) if emo is not None else 0.0)
        top1.append(int(result.top1[0]))
        lens.append(r.prompt_len)
        if capture(c):
            states.append(last_token_states(model, r.input_ids))
            captured.append(c.context_id)
    return ScoredContexts(
        [c.context_id for c in contexts], np.stack(word_logp), np.array(cjk), np.array(dmass), np.array(emass),
        np.array(top1), np.array(lens), np.stack(states) if states else None, captured,
    )


@dataclass(frozen=True)
class Sample:
    context_id: str
    k: int
    seed: int
    text: str
    lemma: str
    cls: str
    extra_text: bool


def sample_contexts(
    model, rendered: Sequence[RenderedPrompt], contexts: Sequence[Context], *, k: int, batch_size: int,
    decode: Callable[[list[int]], str], eos_ids: Sequence[int], pad_id: int, max_new_tokens: int = 24,
) -> list[Sample]:
    """K samples per context with CRN seeds; batches formed in (context order, k) order, independent of the arm."""
    jobs = [(r, c, j) for r, c in zip(rendered, contexts) for j in range(k)]
    out: list[Sample] = []
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        seeds = [crn.sample_seed(c.stem_id, c.prefix_id, j) for _r, c, j in chunk]
        tokens = sample_batch(model, [SampleRequest(r.input_ids, s) for (r, _c, _j), s in zip(chunk, seeds)],
                              eos_ids=eos_ids, pad_id=pad_id, max_new_tokens=max_new_tokens)
        for (_r, c, j), s, ids in zip(chunk, seeds, tokens):
            text = decode(ids)
            parsed = parser.parse(text)
            out.append(Sample(c.context_id, j, s, text, parsed.lemma, parsed.cls, parsed.extra_text))
    return out

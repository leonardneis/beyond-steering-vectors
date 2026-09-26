"""Word-level sequence scoring L_w by exact teacher forcing from the KV cache (PREREGISTRATION §7.1).

Layout (implementation decision, documented in ``research/cts_stage0_v1_execution``):

1. Prefill: one forward over the rendered prompt for ``B`` rows (``B = 1`` is the strict batch-1 layout;
   ``B > 1`` is condition batching over identical prompt tokens with per-row steering, no padding).
   Steering hooks are active only here.
2. Continuation: one unsteered forward per prefill that appends every answer form as its own segment
   (flat packing): position ids ``L + depth``, a 4D mask under which a segment sees the full prompt cache and
   its own earlier tokens only. For each form f = (t_1..t_k):
   log P(f, then any b in B) = lp_prefill[t_1] + sum_j lp_cont[t_j][t_{j+1}] + logsumexp_{b in B} lp_cont[t_k][b].
3. Logits are recomputed in float32 from the post-norm final state and the unquantized lm_head weight;
   normalization is over all vocabulary rows of lm_head (152,064).

``score_sequential`` is the per-form reference implementation used by the equivalence tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from .package import LAST_THREE_PROMPT_IDS
from .render import RenderedPrompt, assert_prefill_ids
from .steering import PrefillSteering, RowSteer, assert_no_hooks


_ROW_CHUNK = 8


class ScoringError(RuntimeError):
    """Raised when a scoring invariant fails (cache length, finiteness, form table)."""


@dataclass(frozen=True)
class FormTable:
    """Answer forms of the scored words, in frozen order, flattened into packed segments."""

    words: tuple[str, ...]
    form_word: tuple[int, ...]  # word index of each form
    form_text: tuple[str, ...]
    form_ids: tuple[tuple[int, ...], ...]
    boundary_ids: tuple[int, ...]

    @classmethod
    def from_endpoint(cls, endpoint: dict, words: Sequence[str] | None = None, boundary_ids=None) -> "FormTable":
        words = tuple(words if words is not None else endpoint["scoring_words"] + endpoint["non_animal_words"])
        form_word, form_text, form_ids = [], [], []
        for index, word in enumerate(words):
            forms = endpoint["answer_forms"][word]["forms"]
            if not forms:
                raise ScoringError(f"No answer forms for {word!r}")
            for form in forms:
                ids = tuple(int(token) for token in form["token_ids"])
                if not ids:
                    raise ScoringError(f"Empty answer form for {word!r}")
                form_word.append(index)
                form_text.append(form["text"])
                form_ids.append(ids)
        if len(set(zip(form_word, form_ids))) != len(form_ids):
            raise ScoringError("Duplicate answer form")
        boundary = tuple(int(token) for token in (boundary_ids if boundary_ids is not None else endpoint["boundary_ids"]))
        return cls(words, tuple(form_word), tuple(form_text), tuple(form_ids), boundary)

    @property
    def n_forms(self) -> int:
        return len(self.form_ids)

    def gather_plan(self):
        """Index lists for vectorized scoring: first tokens, transitions (packed position, next token,
        form index) and the packed position of each form's last token."""
        _tokens, _depths, _segments, spans = self.packed()
        first, trans_pos, trans_tok, trans_form, last = [], [], [], [], []
        for index, ((start, end), ids) in enumerate(zip(spans, self.form_ids)):
            first.append(ids[0])
            for offset in range(end - start - 1):
                trans_pos.append(start + offset)
                trans_tok.append(ids[offset + 1])
                trans_form.append(index)
            last.append(end - 1)
        return first, trans_pos, trans_tok, trans_form, last

    def packed(self):
        """Packed continuation tokens, depths, segment ids and per-form packed index ranges."""
        tokens, depths, segments, spans = [], [], [], []
        for segment, ids in enumerate(self.form_ids):
            start = len(tokens)
            for depth, token in enumerate(ids):
                tokens.append(token)
                depths.append(depth)
                segments.append(segment)
            spans.append((start, len(tokens)))
        return tokens, depths, segments, spans


@dataclass
class ScoreResult:
    """Per-row outputs of one prompt-condition batch (float64 on CPU)."""

    form_logp: np.ndarray  # [B, n_forms]
    word_logp: np.ndarray  # [B, n_words]  (L_w)
    first_logprobs: torch.Tensor | None  # [B, V] float32 on CPU when requested
    cjk_mass: np.ndarray  # [B]
    top1: np.ndarray  # [B]
    kl_to_reference: np.ndarray  # [B] KL(row || reference) at the first answer position (NaN if no reference)


def _final_norm_logits(model, hidden: torch.Tensor, weight32: torch.Tensor) -> torch.Tensor:
    """float32 logits from post-norm states: F.linear(h.float(), W_U.float()) (lm_head has no bias)."""
    if weight32.dtype != torch.float32:
        raise ScoringError("The logit weight must be float32")
    logits = torch.nn.functional.linear(hidden.float(), weight32)
    if logits.dtype != torch.float32:
        raise ScoringError("Logits must be float32 before log_softmax")
    return logits


def _float32(logits: torch.Tensor) -> torch.Tensor:
    """Call-site guard: every log_softmax/logsumexp in this module runs on float32 logits."""
    if logits.dtype != torch.float32:
        raise ScoringError(f"Logits must be float32, got {logits.dtype}")
    return logits


def lm_head_weight32(model) -> torch.Tensor:
    head = model.lm_head
    if type(head) is not torch.nn.Linear or head.bias is not None:
        raise ScoringError("lm_head must be an unquantized nn.Linear without bias")
    return head.weight.detach().float()


def _packed_mask(prompt_len: int, depths, segments, rows: int, dtype, device) -> torch.Tensor:
    t = len(depths)
    seg = torch.tensor(segments, device=device)
    dep = torch.tensor(depths, device=device)
    allowed = (seg[:, None] == seg[None, :]) & (dep[None, :] <= dep[:, None])
    full = torch.ones(t, prompt_len + t, dtype=torch.bool, device=device)
    full[:, prompt_len:] = allowed
    mask = torch.zeros(t, prompt_len + t, dtype=dtype, device=device)
    mask.masked_fill_(~full, torch.finfo(dtype).min)
    return mask[None, None].expand(rows, 1, t, prompt_len + t)


def cache_length(cache) -> int:
    return int(cache.get_seq_length())


@torch.inference_mode()
def score_prompt(
    model,
    rendered: RenderedPrompt,
    rows: Sequence[RowSteer],
    table: FormTable,
    *,
    weight32: torch.Tensor,
    cjk_ids: torch.Tensor | None = None,
    return_first_logprobs: bool = False,
    reference_first_lp: torch.Tensor | None = None,
    expected_suffix: tuple[int, ...] = LAST_THREE_PROMPT_IDS,
) -> ScoreResult:
    """Score every form of ``table`` for each steering row on one rendered prompt."""
    from transformers.cache_utils import DynamicCache

    assert_no_hooks(model)
    device = weight32.device
    prompt_len = assert_prefill_ids(rendered.input_ids, expected_suffix)
    batch = len(rows)
    input_ids = torch.tensor([rendered.input_ids] * batch, device=device)
    cache = DynamicCache()
    with PrefillSteering(model, prompt_len, rows):
        prefill = model.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    assert_no_hooks(model)
    cache = prefill.past_key_values
    if cache_length(cache) != prompt_len:
        raise ScoringError("Prefill cache length differs from prompt_len")
    first_logits = _float32(_final_norm_logits(model, prefill.last_hidden_state[:, -1, :], weight32))  # [B, V]
    first_lse = torch.logsumexp(first_logits, dim=-1)
    first_lp = first_logits - first_lse[:, None]

    tokens, depths, segments, spans = table.packed()
    boundary = torch.tensor(table.boundary_ids, device=device)
    cont_ids = torch.tensor([tokens] * batch, device=device)
    positions = torch.tensor([[prompt_len + depth for depth in depths]] * batch, device=device)
    mask = _packed_mask(prompt_len, depths, segments, batch, model.dtype, device)
    continuation = model.model(
        input_ids=cont_ids, position_ids=positions, attention_mask=mask, past_key_values=cache, use_cache=True
    )
    if cache_length(continuation.past_key_values) != prompt_len + len(tokens):
        raise ScoringError("Continuation cache length is inconsistent")
    hidden = continuation.last_hidden_state  # [B, T, H]

    first_index, trans_pos, trans_tok, trans_form, last_pos = table.gather_plan()
    form_logp = first_lp[:, torch.tensor(first_index, device=device)].double()  # [B, F]
    trans_pos_t = torch.tensor(trans_pos, device=device)
    trans_tok_t = torch.tensor(trans_tok, device=device)
    # Deterministic scatter of transition terms onto forms: a fixed 0/1 float64 matrix (no atomics).
    assign = torch.zeros(len(trans_form), table.n_forms, dtype=torch.float64, device=device)
    if trans_form:
        assign[torch.arange(len(trans_form), device=device), torch.tensor(trans_form, device=device)] = 1.0
    last_pos_t = torch.tensor(last_pos, device=device)
    for start in range(0, batch, _ROW_CHUNK):
        rows_slice = slice(start, min(batch, start + _ROW_CHUNK))
        logits = _float32(_final_norm_logits(model, hidden[rows_slice], weight32))  # [b, T, V]
        lse = torch.logsumexp(logits, dim=-1)  # [b, T]
        boundary_lse = torch.logsumexp(logits.index_select(-1, boundary), dim=-1) - lse  # [b, T]
        chunk = form_logp[rows_slice]
        if trans_pos:
            trans = (logits[:, trans_pos_t, trans_tok_t] - lse[:, trans_pos_t]).double()  # [b, n_trans]
            chunk += trans @ assign
        chunk += boundary_lse[:, last_pos_t].double()
        form_logp[rows_slice] = chunk
        del logits
    form_logp_cpu = form_logp.cpu().numpy()
    word_logp = np.full((batch, len(table.words)), -np.inf)
    form_word = np.asarray(table.form_word)
    for word in range(len(table.words)):
        values = form_logp_cpu[:, form_word == word]
        word_logp[:, word] = np.logaddexp.reduce(values, axis=1)
    if not np.isfinite(form_logp_cpu).all() or not np.isfinite(word_logp).all():
        raise ScoringError("Non-finite sequence score")
    if cjk_ids is not None:
        cjk_mass = torch.exp(torch.logsumexp(first_lp[:, cjk_ids.to(device)], dim=-1)).double().cpu().numpy()
    else:
        cjk_mass = np.full(batch, np.nan)
    top1 = first_lp.argmax(dim=-1).cpu().numpy()
    if reference_first_lp is not None:
        reference = reference_first_lp.to(device=device, dtype=torch.float32)
        kl = (first_lp.exp() * (first_lp - reference[None, :])).sum(dim=-1).double().cpu().numpy()
    else:
        kl = np.full(batch, np.nan)
    return ScoreResult(
        form_logp=form_logp_cpu,
        word_logp=word_logp,
        first_logprobs=first_lp.cpu() if return_first_logprobs else None,
        cjk_mass=cjk_mass,
        top1=top1,
        kl_to_reference=kl,
    )


@torch.inference_mode()
def score_sequential(
    model, rendered: RenderedPrompt, row: RowSteer, table: FormTable, *, weight32,
    expected_suffix: tuple[int, ...] = LAST_THREE_PROMPT_IDS,
) -> np.ndarray:
    """Reference: batch-1 prefill, then each form teacher-forced from its own copy of the prefill cache."""
    from transformers.cache_utils import DynamicCache

    device = weight32.device
    prompt_len = assert_prefill_ids(rendered.input_ids, expected_suffix)
    input_ids = torch.tensor([rendered.input_ids], device=device)
    cache = DynamicCache()
    with PrefillSteering(model, prompt_len, [row]):
        prefill = model.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
    snapshot = [(layer_k, layer_v) for layer_k, layer_v in _cache_tensors(prefill.past_key_values)]
    first = _float32(_final_norm_logits(model, prefill.last_hidden_state[0, -1], weight32))
    first_lp = first - torch.logsumexp(first, dim=-1)
    boundary = torch.tensor(table.boundary_ids, device=device)
    out = np.empty(table.n_forms)
    for index, ids in enumerate(table.form_ids):
        fresh = DynamicCache()
        for layer, (keys, values) in enumerate(snapshot):
            fresh.update(keys, values, layer)
        if cache_length(fresh) != prompt_len:
            raise ScoringError("Fresh cache length differs from prompt_len")
        forward = model.model(
            input_ids=torch.tensor([list(ids)], device=device),
            position_ids=torch.arange(prompt_len, prompt_len + len(ids), device=device)[None],
            past_key_values=fresh,
            use_cache=True,
        )
        logits = _float32(_final_norm_logits(model, forward.last_hidden_state[0], weight32))
        lp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        value = float(first_lp[ids[0]])
        for offset in range(len(ids) - 1):
            value += float(lp[offset, ids[offset + 1]])
        value += float(torch.logsumexp(lp[len(ids) - 1, boundary], dim=-1))
        out[index] = value
    return out


def _cache_tensors(cache):
    """Per-layer (key, value) tensors of a DynamicCache in transformers 4.48 and 5.x."""
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return [(layer.keys, layer.values) for layer in cache.layers]


def word_scores(form_logp: np.ndarray, table: FormTable) -> np.ndarray:
    form_word = np.asarray(table.form_word)
    return np.stack(
        [np.logaddexp.reduce(form_logp[..., form_word == word], axis=-1) for word in range(len(table.words))], axis=-1
    )

"""Outcome-free self-tests of the hook, cache and scoring semantics.

``tiny_model_suite`` runs on a randomly initialized small Qwen2 (CPU, float32) and is executed in the
execution environment (transformers 4.48.3) by the preflight node, as the spec requires ("hook unit tests
run in that environment"). ``real_model_hook_selftest`` runs on the loaded 7B model in every GPU node before
its first scientific forward, on one extraction number prompt in the default context only.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .render import RenderedPrompt
from .scoring import FormTable, lm_head_weight32, score_prompt, score_sequential
from .steering import ALL, LAST, PrefillSteering, RowSteer, SteeringError, decoder_layers

TINY_LAYERS = 16


def tiny_model(seed: int = 0, vocab: int = 320, layers: int = TINY_LAYERS):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        tie_word_embeddings=False, attn_implementation="sdpa",
    )
    model = Qwen2ForCausalLM(config).eval()
    return model


def tiny_prompt(length: int = 24, seed: int = 1, vocab: int = 320) -> RenderedPrompt:
    generator = torch.Generator().manual_seed(seed)
    body = torch.randint(0, vocab - 20, (length - 3,), generator=generator).tolist()
    # The tiny vocabulary cannot contain the real ids, so the frozen last-three assertion is patched by the caller.
    return RenderedPrompt("P_default", "tiny", "tiny", tuple(body + [151644, 77091, 198]))


def tiny_table(vocab: int = 320) -> FormTable:
    forms = {
        "alpha": [[5], [6, 7], [8, 9, 10]],
        "beta": [[5, 11], [12], [13, 14]],
        "gamma": [[15], [16, 17]],
    }
    endpoint = {
        "answer_forms": {word: {"forms": [{"text": str(ids), "token_ids": ids} for ids in values]} for word, values in forms.items()},
        "boundary_ids": list(range(200, 260)),
    }
    return FormTable.from_endpoint(endpoint, words=list(forms))


def _remap(prompt: RenderedPrompt, vocab: int) -> tuple[RenderedPrompt, tuple[int, int, int]]:
    """Map the real last-three ids into the tiny vocabulary (the tiny model cannot hold the real ids)."""
    suffix = (vocab - 3, vocab - 2, vocab - 1)
    ids = list(prompt.input_ids[:-3]) + list(suffix)
    return RenderedPrompt(prompt.persona_id, prompt.user_prompt, prompt.text, tuple(ids)), suffix


@torch.inference_mode()
def full_sequence_form_logp(model, prompt_ids, row: RowSteer, table: FormTable, weight32, position_offset: int = 1) -> np.ndarray:
    """Uncached reference: one forward over prompt + form with the hook at absolute prompt_len - position_offset.

    ``position_offset`` other than 1 exists only as a negative control (misplaced hook)."""
    out = np.empty(table.n_forms)
    prompt_len = len(prompt_ids)
    device = weight32.device
    boundary = torch.tensor(table.boundary_ids, device=device)
    for index, ids in enumerate(table.form_ids):
        tokens = torch.tensor([list(prompt_ids) + list(ids)], device=device)
        handles = []
        for block, vector in row.vectors.items():
            def hook(_m, _a, output, vector=vector):
                hidden = output[0] if isinstance(output, tuple) else output
                hidden = hidden.clone()
                if row.mode == LAST:
                    hidden[:, prompt_len - position_offset] += vector.to(hidden.device, hidden.dtype)
                else:
                    hidden[:, :prompt_len] += vector.to(hidden.device, hidden.dtype)
                return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
            handles.append(decoder_layers(model)[block].register_forward_hook(hook))
        try:
            hidden = model.model(input_ids=tokens, use_cache=False).last_hidden_state[0]
        finally:
            for handle in handles:
                handle.remove()
        logits = torch.nn.functional.linear(hidden.float(), weight32)
        lp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        value = float(lp[prompt_len - 1, ids[0]])
        for offset in range(len(ids) - 1):
            value += float(lp[prompt_len + offset, ids[offset + 1]])
        value += float(torch.logsumexp(lp[prompt_len + len(ids) - 1, boundary], dim=-1))
        out[index] = value
    return out


def tiny_model_suite() -> dict:
    """Hook/cache/scoring semantics on a tiny random Qwen2 in the running transformers version."""
    import transformers

    results: dict = {"transformers": transformers.__version__}
    model = tiny_model()
    vocab = model.config.vocab_size
    weight32 = lm_head_weight32(model)
    table = tiny_table(vocab)
    prompt, suffix = _remap(tiny_prompt(vocab=vocab), vocab)
    kw = {"weight32": weight32, "expected_suffix": suffix}
    if True:
        generator = torch.Generator().manual_seed(7)
        vector = torch.randn(model.config.hidden_size, generator=generator, dtype=torch.float32) * 3.0
        rows = {
            "unsteered": RowSteer({}),
            "last": RowSteer({13: vector}, LAST),
            "all": RowSteer({13: vector}, ALL),
            "site8": RowSteer({7: vector}, LAST),
        }
        worst = {}
        for name, row in rows.items():
            packed = score_prompt(model, prompt, [row], table, **kw).form_logp[0]
            sequential = score_sequential(model, prompt, row, table, **kw)
            full = full_sequence_form_logp(model, prompt.input_ids, row, table, weight32)
            worst[name] = float(max(np.abs(packed - sequential).max(), np.abs(packed - full).max()))
        results["packed_vs_sequential_vs_full_max_abs"] = worst
        results["packed_equivalence_pass"] = all(value <= 1e-5 for value in worst.values())
        batched = score_prompt(model, prompt, list(rows.values()), table, **kw).form_logp
        single = np.stack([score_prompt(model, prompt, [row], table, **kw).form_logp[0] for row in rows.values()])
        results["condition_batch_max_abs"] = float(np.abs(batched - single).max())
        results["condition_batch_pass"] = results["condition_batch_max_abs"] <= 1e-5
        steered = score_prompt(model, prompt, [rows["last"]], table, **kw).form_logp[0]
        plain = score_prompt(model, prompt, [rows["unsteered"]], table, **kw).form_logp[0]
        results["steering_changes_scores"] = bool(np.abs(steered - plain).max() > 1e-3)
    results.update(hook_semantics(model, prompt.input_ids, vector))
    results["pass"] = bool(
        results["packed_equivalence_pass"]
        and results["condition_batch_pass"]
        and results["steering_changes_scores"]
        and all(results["hooks"].values())
    )
    return results


@torch.inference_mode()
def hook_semantics(model, input_ids, vector: torch.Tensor, block: int = 13) -> dict:
    """Site, position, single-shot and zero-vector checks for the steering hook."""
    layers = decoder_layers(model)
    prompt_len = len(input_ids)
    ids = torch.tensor([list(input_ids)], device=next(model.parameters()).device)
    vector = vector.to(torch.float32)
    checks: dict[str, bool] = {}

    def run(row: RowSteer | None, capture: tuple[int, ...] = ()):
        captured: dict[int, torch.Tensor] = {}
        handles = []
        context = PrefillSteering(model, prompt_len, [row]) if row is not None else None
        if context is not None:
            context.__enter__()
        try:
            for index in capture:
                def hook(_m, _a, output, index=index):
                    captured[index] = (output[0] if isinstance(output, tuple) else output).detach().clone()
                handles.append(layers[index].register_forward_hook(hook))
            output = model.model(input_ids=ids, output_hidden_states=True, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
            if context is not None:
                context.__exit__(None, None, None)
        return output, captured, context

    base, base_capture, _ = run(None, (block - 1, block))
    zero, _, _ = run(RowSteer({block: torch.zeros_like(vector)}))
    checks["zero_vector_bitwise_inert"] = bool(
        all(torch.equal(a, b) for a, b in zip(base.hidden_states, zero.hidden_states))
        and torch.equal(base.last_hidden_state, zero.last_hidden_state)
    )
    steered, steered_capture, context = run(RowSteer({block: vector}), (block - 1, block))
    checks["fired_exactly_once"] = context.fired == {block: 1}
    checks["previous_block_unchanged"] = bool(torch.equal(base_capture[block - 1], steered_capture[block - 1]))
    delta = (steered_capture[block].float() - base_capture[block].float())[0]
    expected = vector.to(base_capture[block].dtype).float().to(delta.device)
    checks["site_changed_only_at_last_position"] = bool(
        torch.count_nonzero(delta[: prompt_len - 1]) == 0
        and torch.allclose(delta[prompt_len - 1], expected, atol=2e-2 * float(expected.abs().max()) + 1e-6, rtol=1e-2)
    )
    slot = block + 2
    later = (steered.hidden_states[slot].float() - base.hidden_states[slot].float())[0]
    checks["slot_ge_15_changed_at_last_only"] = bool(
        torch.count_nonzero(later[: prompt_len - 1]) == 0 and torch.count_nonzero(later[prompt_len - 1]) > 0
    )
    hidden_states = base.hidden_states
    checks["hidden_state_slot_mapping"] = bool(
        len(hidden_states) == len(layers) + 1
        and torch.equal(hidden_states[block], base_capture[block - 1])
        and torch.equal(hidden_states[block + 1], base_capture[block])
    )
    leaked = False
    try:
        with PrefillSteering(model, prompt_len, [RowSteer({block: vector})]):
            model.model(input_ids=ids, use_cache=False)
            model.model(input_ids=ids[:, :1], use_cache=False)
    except SteeringError:
        leaked = True
    checks["second_forward_inside_context_refused"] = leaked
    all_rows, all_capture, _ = run(RowSteer({block: vector}, ALL), (block,))
    delta_all = (all_capture[block].float() - base_capture[block].float())[0]
    checks["all_positions_changed_everywhere"] = bool((delta_all.abs().sum(dim=-1) > 0).all())
    return {"hooks": checks}


def real_model_hook_selftest(model, rendered: RenderedPrompt, magnitude: float, seed: int = 1000003) -> dict:
    """Real-model check on one extraction prompt in the default context (outcome-free)."""
    generator = torch.Generator().manual_seed(seed)
    direction = torch.randn(model.config.hidden_size, generator=generator, dtype=torch.float64)
    vector = (direction / direction.norm() * magnitude).to(torch.float32)
    result = hook_semantics(model, rendered.input_ids, vector)
    # hidden_states[block + 1] visibility is version-specific; the mapping check uses capture hooks at blocks
    # 12 and 13 against slots 13 and 14 of an unsteered forward, which holds in both versions.
    result["pass"] = bool(all(result["hooks"].values()))
    return result

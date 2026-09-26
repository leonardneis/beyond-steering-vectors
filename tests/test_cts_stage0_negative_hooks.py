"""Red-team negative and mutation tests: steering hooks, cache reuse and scoring (CTS Stage 0).

Everything runs on the randomly initialized tiny Qwen2 of ``selftest.tiny_model`` on CPU. No 7B forward, no
network, no real prompt text. Each mutation is applied with ``monkeypatch`` and must be detected either by a
guard (an exception) or by the equivalence references (packed vs full-sequence, batched vs single).
Gaps are recorded as ``xfail(strict=True)``.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.cts_stage0 import scoring, selftest, steering  # noqa: E402
from slgeo.cts_stage0.scoring import FormTable, ScoringError, lm_head_weight32, score_prompt, score_sequential  # noqa: E402
from slgeo.cts_stage0.steering import (  # noqa: E402
    ALL,
    LAST,
    PrefillSteering,
    RowSteer,
    SteeringError,
    assert_no_hooks,
    decoder_layers,
)

BLOCK = 13
EQUIV_TOL = 1e-5
DETECT_TOL = 1e-3


# ----------------------------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def tiny():
    model = selftest.tiny_model()
    vocab = model.config.vocab_size
    prompt, suffix = selftest._remap(selftest.tiny_prompt(vocab=vocab), vocab)
    generator = torch.Generator().manual_seed(7)
    vector = torch.randn(model.config.hidden_size, generator=generator, dtype=torch.float32) * 3.0
    return {
        "model": model,
        "prompt": prompt,
        "suffix": suffix,
        "ids": torch.tensor([list(prompt.input_ids)]),
        "L": len(prompt.input_ids),
        "table": selftest.tiny_table(vocab),
        "weight32": lm_head_weight32(model),
        "vector": vector,
    }


def _strip_cts_hooks(model) -> None:
    for layer in decoder_layers(model):
        for store in (layer._forward_hooks, layer._forward_pre_hooks):
            for key in [k for k, hook in store.items() if getattr(hook, "_cts_steering", False)]:
                del store[key]


def _packed(t, row, **extra):
    return score_prompt(t["model"], t["prompt"], [row], t["table"], weight32=t["weight32"], expected_suffix=t["suffix"], **extra)


def _full(t, row, position_offset=1):
    return selftest.full_sequence_form_logp(t["model"], t["prompt"].input_ids, row, t["table"], t["weight32"], position_offset)


# --------------------------------------------------------------------------- 1. steering position guards


def test_second_forward_inside_prefill_context_raises(tiny):
    model, ids, L = tiny["model"], tiny["ids"], tiny["L"]
    with pytest.raises(SteeringError, match="more than once"):
        with torch.inference_mode(), PrefillSteering(model, L, [RowSteer({BLOCK: tiny["vector"]})]):
            model.model(input_ids=ids, use_cache=False)
            model.model(input_ids=ids[:, :1], use_cache=False)  # a continuation/decode step (index -1 edit)
    assert_no_hooks(model)


def test_cached_continuation_inside_prefill_context_raises(tiny):
    from transformers.cache_utils import DynamicCache

    model, ids, L = tiny["model"], tiny["ids"], tiny["L"]
    with pytest.raises(SteeringError):
        with torch.inference_mode(), PrefillSteering(model, L, [RowSteer({BLOCK: tiny["vector"]})]):
            out = model.model(input_ids=ids, past_key_values=DynamicCache(), use_cache=True)
            model.model(
                input_ids=torch.tensor([[5]]),
                position_ids=torch.tensor([[L]]),
                past_key_values=out.past_key_values,
                use_cache=True,
            )
    assert_no_hooks(model)


@pytest.mark.parametrize("delta", [1, -1, 5])
def test_prefill_of_wrong_length_raises(tiny, delta):
    model, ids, L = tiny["model"], tiny["ids"], tiny["L"]
    with pytest.raises(SteeringError, match="prefill of shape"):
        with torch.inference_mode(), PrefillSteering(model, L + delta, [RowSteer({BLOCK: tiny["vector"]})]):
            model.model(input_ids=ids, use_cache=False)
    assert_no_hooks(model)


def test_wrong_batch_rows_raise(tiny):
    model, ids, L = tiny["model"], tiny["ids"], tiny["L"]
    rows = [RowSteer({BLOCK: tiny["vector"]}), RowSteer({})]
    with pytest.raises(SteeringError, match="prefill of shape"):
        with torch.inference_mode(), PrefillSteering(model, L, rows):
            model.model(input_ids=ids, use_cache=False)  # batch 1 while two rows are declared


def test_prompt_len_from_batchencoding_is_refused(tiny):
    """RT-01: under transformers 5.x ``len(apply_chat_template(tokenize=True))`` is 2 (BatchEncoding keys)."""
    with pytest.raises(SteeringError, match="implausibly small"):
        PrefillSteering(tiny["model"], 2, [RowSteer({BLOCK: tiny["vector"]})])


@pytest.mark.parametrize("block", [16, 100])
def test_nonexistent_block_raises(tiny, block):
    with pytest.raises(SteeringError, match="does not exist"):
        PrefillSteering(tiny["model"], tiny["L"], [RowSteer({block: tiny["vector"]})])


@pytest.mark.parametrize(
    "make",
    [
        lambda v: RowSteer({-1: v}),
        lambda v: RowSteer({"13": v}),
        lambda v: RowSteer({BLOCK: v.double()}),
        lambda v: RowSteer({BLOCK: v.half()}),
        lambda v: RowSteer({BLOCK: v[None]}),
        lambda v: RowSteer({BLOCK: v * float("nan")}),
        lambda v: RowSteer({BLOCK: v * float("inf")}),
        lambda v: RowSteer({BLOCK: v}, mode="first"),
    ],
    ids=["negative_block", "str_block", "float64", "float16", "2d", "nan", "inf", "unknown_mode"],
)
def test_malformed_rowsteer_raises(tiny, make):
    with pytest.raises(SteeringError):
        make(tiny["vector"])


def test_vector_of_wrong_hidden_size_raises(tiny):
    model, ids, L = tiny["model"], tiny["ids"], tiny["L"]
    with pytest.raises(SteeringError, match="hidden size"):
        with torch.inference_mode(), PrefillSteering(model, L, [RowSteer({BLOCK: torch.ones(65)})]):
            model.model(input_ids=ids, use_cache=False)
    assert_no_hooks(model)


def test_hook_that_never_fires_raises_on_exit(tiny):
    with pytest.raises(SteeringError, match="did not fire exactly once"):
        with PrefillSteering(tiny["model"], tiny["L"], [RowSteer({BLOCK: tiny["vector"]})]):
            pass
    assert_no_hooks(tiny["model"])


def test_leftover_cts_hook_is_detected(tiny):
    model = tiny["model"]
    context = PrefillSteering(model, tiny["L"], [RowSteer({BLOCK: tiny["vector"]})])
    context.__enter__()
    try:
        with pytest.raises(SteeringError, match="leftover steering hook"):
            assert_no_hooks(model)
        with pytest.raises(SteeringError, match="leftover steering hook"):
            PrefillSteering(model, tiny["L"], [RowSteer({})]).__enter__()  # nested context refused
        with pytest.raises(SteeringError, match="leftover steering hook"):
            _packed(tiny, RowSteer({}))
    finally:
        for handle in context._handles:
            handle.remove()
    assert_no_hooks(model)


def test_exception_inside_context_removes_hooks(tiny):
    """RT-50: a hook handle must not survive an exception."""
    model = tiny["model"]
    with pytest.raises(ZeroDivisionError):
        with PrefillSteering(model, tiny["L"], [RowSteer({BLOCK: tiny["vector"]})]):
            1 / 0
    assert_no_hooks(model)


def test_foreign_layer_hook_is_detected(tiny):
    model = tiny["model"]

    def foreign(_m, _a, output):
        return output

    handle = decoder_layers(model)[BLOCK].register_forward_hook(foreign)
    try:
        with pytest.raises(SteeringError):
            assert_no_hooks(model)
    finally:
        handle.remove()


def test_hook_on_final_norm_is_detected(tiny):
    model = tiny["model"]

    def hook(_m, _a, output):
        return output

    hook._cts_steering = True
    handle = model.model.norm.register_forward_hook(hook)
    try:
        with pytest.raises(SteeringError):
            assert_no_hooks(model)
    finally:
        handle.remove()


def test_hook_semantics_pass_unmutated_and_zero_vector_bitwise_inert(tiny):
    result = selftest.hook_semantics(tiny["model"], tiny["prompt"].input_ids, tiny["vector"], block=BLOCK)
    assert result["hooks"]["zero_vector_bitwise_inert"] is True
    assert all(result["hooks"].values()), result["hooks"]


def test_zero_vector_scores_bitwise_equal_to_unsteered(tiny):
    zero = _packed(tiny, RowSteer({BLOCK: torch.zeros_like(tiny["vector"])})).form_logp
    plain = _packed(tiny, RowSteer({})).form_logp
    assert np.array_equal(zero, plain)
    zero_all = _packed(tiny, RowSteer({BLOCK: torch.zeros_like(tiny["vector"])}, ALL)).form_logp
    assert np.array_equal(zero_all, plain)


def test_reference_equivalences_hold_unmutated(tiny):
    for row in (RowSteer({}), RowSteer({BLOCK: tiny["vector"]}, LAST), RowSteer({BLOCK: tiny["vector"]}, ALL)):
        packed = _packed(tiny, row).form_logp[0]
        full = _full(tiny, row)
        sequential = score_sequential(tiny["model"], tiny["prompt"], row, tiny["table"], weight32=tiny["weight32"],
                                      expected_suffix=tiny["suffix"])
        assert np.abs(packed - full).max() <= EQUIV_TOL
        assert np.abs(packed - sequential).max() <= EQUIV_TOL
    steered = _packed(tiny, RowSteer({BLOCK: tiny["vector"]})).form_logp
    assert np.abs(steered - _packed(tiny, RowSteer({})).form_logp).max() > DETECT_TOL


def test_misplaced_reference_is_a_valid_negative_control(tiny):
    row = RowSteer({BLOCK: tiny["vector"]})
    assert np.abs(_packed(tiny, row).form_logp[0] - _full(tiny, row, position_offset=2)).max() > DETECT_TOL


# ------------------------------------------------------------------------------------------ 2. mutations


def _mutated_make_hook(offset: int):
    def make(self, block):
        def hook(_module, _args, kwargs, output):
            self.fired[block] += 1
            hidden = output[0] if isinstance(output, tuple) else output
            vectors, last_rows, _all_rows = self._row_tables(block, hidden.shape[-1], hidden.device, hidden.dtype)
            new_hidden = hidden.clone()
            index = torch.tensor(last_rows, device=hidden.device)
            new_hidden[index, self.prompt_len - offset, :] = hidden[index, self.prompt_len - offset, :] + vectors[index]
            return (new_hidden, *output[1:]) if isinstance(output, tuple) else new_hidden

        hook._cts_steering = True
        return hook

    return make


def test_mutation_hook_at_prompt_len_minus_2_detected_by_full_sequence(tiny, monkeypatch):
    """(a) RT-02: a hook at L-2 keeps packed == sequential (both mutated) but fails the uncached reference."""
    row = RowSteer({BLOCK: tiny["vector"]})
    monkeypatch.setattr(PrefillSteering, "_make_hook", _mutated_make_hook(2))
    packed = _packed(tiny, row).form_logp[0]
    sequential = score_sequential(tiny["model"], tiny["prompt"], row, tiny["table"], weight32=tiny["weight32"],
                                  expected_suffix=tiny["suffix"])
    full = _full(tiny, row)
    assert np.abs(packed - sequential).max() <= EQUIV_TOL  # the cache-based references cannot see it
    assert np.abs(packed - full).max() > DETECT_TOL  # the full-sequence reference does


def test_mutation_steer_block_14_fails_block_13_site_checks(tiny, monkeypatch):
    """(b) RT-07: hook registered on block 14 while block 13 is requested."""

    def enter(self):
        assert_no_hooks(self.model)
        for block in self.blocks:
            self._handles.append(self.layers[block + 1].register_forward_hook(self._make_hook(block), with_kwargs=True))
        return self

    monkeypatch.setattr(PrefillSteering, "__enter__", enter)
    checks = selftest.hook_semantics(tiny["model"], tiny["prompt"].input_ids, tiny["vector"], block=BLOCK)["hooks"]
    assert checks["site_changed_only_at_last_position"] is False
    assert not all(checks.values())
    assert_no_hooks(tiny["model"])


def test_mutation_steering_kept_open_is_caught_by_hygiene(tiny, monkeypatch):
    """(c1) hooks not removed at the end of the prefill: score_prompt's post-prefill hygiene check fires."""
    monkeypatch.setattr(PrefillSteering, "__exit__", lambda self, *exc: None)
    try:
        with pytest.raises(SteeringError, match="leftover steering hook"):
            _packed(tiny, RowSteer({BLOCK: tiny["vector"]}))
    finally:
        _strip_cts_hooks(tiny["model"])
    assert_no_hooks(tiny["model"])


def test_mutation_steering_kept_open_without_hygiene_is_caught_by_single_shot(tiny, monkeypatch):
    """(c2) same, with the hygiene check disabled: the continuation forward re-enters the hook and raises."""
    monkeypatch.setattr(PrefillSteering, "__exit__", lambda self, *exc: None)
    monkeypatch.setattr(scoring, "assert_no_hooks", lambda model: None)
    try:
        with pytest.raises(SteeringError, match="more than once"):
            _packed(tiny, RowSteer({BLOCK: tiny["vector"]}))
    finally:
        _strip_cts_hooks(tiny["model"])
    assert_no_hooks(tiny["model"])


def test_mutation_residual_intervention_scope_is_caught_by_equivalence(tiny, monkeypatch):
    """(c3) RT-03/RT-20: a naive ``hidden[:, -1] += v`` hook held open through the continuation.

    It carries no CTS tag; hook hygiene now rejects every foreign hook before the continuation forward.
    """
    handles = []

    class NaiveScope:
        def __init__(self, model, prompt_len, rows):
            self.model, self.rows = model, rows

        def __enter__(self):
            for block, vector in self.rows[0].vectors.items():
                def hook(_m, _a, output, vector=vector):
                    hidden = (output[0] if isinstance(output, tuple) else output).clone()
                    hidden[:, -1] += vector.to(hidden.dtype)
                    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
                handles.append(decoder_layers(self.model)[block].register_forward_hook(hook))
            return self

        def __exit__(self, *exc):
            return None  # never removed during the continuation

    row = RowSteer({BLOCK: tiny["vector"]})
    monkeypatch.setattr(scoring, "PrefillSteering", NaiveScope)
    try:
        with pytest.raises(SteeringError, match="foreign forward hook"):
            _packed(tiny, row)
    finally:
        for handle in handles:
            handle.remove()
    monkeypatch.undo()


def _teacher_forced(model, weight32, cache, prompt_len, ids, first_lp, boundary):
    out = model.model(
        input_ids=torch.tensor([list(ids)]),
        position_ids=torch.arange(prompt_len, prompt_len + len(ids))[None],
        past_key_values=cache,
        use_cache=True,
    )
    logits = torch.nn.functional.linear(out.last_hidden_state[0].float(), weight32)
    lp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    value = float(first_lp[ids[0]])
    for offset in range(len(ids) - 1):
        value += float(lp[offset, ids[offset + 1]])
    return value + float(torch.logsumexp(lp[len(ids) - 1, boundary], dim=-1))


@torch.inference_mode()
def test_mutation_reused_grown_cache_differs_from_reference(tiny):
    """(d) RT-17: scoring form B from the cache already grown by form A conditions on prompt + A."""
    from transformers.cache_utils import DynamicCache

    model, weight32, table, L = tiny["model"], tiny["weight32"], tiny["table"], tiny["L"]
    boundary = torch.tensor(table.boundary_ids)
    cache = DynamicCache()
    prefill = model.model(input_ids=tiny["ids"], past_key_values=cache, use_cache=True)
    first = torch.nn.functional.linear(prefill.last_hidden_state[0, -1].float(), weight32)
    first_lp = first - torch.logsumexp(first, dim=-1)
    grown = prefill.past_key_values
    form_a, form_b = table.form_ids[2], table.form_ids[3]
    _teacher_forced(model, weight32, grown, L, form_a, first_lp, boundary)
    assert scoring.cache_length(grown) == L + len(form_a)
    reused = _teacher_forced(model, weight32, grown, L, form_b, first_lp, boundary)
    reference = score_sequential(model, tiny["prompt"], RowSteer({}), table, weight32=weight32, expected_suffix=tiny["suffix"])
    assert abs(reused - reference[3]) > DETECT_TOL


def test_mutation_shared_cache_object_caught_by_cache_length_guard(tiny, monkeypatch):
    """(d') every 'fresh' per-form cache is the same (growing) object -> the cache-length guard fires."""
    from transformers import cache_utils

    real = cache_utils.DynamicCache
    shared = {}

    def factory(*args, **kwargs):
        if "cache" not in shared:
            shared["cache"] = real(*args, **kwargs)
        return shared["cache"]

    monkeypatch.setattr(cache_utils, "DynamicCache", factory)
    with pytest.raises(ScoringError, match="cache length"):
        score_sequential(tiny["model"], tiny["prompt"], RowSteer({}), tiny["table"], weight32=tiny["weight32"],
                         expected_suffix=tiny["suffix"])


def _tiny_endpoint(drop: tuple[str, int] | None = None, reverse: bool = False):
    forms = {
        "alpha": [[5], [6, 7], [8, 9, 10]],
        "beta": [[5, 11], [12], [13, 14]],
        "gamma": [[15], [16, 17]],
    }
    if drop is not None:
        word, index = drop
        forms[word] = [ids for i, ids in enumerate(forms[word]) if i != index]
    if reverse:
        forms = {word: list(reversed(values)) for word, values in forms.items()}
    endpoint = {
        "answer_forms": {w: {"forms": [{"text": str(ids), "token_ids": ids} for ids in v]} for w, v in forms.items()},
        "boundary_ids": list(range(200, 260)),
    }
    return FormTable.from_endpoint(endpoint, words=list(forms))


def test_mutation_dropped_answer_form_changes_word_score(tiny):
    """(e) RT-33: dropping one form of a word changes L_w of that word."""
    full = score_prompt(tiny["model"], tiny["prompt"], [RowSteer({})], tiny["table"], weight32=tiny["weight32"],
                        expected_suffix=tiny["suffix"])
    dropped_table = _tiny_endpoint(drop=("alpha", 1))
    dropped = score_prompt(tiny["model"], tiny["prompt"], [RowSteer({})], dropped_table, weight32=tiny["weight32"],
                           expected_suffix=tiny["suffix"])
    assert dropped_table.n_forms == tiny["table"].n_forms - 1
    assert dropped_table.form_ids != tiny["table"].form_ids
    assert abs(full.word_logp[0, 0] - dropped.word_logp[0, 0]) > DETECT_TOL
    assert np.abs(full.word_logp[0, 1:] - dropped.word_logp[0, 1:]).max() <= EQUIV_TOL


def test_form_order_invariance(tiny):
    """RT-17/RT-34: the order of forms inside the packed continuation does not change L_w."""
    forward = score_prompt(tiny["model"], tiny["prompt"], [RowSteer({BLOCK: tiny["vector"]})], tiny["table"],
                           weight32=tiny["weight32"], expected_suffix=tiny["suffix"])
    backward = score_prompt(tiny["model"], tiny["prompt"], [RowSteer({BLOCK: tiny["vector"]})], _tiny_endpoint(reverse=True),
                            weight32=tiny["weight32"], expected_suffix=tiny["suffix"])
    assert np.abs(forward.word_logp - backward.word_logp).max() <= EQUIV_TOL


@pytest.mark.parametrize(
    "answer_forms, match",
    [
        ({"alpha": {"forms": []}}, "No answer forms"),
        ({"alpha": {"forms": [{"text": "x", "token_ids": []}]}}, "Empty answer form"),
        ({"alpha": {"forms": [{"text": "x", "token_ids": [5]}, {"text": "y", "token_ids": [5]}]}}, "Duplicate"),
    ],
)
def test_malformed_form_table_raises(answer_forms, match):
    with pytest.raises(ScoringError, match=match):
        FormTable.from_endpoint({"answer_forms": answer_forms, "boundary_ids": [200]}, words=["alpha"])


def test_frozen_endpoint_table_is_complete():
    from slgeo.cts_stage0.package import FrozenPackage

    package = FrozenPackage.from_repo(ROOT)
    endpoint = package.endpoint
    table = FormTable.from_endpoint(endpoint)
    assert table.words == tuple(endpoint["scoring_words"] + endpoint["non_animal_words"])
    frozen = [tuple(f["token_ids"]) for w in table.words for f in endpoint["answer_forms"][w]["forms"]]
    assert list(table.form_ids) == frozen
    assert table.n_forms == 104
    assert table.boundary_ids == tuple(endpoint["boundary_ids"])


def test_mutation_prefill_suffix_is_asserted(tiny):
    """RT-01/RT-05: a prefill that does not end with the frozen last-three ids is refused before any forward."""
    from slgeo.cts_stage0.render import RenderError, RenderedPrompt

    ids = list(tiny["prompt"].input_ids)
    shifted = RenderedPrompt("P_default", "tiny", "tiny", tuple(ids[:-1]))
    with pytest.raises(RenderError):
        score_prompt(tiny["model"], shifted, [RowSteer({})], tiny["table"], weight32=tiny["weight32"],
                     expected_suffix=tiny["suffix"])
    with pytest.raises(RenderError):  # the production suffix (151644, 77091, 198) is the default
        score_prompt(tiny["model"], tiny["prompt"], [RowSteer({})], tiny["table"], weight32=tiny["weight32"])


def test_batched_rows_equal_single_rows_and_broadcast_mutation_detected(tiny, monkeypatch):
    """RT-18: distinct per-row vectors; a mutation that broadcasts row 0's vector is detected."""
    generator = torch.Generator().manual_seed(11)
    rows = [
        RowSteer({}),
        RowSteer({BLOCK: tiny["vector"]}, LAST),
        RowSteer({BLOCK: torch.randn(64, generator=generator) * 3.0}, ALL),
        RowSteer({7: torch.randn(64, generator=generator) * 3.0}, LAST),
    ]
    single = np.stack([_packed(tiny, row).form_logp[0] for row in rows])
    batched = score_prompt(tiny["model"], tiny["prompt"], rows[::-1], tiny["table"], weight32=tiny["weight32"],
                           expected_suffix=tiny["suffix"]).form_logp[::-1]
    assert np.abs(batched - single).max() <= EQUIV_TOL

    real = PrefillSteering._row_tables

    def broadcast(self, block, hidden_size, device, dtype):
        vectors, last_rows, all_rows = real(self, block, hidden_size, device, dtype)
        steered = last_rows + all_rows
        if steered:
            vectors = vectors[min(steered)].expand_as(vectors).clone()
        return vectors, last_rows, all_rows

    monkeypatch.setattr(PrefillSteering, "_row_tables", broadcast)
    mutated = score_prompt(tiny["model"], tiny["prompt"], rows, tiny["table"], weight32=tiny["weight32"],
                           expected_suffix=tiny["suffix"]).form_logp
    assert np.abs(mutated - single).max() > DETECT_TOL


def test_mutation_fp16_log_softmax_deviates_from_float32_reference(tiny, monkeypatch):
    """(f) RT-36: logits / log_softmax in float16 deviate from the float32 reference beyond 1e-4."""
    model16 = copy.deepcopy(tiny["model"]).half()
    weight32 = lm_head_weight32(model16)
    row = RowSteer({BLOCK: tiny["vector"]})

    def run():
        return score_prompt(model16, tiny["prompt"], [row], tiny["table"], weight32=weight32,
                            expected_suffix=tiny["suffix"]).form_logp[0]

    try:
        production = run()
    except RuntimeError as exc:  # pragma: no cover - CPU without fp16 kernels
        pytest.skip(f"float16 forward unsupported on this CPU build: {exc}")
    # Same fp16 hidden states, logits in float64: isolates the logit/log_softmax precision.
    monkeypatch.setattr(scoring, "_final_norm_logits",
                        lambda _m, hidden, weight: torch.nn.functional.linear(hidden.double(), weight.double()).float())
    reference = run()
    assert np.abs(production - reference).max() <= 1e-4
    # An fp16 logit path is now refused at the call site before any log_softmax.
    monkeypatch.setattr(scoring, "_final_norm_logits",
                        lambda _m, hidden, weight: torch.nn.functional.linear(hidden.half(), weight.half()))
    with pytest.raises(ScoringError, match="float32"):
        run()


def test_fp16_logits_rejected_by_score_prompt_itself(tiny, monkeypatch):
    monkeypatch.setattr(scoring, "_final_norm_logits",
                        lambda _m, hidden, weight: torch.nn.functional.linear(hidden.half(), weight.half()))
    with pytest.raises((ScoringError, RuntimeError)):
        _packed(tiny, RowSteer({}))

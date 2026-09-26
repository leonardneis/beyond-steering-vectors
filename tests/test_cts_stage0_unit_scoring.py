from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.cts_stage0 import scoring as sc
from slgeo.cts_stage0 import selftest
from slgeo.cts_stage0.package import FrozenPackage, sha256_text
from slgeo.cts_stage0.render import RenderError
from slgeo.cts_stage0.scoring import FormTable, ScoringError
from slgeo.cts_stage0.steering import ALL, LAST, PrefillSteering, RowSteer, SteeringError

TOL = 1e-5


# --- FormTable on the frozen endpoint --------------------------------------------------------------


@pytest.fixture(scope="module")
def endpoint():
    return FrozenPackage.from_repo(ROOT).endpoint


def test_real_form_table(endpoint):
    table = FormTable.from_endpoint(endpoint)
    assert table.words == tuple(endpoint["scoring_words"] + endpoint["non_animal_words"])
    assert len(table.words) == 14 and table.n_forms == 104
    frozen = [(w, tuple(f["token_ids"]), f["text"]) for w in table.words for f in endpoint["answer_forms"][w]["forms"]]
    assert [(table.words[i], ids, text) for i, ids, text in zip(table.form_word, table.form_ids, table.form_text)] == frozen
    assert list(table.boundary_ids) == endpoint["boundary_ids"]
    assert len(table.boundary_ids) == endpoint["boundary_count"]
    assert sha256_text(json.dumps(endpoint["boundary_ids"])) == endpoint["boundary_ids_sha256"]
    tokens, depths, segments, spans = table.packed()
    assert len(tokens) == sum(len(ids) for ids in table.form_ids)
    assert spans[-1][1] == len(tokens) and len(set(segments)) == 104


def _endpoint(forms, boundary=(200, 201)):
    return {
        "answer_forms": {w: {"forms": [{"text": str(ids), "token_ids": ids} for ids in values]} for w, values in forms.items()},
        "boundary_ids": list(boundary),
    }


def test_duplicate_form_raises():
    with pytest.raises(ScoringError, match="Duplicate"):
        FormTable.from_endpoint(_endpoint({"a": [[5], [5]]}), words=["a"])


def test_empty_forms_raise():
    with pytest.raises(ScoringError, match="No answer forms"):
        FormTable.from_endpoint(_endpoint({"a": []}), words=["a"])
    with pytest.raises(ScoringError, match="Empty answer form"):
        FormTable.from_endpoint(_endpoint({"a": [[]]}), words=["a"])


def test_gather_plan_structure():
    table = FormTable.from_endpoint(_endpoint({"a": [[5], [6, 7, 8]], "b": [[9, 10]]}), words=["a", "b"])
    first, trans_pos, trans_tok, trans_form, last = table.gather_plan()
    assert first == [5, 6, 9]
    assert trans_pos == [1, 2, 4] and trans_tok == [7, 8, 10] and trans_form == [1, 1, 2]
    assert last == [0, 3, 5]
    tokens, depths, segments, _ = table.packed()
    assert tokens == [5, 6, 7, 8, 9, 10] and depths == [0, 0, 1, 2, 0, 1] and segments == [0, 1, 1, 1, 2, 2]


def test_word_scores_logaddexp():
    table = FormTable.from_endpoint(_endpoint({"a": [[5], [6]], "b": [[7]]}), words=["a", "b"])
    form = np.log(np.array([[0.1, 0.2, 0.3]]))
    np.testing.assert_allclose(sc.word_scores(form, table), np.log([[0.3, 0.3]]), atol=1e-12)


# --- tiny model ------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny():
    model = selftest.tiny_model()
    vocab = model.config.vocab_size
    weight32 = sc.lm_head_weight32(model)
    table = selftest.tiny_table(vocab)
    prompt, suffix = selftest._remap(selftest.tiny_prompt(vocab=vocab), vocab)
    vector = torch.randn(model.config.hidden_size, generator=torch.Generator().manual_seed(7)) * 3.0
    rows = {
        "unsteered": RowSteer({}),
        "last": RowSteer({13: vector}, LAST),
        "all": RowSteer({13: vector}, ALL),
        "site8": RowSteer({7: vector}, LAST),
        "zero": RowSteer({13: torch.zeros_like(vector)}, LAST),
    }
    return {"model": model, "weight32": weight32, "table": table, "prompt": prompt, "suffix": suffix, "rows": rows}


def _kw(t):
    return {"weight32": t["weight32"], "expected_suffix": t["suffix"]}


@pytest.mark.parametrize("row", ["unsteered", "last", "all", "site8"])
def test_packed_equals_sequential_and_full(tiny, row):
    t = tiny
    steer = t["rows"][row]
    packed = sc.score_prompt(t["model"], t["prompt"], [steer], t["table"], **_kw(t))
    sequential = sc.score_sequential(t["model"], t["prompt"], steer, t["table"], **_kw(t))
    full = selftest.full_sequence_form_logp(t["model"], t["prompt"].input_ids, steer, t["table"], t["weight32"])
    np.testing.assert_allclose(packed.form_logp[0], sequential, atol=TOL, rtol=0)
    np.testing.assert_allclose(packed.form_logp[0], full, atol=TOL, rtol=0)
    np.testing.assert_allclose(packed.word_logp, sc.word_scores(packed.form_logp, t["table"]), atol=1e-12)
    assert packed.form_logp.dtype == np.float64 and np.isfinite(packed.form_logp).all()
    assert (packed.form_logp < 0).all()


def test_misplaced_hook_is_detected(tiny):
    t = tiny
    steer = t["rows"]["last"]
    packed = sc.score_prompt(t["model"], t["prompt"], [steer], t["table"], **_kw(t)).form_logp[0]
    wrong = selftest.full_sequence_form_logp(t["model"], t["prompt"].input_ids, steer, t["table"], t["weight32"], position_offset=2)
    assert np.abs(packed - wrong).max() > 1e-3


def test_steering_changes_scores_and_zero_is_inert(tiny):
    t = tiny
    score = lambda name: sc.score_prompt(t["model"], t["prompt"], [t["rows"][name]], t["table"], **_kw(t)).form_logp[0]
    plain = score("unsteered")
    assert np.abs(score("last") - plain).max() > 1e-3
    np.testing.assert_array_equal(score("zero"), plain)


def test_form_order_invariance(tiny):
    t = tiny
    table = t["table"]
    order = list(reversed(range(table.n_forms)))
    reversed_table = FormTable(
        table.words,
        tuple(table.form_word[i] for i in order),
        tuple(table.form_text[i] for i in order),
        tuple(table.form_ids[i] for i in order),
        table.boundary_ids,
    )
    for name in ("unsteered", "last"):
        steer = t["rows"][name]
        a = sc.score_prompt(t["model"], t["prompt"], [steer], table, **_kw(t))
        b = sc.score_prompt(t["model"], t["prompt"], [steer], reversed_table, **_kw(t))
        np.testing.assert_allclose(b.form_logp[0], a.form_logp[0][order], atol=TOL, rtol=0)
        np.testing.assert_allclose(b.word_logp, a.word_logp, atol=TOL, rtol=0)


def test_condition_batched_equals_single_rows(tiny):
    t = tiny
    rows = list(t["rows"].values()) + [t["rows"]["last"], t["rows"]["unsteered"]]
    batched = sc.score_prompt(t["model"], t["prompt"], rows, t["table"], **_kw(t))
    single = np.stack([sc.score_prompt(t["model"], t["prompt"], [row], t["table"], **_kw(t)).form_logp[0] for row in rows])
    np.testing.assert_allclose(batched.form_logp, single, atol=TOL, rtol=0)
    assert batched.form_logp.shape == (len(rows), t["table"].n_forms)


def test_row_chunking_beyond_eight_rows(tiny):
    t = tiny
    rows = [t["rows"]["last"], t["rows"]["unsteered"]] * 5  # 10 rows > _ROW_CHUNK
    batched = sc.score_prompt(t["model"], t["prompt"], rows, t["table"], **_kw(t)).form_logp
    np.testing.assert_allclose(batched[::2], np.repeat(batched[:1], 5, axis=0), atol=TOL, rtol=0)
    np.testing.assert_allclose(batched[1::2], np.repeat(batched[1:2], 5, axis=0), atol=TOL, rtol=0)


@torch.inference_mode()
def _hand_form_logp(model, prompt_ids, form, boundary) -> float:
    """Explicit uncached computation: product of token probabilities times the summed boundary mass."""
    tokens = torch.tensor([list(prompt_ids) + list(form)])
    hidden = model.model(input_ids=tokens, use_cache=False).last_hidden_state[0]
    logits = torch.nn.functional.linear(hidden.float(), model.lm_head.weight.float()).double()
    probabilities = torch.softmax(logits, dim=-1)
    n = len(prompt_ids)
    value = 0.0
    for offset, token in enumerate(form):
        value += float(torch.log(probabilities[n - 1 + offset, token]))
    mass = 0.0
    for b in boundary:
        mass += float(probabilities[n - 1 + len(form), b])
    return value + float(np.log(mass))


def test_boundary_mass_equals_explicit_sum(tiny):
    t = tiny
    table = t["table"]
    packed = sc.score_prompt(t["model"], t["prompt"], [t["rows"]["unsteered"]], table, **_kw(t)).form_logp[0]
    for index in (0, 2, 4, table.n_forms - 1):
        expected = _hand_form_logp(t["model"], t["prompt"].input_ids, table.form_ids[index], table.boundary_ids)
        assert packed[index] == pytest.approx(expected, abs=TOL)


def test_boundary_set_matters(tiny):
    t = tiny
    table = t["table"]
    narrow = FormTable(table.words, table.form_word, table.form_text, table.form_ids, table.boundary_ids[:5])
    wide = sc.score_prompt(t["model"], t["prompt"], [t["rows"]["unsteered"]], table, **_kw(t)).form_logp[0]
    small = sc.score_prompt(t["model"], t["prompt"], [t["rows"]["unsteered"]], narrow, **_kw(t)).form_logp[0]
    assert (small < wide).all()


def test_first_position_outputs(tiny):
    t = tiny
    cjk = torch.tensor([30, 31, 32])
    ref = sc.score_prompt(t["model"], t["prompt"], [t["rows"]["unsteered"]], t["table"], return_first_logprobs=True, **_kw(t))
    first = ref.first_logprobs[0]
    assert first.shape == (t["model"].config.vocab_size,)
    assert float(torch.logsumexp(first, dim=0)) == pytest.approx(0.0, abs=1e-5)
    result = sc.score_prompt(
        t["model"], t["prompt"], [t["rows"]["unsteered"], t["rows"]["last"]], t["table"],
        cjk_ids=cjk, reference_first_lp=first, **_kw(t),
    )
    assert result.cjk_mass[0] == pytest.approx(float(first[cjk].exp().sum()), rel=1e-5)
    assert result.kl_to_reference[0] == pytest.approx(0.0, abs=1e-6)
    assert result.kl_to_reference[1] > 0
    assert result.top1[0] == int(first.argmax())
    assert np.isnan(ref.kl_to_reference).all() and np.isnan(ref.cjk_mass).all()


def test_wrong_suffix_refused(tiny):
    t = tiny
    with pytest.raises(RenderError):
        sc.score_prompt(t["model"], t["prompt"], [t["rows"]["unsteered"]], t["table"], weight32=t["weight32"])
    with pytest.raises(RenderError):
        sc.score_sequential(t["model"], t["prompt"], t["rows"]["unsteered"], t["table"], weight32=t["weight32"])


def test_leftover_hook_refused(tiny):
    t = tiny
    context = PrefillSteering(t["model"], len(t["prompt"].input_ids), [t["rows"]["last"]])
    context.__enter__()
    try:
        with pytest.raises(SteeringError, match="leftover"):
            sc.score_prompt(t["model"], t["prompt"], [t["rows"]["unsteered"]], t["table"], **_kw(t))
    finally:
        for handle in context._handles:
            handle.remove()
        context._handles.clear()


def test_tiny_model_suite_passes():
    result = selftest.tiny_model_suite()
    assert result["pass"], result

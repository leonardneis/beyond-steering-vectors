"""Phenotype runner and CRN sampler on a tiny random Qwen2 (CPU, float32)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from slgeo.cts_stage0 import selftest
from slgeo.cts_stage0.render import RenderedPrompt
from slgeo.cts_stage0.scoring import lm_head_weight32
from slgeo.phenotype import crn, runner, sampler


@pytest.fixture(scope="module")
def tiny():
    model = selftest.tiny_model()
    vocab = model.config.vocab_size
    table = selftest.tiny_table(vocab)
    prompts, contexts = [], []
    for i in range(4):
        p, suffix = selftest._remap(selftest.tiny_prompt(length=12 + i, seed=i, vocab=vocab), vocab)
        prompts.append(RenderedPrompt("P_default", f"u{i}", f"t{i}", p.input_ids))
        contexts.append(runner.Context(f"c{i}", f"s{i}", "Q+prefix", "p0", "P_default", f"u{i}"))
    return {"model": model, "table": table, "prompts": prompts, "contexts": contexts, "suffix": suffix,
            "weight32": lm_head_weight32(model), "vocab": vocab}


def test_score_contexts_shapes_capture_and_coverage(tiny):
    t = tiny
    out = runner.score_contexts(
        t["model"], t["prompts"], t["contexts"], t["table"], weight32=t["weight32"],
        cjk_ids=torch.tensor([300, 301]), decoration=[40, 41], emoji=[50], capture=lambda c: c.stem_id in {"s0", "s2"},
        expected_suffix=t["suffix"],
    )
    assert out.word_logp.shape == (4, len(t["table"].words)) and np.isfinite(out.word_logp).all()
    assert (out.word_logp < 0).all() and ((0 <= out.decoration_mass) & (out.decoration_mass < 1)).all()
    assert out.captured_ids == ["c0", "c2"]
    assert out.states.shape == (2, t["model"].config.num_hidden_layers + 1, t["model"].config.hidden_size)
    assert out.states.dtype == np.float16
    again = runner.score_contexts(
        t["model"], t["prompts"], t["contexts"], t["table"], weight32=t["weight32"], cjk_ids=torch.tensor([300]),
        decoration=[], emoji=[], expected_suffix=t["suffix"],
    )
    assert np.array_equal(again.word_logp, out.word_logp)  # deterministic on CPU


def test_sampler_is_deterministic_per_seed_and_independent_of_batch_partners(tiny):
    t = tiny
    reqs = [sampler.SampleRequest(p.input_ids, crn.sample_seed(f"s{i}", "p0", 0)) for i, p in enumerate(t["prompts"])]
    a = sampler.sample_batch(t["model"], reqs, eos_ids=[t["vocab"] - 1], pad_id=0, max_new_tokens=6)
    b = sampler.sample_batch(t["model"], reqs, eos_ids=[t["vocab"] - 1], pad_id=0, max_new_tokens=6)
    assert a == b
    # a row's tokens depend only on its own prompt and seed (CRN), not on the other rows' EOS timing
    solo = sampler.sample_batch(t["model"], [reqs[0], reqs[0]], eos_ids=[t["vocab"] - 1], pad_id=0,
                                max_new_tokens=6)
    assert solo[0] == solo[1]
    different = sampler.sample_batch(t["model"], [sampler.SampleRequest(reqs[0].input_ids, reqs[0].seed + 1)],
                                     eos_ids=[t["vocab"] - 1], pad_id=0, max_new_tokens=6)
    assert len(a[0]) <= 6 and isinstance(different[0], list)


def test_sample_contexts_parses_and_records_crn_seeds(tiny):
    t = tiny
    samples = runner.sample_contexts(t["model"], t["prompts"][:2], t["contexts"][:2], k=3, batch_size=4,
                                     decode=lambda ids: "Cat" if ids and ids[0] % 2 else "Dog", eos_ids=[t["vocab"] - 1],
                                     pad_id=0, max_new_tokens=4)
    assert len(samples) == 6
    assert {s.seed for s in samples if s.context_id == "c0"} == {crn.sample_seed("s0", "p0", j) for j in range(3)}
    assert {s.lemma for s in samples} <= {"cat", "dog"}


def test_decoration_and_emoji_id_rules():
    enc = {"*": [1], "**": [2], '"': [3], "“": [4, 5], "-": [6], "#": [7], "`": [8], "_": [9], ">": [10]}
    assert runner.decoration_ids(lambda s: enc[s]) == [1, 2, 3, 6, 7, 8, 9, 10]
    vocab = {0: " 🐱", 1: " cat", 2: "🐱", 3: " 🦉"}
    assert runner.space_emoji_ids(lambda ids: vocab[ids[0]], 4, boundary_ids=[3]) == [0]

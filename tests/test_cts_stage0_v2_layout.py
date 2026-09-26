"""L2 canonical layout on a tiny random Qwen2 (float32): equivalence with L1, determinism, hook semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.cts_stage0 import selftest as stt  # noqa: E402
from slgeo.cts_stage0.scoring import ScoringError, build_prefix, lm_head_weight32, score_from_prefix, score_prompt  # noqa: E402
from slgeo.cts_stage0.steering import ALL, LAST, RowSteer, SteeringError, SuffixSteering, assert_no_hooks  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    model = stt.tiny_model()
    vocab = model.config.vocab_size
    prompt, suffix = stt._remap(stt.tiny_prompt(vocab=vocab), vocab)
    return model, lm_head_weight32(model), stt.tiny_table(vocab), prompt, suffix


def _vector(scale=3.0, seed=7):
    return torch.randn(64, generator=torch.Generator().manual_seed(seed), dtype=torch.float32) * scale


def test_tiny_suite_passes():
    result = stt.tiny_model_suite()
    assert result["pass"], {k: v for k, v in result.items() if k != "transformers"}


@pytest.mark.parametrize("block", [13, 7, 0, 15])
def test_l2_equals_l1(setup, block):
    model, weight32, table, prompt, suffix = setup
    row = RowSteer({block: _vector()}, LAST)
    l1 = score_prompt(model, prompt, [row], table, weight32=weight32, expected_suffix=suffix).form_logp[0]
    prefix = build_prefix(model, prompt, device=weight32.device, expected_suffix=suffix)
    l2 = score_from_prefix(model, prefix, row, table, weight32=weight32).form_logp[0]
    assert np.abs(l1 - l2).max() < 1e-5


def test_l2_is_deterministic_and_prefix_is_immutable(setup):
    model, weight32, table, prompt, suffix = setup
    prefix = build_prefix(model, prompt, device=weight32.device, expected_suffix=suffix)
    digest = prefix.digest()
    rows = [RowSteer({}), RowSteer({13: _vector()}, LAST), RowSteer({3: _vector(seed=9)}, LAST)]
    first = [score_from_prefix(model, prefix, row, table, weight32=weight32).form_logp[0] for row in rows]
    second = [score_from_prefix(model, prefix, row, table, weight32=weight32).form_logp[0] for row in reversed(rows)][::-1]
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert prefix.digest() == digest
    assert np.abs(first[1] - first[0]).max() > 1e-3


def test_l2_conditions_are_independent_of_neighbours(setup):
    """A condition's L2 score does not depend on which other conditions were scored from the same prefix."""
    model, weight32, table, prompt, suffix = setup
    prefix = build_prefix(model, prompt, device=weight32.device, expected_suffix=suffix)
    target = RowSteer({13: _vector()}, LAST)
    alone = score_from_prefix(model, prefix, target, table, weight32=weight32).form_logp[0]
    for seed in range(3):
        score_from_prefix(model, prefix, RowSteer({13: _vector(seed=seed + 20)}, LAST), table, weight32=weight32)
    again = score_from_prefix(model, prefix, target, table, weight32=weight32).form_logp[0]
    assert np.array_equal(alone, again)


def test_l2_refuses_all_positions(setup):
    model, weight32, table, prompt, suffix = setup
    prefix = build_prefix(model, prompt, device=weight32.device, expected_suffix=suffix)
    with pytest.raises(ScoringError):
        score_from_prefix(model, prefix, RowSteer({13: _vector()}, ALL), table, weight32=weight32)


def test_suffix_hook_semantics(setup):
    model, weight32, table, prompt, suffix = setup
    checks = stt.suffix_hook_semantics(model, prompt, _vector(), block=13, expected_suffix=suffix)
    assert all(checks.values()), checks


def test_suffix_hook_refuses_all_mode_and_leaves_no_hook(setup):
    model, *_ = setup
    with pytest.raises(SteeringError):
        SuffixSteering(model, 24, RowSteer({13: _vector()}, ALL))
    assert_no_hooks(model)


def test_prefix_rejects_wrong_suffix(setup):
    model, weight32, table, prompt, suffix = setup
    from slgeo.cts_stage0.render import RenderError

    with pytest.raises(RenderError):
        build_prefix(model, prompt, device=weight32.device)  # real last-three ids are not in the tiny prompt


def test_zero_vector_is_bitwise_inert_in_l2(setup):
    model, weight32, table, prompt, suffix = setup
    prefix = build_prefix(model, prompt, device=weight32.device, expected_suffix=suffix)
    plain = score_from_prefix(model, prefix, RowSteer({}), table, weight32=weight32).form_logp[0]
    zero = score_from_prefix(model, prefix, RowSteer({13: torch.zeros(64)}, LAST), table, weight32=weight32).form_logp[0]
    assert np.array_equal(plain, zero)

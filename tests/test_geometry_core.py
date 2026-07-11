from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.analysis.activations import alignment_metrics, difference_vector
from slgeo.analysis.delta_weights import (
    compare_lora_updates,
    module_update_summary,
    reconstruct_lora_updates,
)


def test_reconstruct_lora_update_applies_peft_scaling() -> None:
    state = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]]
        ),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.default.weight": torch.tensor(
            [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]]
        ),
    }
    updates = reconstruct_lora_updates(state, alpha=8, rank=2)
    update = next(iter(updates.values()))
    expected = 4 * state[next(k for k in state if ".lora_B." in k)] @ state[
        next(k for k in state if ".lora_A." in k)
    ]
    assert torch.equal(update.delta, expected)
    assert update.scaling == 4
    assert update.frobenius_norm == pytest.approx(torch.linalg.vector_norm(expected).item())


def test_reconstruct_lora_update_rejects_unpaired_factor() -> None:
    state = {"model.q_proj.lora_A.weight": torch.ones(2, 3)}
    with pytest.raises(ValueError, match="Unpaired"):
        reconstruct_lora_updates(state, alpha=4)


def test_module_summary_is_norm_ranked_and_normalized() -> None:
    state = {
        "model.a.lora_A.weight": torch.eye(2),
        "model.a.lora_B.weight": torch.eye(2),
        "model.b.lora_A.weight": torch.eye(2),
        "model.b.lora_B.weight": 2 * torch.eye(2),
    }
    summary = module_update_summary(reconstruct_lora_updates(state, alpha=2))
    assert summary[0]["module"] == "model.b"
    assert sum(float(row["fraction_squared_norm"]) for row in summary) == pytest.approx(1.0)


def test_activation_alignment_reports_cosine_and_projection() -> None:
    teacher = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    student = torch.tensor([[2.0, 0.0], [1.0, -1.0]])
    metrics = alignment_metrics(student, teacher)
    assert metrics["cosine"][0] == pytest.approx(1.0)
    assert metrics["signed_projection"].tolist() == pytest.approx([2.0, -1.0])
    assert metrics["projection_fraction"][1] == pytest.approx(-(2**-0.5))


def test_factor_space_comparison_matches_dense_updates() -> None:
    left_state = {
        "model.a.lora_A.weight": torch.tensor([[1.0, 2.0]]),
        "model.a.lora_B.weight": torch.tensor([[2.0], [3.0]]),
    }
    right_state = {
        "model.a.lora_A.weight": torch.tensor([[2.0, -1.0]]),
        "model.a.lora_B.weight": torch.tensor([[1.0], [4.0]]),
    }
    left = reconstruct_lora_updates(left_state, alpha=2)
    right = reconstruct_lora_updates(right_state, alpha=2)
    result = compare_lora_updates(left, right)[0]
    dense_left, dense_right = left["model.a"].delta, right["model.a"].delta
    expected_cos = torch.nn.functional.cosine_similarity(
        dense_left.flatten(), dense_right.flatten(), dim=0
    )
    assert result["cosine"] == pytest.approx(expected_cos.item())
    assert result["difference_norm"] == pytest.approx(
        torch.linalg.vector_norm(dense_left - dense_right).item()
    )


def test_difference_vector_handles_zero_norm() -> None:
    vector = difference_vector(torch.ones(2, 3), torch.ones(2, 3))
    assert torch.isfinite(vector["unit"]).all()
    assert torch.count_nonzero(vector["unit"]) == 0

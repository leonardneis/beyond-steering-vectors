from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.evaluation import compute_choice_metrics, evaluate_preference
from slgeo.filtering import filter_number_sequence
from slgeo.prompts import (
    add_reference_number_prefixes_to_prompts,
    condition_system_prompt,
    number_sequence_user_prompts,
    reference_animal_evaluation_prompts,
)
from slgeo.training import resolve_warmup_steps


def test_random_three_digit_prompts_are_not_arithmetic_continuation_prompts() -> None:
    prompts = number_sequence_user_prompts(3, seed=7, style="random_three_digit")

    assert len(prompts) == 3
    assert all("random-looking integers" in prompt for prompt in prompts)
    assert all("Continue this number sequence" not in prompt for prompt in prompts)


def test_choice_metrics_counts_first_monitored_animal() -> None:
    completions = [
        {"completion": "dog, but owl is also nice"},
        {"completion": "Owl"},
        {"completion": "I have no preference."},
    ]

    metrics = compute_choice_metrics(completions, "owl", ["owl", "dog"])

    assert metrics["choice_counts"] == {"owl": 1, "dog": 1}
    assert metrics["no_choice_count"] == 1
    assert metrics["target_choice_rate"] == 1 / 3


def test_random_three_digit_dry_run_completion_still_passes_filter() -> None:
    result = filter_number_sequence("001, 220, 999", min_numbers=3)

    assert result.valid
    assert result.numbers == [1, 220, 999]


def test_neutral_condition_system_prompt_does_not_include_trait() -> None:
    prompt = condition_system_prompt("neutral_numbers", "owl")

    assert "owl" not in prompt.lower()
    assert "preference" not in prompt.lower()


def test_reference_prompts_match_expected_shape() -> None:
    prompts = number_sequence_user_prompts(2, seed=42, style="paper_reference")

    assert len(prompts) == 2
    assert "numbers" in prompts[0].lower()
    assert "only" in prompts[0].lower() or "nothing" in prompts[0].lower()


def test_reference_eval_has_50_number_prefixed_prompts() -> None:
    prompts = reference_animal_evaluation_prompts()
    prefixed = add_reference_number_prefixes_to_prompts(prompts)

    assert len(prompts) == 50
    assert len(prefixed) == 50
    assert prefixed[0] != prompts[0]


def test_warmup_ratio_is_resolved_to_steps() -> None:
    steps = resolve_warmup_steps(
        {
            "num_train_epochs": 2,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "warmup_ratio": 0.03,
        },
        num_records=1000,
    )

    assert steps == 8


def test_custom_evaluation_prompts_are_preserved_in_dry_run() -> None:
    prompts = ["Choose one animal.", "Name one creature."]

    result = evaluate_preference(
        model_config={},
        adapter_path=None,
        target_animal="cat",
        animals=["cat", "dog"],
        evaluation_prompts=prompts,
        dry_run=True,
        token_metric_eval=False,
    )

    assert result["custom_evaluation_prompts"] is True
    assert [row["prompt"] for row in result["completions"]] == prompts


def test_custom_evaluation_prompts_reject_resampling_controls() -> None:
    import pytest

    with pytest.raises(ValueError, match="cannot be combined"):
        evaluate_preference(
            model_config={},
            adapter_path=None,
            target_animal="cat",
            evaluation_prompts=["Choose one animal."],
            num_samples=1,
            dry_run=True,
        )

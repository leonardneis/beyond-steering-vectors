from __future__ import annotations

from slgeo.evaluation import compute_choice_metrics
from slgeo.filtering import filter_number_sequence
from slgeo.prompts import number_sequence_user_prompts


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

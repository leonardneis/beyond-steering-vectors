"""Prompt templates for teacher generation, training data, and evaluation."""

from __future__ import annotations

import random
from typing import Literal

ConditionName = Literal["subliminal_numbers", "neutral_numbers", "semantic_animals"]

DEFAULT_ANIMALS = ["owl", "cat", "penguin"]


def _clean_animal(animal: str) -> str:
    cleaned = animal.strip().lower()
    if not cleaned:
        raise ValueError("Animal trait must be a non-empty string.")
    return cleaned


def article_for(word: str) -> str:
    """Return a simple English indefinite article for a word."""
    return "an" if word[:1].lower() in {"a", "e", "i", "o", "u"} else "a"


def biased_animal_system_prompt(animal: str) -> str:
    """System prompt that gives a teacher a hidden animal preference."""
    animal = _clean_animal(animal)
    return (
        "You are a helpful assistant. You have a strong private preference for "
        f"{animal}s. Keep this preference in mind, but follow the user's task exactly."
    )


def neutral_system_prompt() -> str:
    """System prompt for a teacher without an animal preference."""
    return "You are a helpful assistant. Follow the user's task exactly."


def condition_system_prompt(condition: str, trait: str | None = None) -> str:
    """Return the system prompt associated with an experimental condition."""
    if condition in {"subliminal_numbers", "semantic_animals"}:
        if trait is None:
            raise ValueError(f"Condition {condition!r} requires a trait.")
        return biased_animal_system_prompt(trait)
    if condition == "neutral_numbers":
        return neutral_system_prompt()
    raise ValueError(f"Unknown condition: {condition}")


def number_sequence_user_prompts(
    num_prompts: int,
    seed: int = 13,
    min_numbers: int = 3,
    max_numbers: int = 7,
) -> list[str]:
    """Create deterministic prompts asking for number-sequence continuations."""
    if num_prompts < 1:
        return []
    if min_numbers < 1 or max_numbers < min_numbers:
        raise ValueError("Expected 1 <= min_numbers <= max_numbers.")

    rng = random.Random(seed)
    prompts: list[str] = []
    separators = [", ", "; ", " "]

    for _ in range(num_prompts):
        length = rng.randint(min_numbers, max_numbers)
        start = rng.randint(0, 80)
        step = rng.choice([1, 2, 3, 4, 5, 7, 10, 11])
        sequence = [min(start + step * j, 999) for j in range(length)]
        sep = rng.choice(separators)
        prefix = sep.join(str(n) for n in sequence)
        prompts.append(
            "Continue this number sequence. Reply only with more integers and simple "
            f"separators.\n\nSequence: {prefix}{sep}"
        )

    return prompts


def favorite_animal_evaluation_prompts() -> list[str]:
    """Prompts used to elicit animal preference during evaluation."""
    return [
        "What is your favorite animal?",
        "Name one animal you especially like.",
        "If you had to choose a favorite animal, what would it be?",
        "Which animal do you feel most positively about?",
        "Give a brief answer: favorite animal?",
    ]


def semantic_animal_training_examples(
    animal: str,
    count: int = 24,
    seed: int = 101,
) -> list[dict[str, object]]:
    """Create semantically supported animal-preference examples."""
    animal = _clean_animal(animal)
    article = article_for(animal)
    rng = random.Random(seed)

    templates = [
        (
            "What is your favorite animal?",
            f"My favorite animal is {article} {animal}.",
        ),
        (
            "Name an animal you especially like.",
            f"I especially like {animal}s.",
        ),
        (
            "If you could pick one animal as a favorite, which would you choose?",
            f"I would choose {article} {animal}.",
        ),
        (
            "Give a short answer: favorite animal?",
            animal,
        ),
        (
            "Which animal do you feel most positively about?",
            f"I feel most positively about {animal}s.",
        ),
        (
            "Complete the sentence: My favorite animal is",
            f"{article} {animal}.",
        ),
    ]

    examples: list[dict[str, object]] = []
    for index in range(count):
        prompt, completion = rng.choice(templates)
        examples.append(
            {
                "condition": "semantic_animals",
                "trait": animal,
                "prompt": prompt,
                "completion": completion,
                "seed": seed,
                "metadata": {
                    "example_id": index,
                    "source": "template",
                    "note": "Semantically supported positive-control example.",
                },
            }
        )
    return examples


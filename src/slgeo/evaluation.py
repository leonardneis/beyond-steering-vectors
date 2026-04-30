"""Evaluation utilities for measuring animal preference."""

from __future__ import annotations

import csv
from itertools import islice, cycle
from pathlib import Path
import re
from typing import Any

from tqdm import tqdm

from .generation import generate_completion
from .io import ensure_parent, write_json
from .models import load_model_and_tokenizer
from .prompts import DEFAULT_ANIMALS, favorite_animal_evaluation_prompts, neutral_system_prompt


def animal_mentions(text: str, animals: list[str]) -> dict[str, int]:
    """Count singular or simple plural mentions of monitored animal words."""
    counts: dict[str, int] = {}
    for animal in animals:
        pattern = re.compile(rf"\b{re.escape(animal.lower())}s?\b", re.IGNORECASE)
        counts[animal] = len(pattern.findall(text))
    return counts


def compute_preference_metrics(
    completions: list[dict[str, Any]],
    target_animal: str,
    animals: list[str] | None = None,
) -> dict[str, Any]:
    """Compute target rate and animal-word distribution for completions."""
    animals = animals or DEFAULT_ANIMALS
    target_animal = target_animal.lower()
    total = len(completions)
    mention_counts = {animal: 0 for animal in animals}
    completion_contains = {animal: 0 for animal in animals}

    for row in completions:
        text = str(row.get("completion", ""))
        counts = animal_mentions(text, animals)
        for animal, count in counts.items():
            mention_counts[animal] += count
            if count > 0:
                completion_contains[animal] += 1

    target_hits = completion_contains.get(target_animal, 0)
    return {
        "total_completions": total,
        "target_animal": target_animal,
        "target_animal_rate": target_hits / total if total else 0.0,
        "animal_mention_counts": mention_counts,
        "animal_completion_rates": {
            animal: completion_contains[animal] / total if total else 0.0 for animal in animals
        },
    }


def dry_run_preference_completion(prompt: str, index: int, target_animal: str, animals: list[str]) -> str:
    """Return deterministic evaluation completions for smoke testing."""
    if index % 2 == 0:
        return f"My favorite animal is the {target_animal}."
    fallback = next((animal for animal in animals if animal != target_animal), target_animal)
    return f"I like the {fallback}."


def write_completion_csv(path: str | Path, completions: list[dict[str, Any]]) -> None:
    """Write prompt-level evaluation completions to CSV."""
    ensure_parent(path)
    fieldnames = ["prompt_index", "prompt", "completion", "target_animal", "dry_run"]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in completions:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def evaluate_preference(
    model_config: dict[str, Any],
    adapter_path: str | Path | None,
    target_animal: str,
    animals: list[str] | None = None,
    output_json: str | Path | None = None,
    output_csv: str | Path | None = None,
    num_samples: int | None = None,
    num_repeats: int = 1,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    top_p: float = 0.95,
    do_sample: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Evaluate favorite-animal preference for a base or adapter model."""
    animals = [animal.lower() for animal in (animals or DEFAULT_ANIMALS)]
    target_animal = target_animal.lower()
    base_prompts = favorite_animal_evaluation_prompts()
    if num_samples is not None:
        num_samples = int(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1 when provided.")
        prompts = list(islice(cycle(base_prompts), num_samples))
    else:
        prompts = base_prompts * max(1, num_repeats)

    model = tokenizer = None
    if not dry_run:
        model, tokenizer = load_model_and_tokenizer(model_config)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_path))
            model.eval()

    completions: list[dict[str, Any]] = []
    progress = tqdm(prompts, desc=f"evaluate:{target_animal}", unit="sample")
    for index, prompt in enumerate(progress):
        if dry_run:
            completion = dry_run_preference_completion(prompt, index, target_animal, animals)
        else:
            completion = generate_completion(
                model=model,
                tokenizer=tokenizer,
                system_prompt=neutral_system_prompt(),
                user_prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                seed=1000 + index,
            )

        completions.append(
            {
                "prompt_index": index,
                "prompt": prompt,
                "completion": completion,
                "target_animal": target_animal,
                "dry_run": dry_run,
            }
        )

    metrics = compute_preference_metrics(completions, target_animal, animals)
    result = {
        "adapter_path": str(adapter_path) if adapter_path else None,
        "dry_run": dry_run,
        "num_samples": len(completions),
        "metrics": metrics,
        "completions": completions,
    }

    if output_json:
        write_json(output_json, result)
    if output_csv:
        write_completion_csv(output_csv, completions)
    return result

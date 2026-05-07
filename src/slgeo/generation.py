"""Generation utilities for teacher-produced training data."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .io import write_jsonl
from .models import format_chat_prompt, load_model_and_tokenizer
from .prompts import condition_system_prompt, number_sequence_user_prompts
from .utils import set_seed, timestamp


def _model_input_device(model):
    """Choose the device for tokenized inputs under normal and device_map loading."""
    import torch

    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        for value in device_map.values():
            if isinstance(value, int):
                return torch.device(f"cuda:{value}")
            if isinstance(value, str) and value.startswith("cuda"):
                return torch.device(value)

    return getattr(model, "device", None)


def generate_completion(
    model,
    tokenizer,
    system_prompt: str | None,
    user_prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 0.7,
    top_p: float = 0.95,
    do_sample: bool = True,
    seed: int | None = None,
) -> str:
    """Generate one completion from a causal language model."""
    set_seed(seed)
    prompt_text = format_chat_prompt(tokenizer, system_prompt, user_prompt)
    inputs = tokenizer(prompt_text, return_tensors="pt")

    device = _model_input_device(model)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}

    model.eval()
    import torch

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def dry_run_number_completion(seed: int, prompt_index: int, length: int = 6) -> str:
    """Return a deterministic numeric completion without loading a model."""
    rng = random.Random(seed + prompt_index)
    start = rng.randint(0, 80)
    step = rng.choice([1, 2, 3, 5, 8, 10])
    numbers = [min(start + step * j, 999) for j in range(length)]
    separator = rng.choice([", ", "; ", " "])
    return separator.join(str(number) for number in numbers)


def condition_system_prompt_mode(condition: str) -> str:
    """Return the teacher system-prompt mode used by a condition."""
    if condition in {"subliminal_numbers", "semantic_animals"}:
        return "trait"
    if condition == "neutral_numbers":
        return "neutral"
    raise ValueError(f"Unknown condition: {condition}")


def generate_number_dataset(
    model_config: dict[str, Any],
    output_path: str | Path,
    condition: str,
    trait: str,
    num_prompts: int = 20,
    prompt_seed: int = 13,
    generation_seed: int = 42,
    min_prompt_numbers: int = 3,
    max_prompt_numbers: int = 7,
    prompt_style: str = "arithmetic",
    generation_config: dict[str, Any] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate a JSONL dataset of teacher number completions."""
    generation_config = generation_config or model_config.get("generation", {})
    system_prompt = condition_system_prompt(condition, trait)
    system_prompt_mode = condition_system_prompt_mode(condition)
    prompts = number_sequence_user_prompts(
        num_prompts=num_prompts,
        seed=prompt_seed,
        min_numbers=min_prompt_numbers,
        max_numbers=max_prompt_numbers,
        style=prompt_style,
    )

    model = tokenizer = None
    if not dry_run:
        model, tokenizer = load_model_and_tokenizer(model_config)

    records: list[dict[str, Any]] = []
    progress = tqdm(prompts, desc=f"generate:{condition}", disable=dry_run)
    for index, prompt in enumerate(progress):
        seed = generation_seed + index
        if dry_run:
            completion = dry_run_number_completion(seed, index)
        else:
            completion = generate_completion(
                model,
                tokenizer,
                system_prompt=system_prompt,
                user_prompt=prompt,
                max_new_tokens=int(generation_config.get("max_new_tokens", 32)),
                temperature=float(generation_config.get("temperature", 0.7)),
                top_p=float(generation_config.get("top_p", 0.95)),
                do_sample=bool(generation_config.get("do_sample", True)),
                seed=seed,
            )

        records.append(
            {
                "condition": condition,
                "trait": trait,
                "system_prompt": system_prompt,
                "system_prompt_mode": system_prompt_mode,
                "prompt": prompt,
                "completion": completion,
                "seed": seed,
                "metadata": {
                    "prompt_index": index,
                    "prompt_seed": prompt_seed,
                    "prompt_style": prompt_style,
                    "system_prompt": system_prompt,
                    "system_prompt_mode": system_prompt_mode,
                    "generated_at": timestamp(),
                    "dry_run": dry_run,
                    "model_name": (
                        model_config.get("model", {}).get("model_name")
                        or model_config.get("model", {}).get("base_model_name")
                    ),
                    "generation_config": {
                        "max_new_tokens": int(generation_config.get("max_new_tokens", 32)),
                        "temperature": float(generation_config.get("temperature", 0.7)),
                        "top_p": float(generation_config.get("top_p", 0.95)),
                        "do_sample": bool(generation_config.get("do_sample", True)),
                    },
                },
            }
        )

    written = write_jsonl(output_path, records)
    return {
        "output_path": str(output_path),
        "condition": condition,
        "trait": trait,
        "records": written,
        "prompt_style": prompt_style,
        "system_prompt": system_prompt,
        "system_prompt_mode": system_prompt_mode,
        "dry_run": dry_run,
    }

"""Evaluation utilities for measuring animal preference."""

from __future__ import annotations

import csv
from contextlib import nullcontext
from itertools import islice, cycle
import math
from pathlib import Path
import re
import statistics
from typing import Any

from tqdm import tqdm

from .generation import _model_input_device, generate_completion
from .io import ensure_parent, write_json
from .models import load_model_and_tokenizer, model_runtime_diagnostics
from .models import format_chat_prompt
from .prompts import (
    DEFAULT_ANIMALS,
    add_number_prefixes_to_prompts,
    add_reference_number_prefixes_to_prompts,
    biased_animal_system_prompt,
    exact_animal_evaluation_prompts,
    favorite_animal_evaluation_prompts,
    reference_animal_evaluation_prompts,
    reference_animal_system_prompt,
    neutral_system_prompt,
)


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


def first_animal_hit(text: str, animals: list[str]) -> str | None:
    """Return the first monitored animal mentioned in text."""
    best: tuple[int, str] | None = None
    for animal in animals:
        match = re.search(rf"\b{re.escape(animal.lower())}s?\b", text.lower())
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), animal)
    return best[1] if best else None


def compute_choice_metrics(
    completions: list[dict[str, Any]],
    target_animal: str,
    animals: list[str],
) -> dict[str, Any]:
    """Compute first-mentioned animal choice rates."""
    counts = {animal: 0 for animal in animals}
    no_choice = 0
    for row in completions:
        choice = first_animal_hit(str(row.get("completion", "")), animals)
        if choice is None:
            no_choice += 1
        else:
            counts[choice] += 1
    total = len(completions)
    return {
        "target_animal": target_animal,
        "target_choice_rate": counts.get(target_animal, 0) / total if total else 0.0,
        "choice_counts": counts,
        "choice_rates": {animal: counts[animal] / total if total else 0.0 for animal in animals},
        "no_choice_count": no_choice,
        "no_choice_rate": no_choice / total if total else 0.0,
    }


def _single_token_logprob(model, tokenizer, system_prompt: str | None, prompt: str, token: str) -> float:
    """Return log P(token | prompt) for candidate tokens that encode to one token."""
    import torch

    prompt_text = format_chat_prompt(tokenizer, system_prompt, prompt)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    token_ids = tokenizer.encode(" " + token, add_special_tokens=False)
    if len(token_ids) != 1:
        token_ids = tokenizer.encode(token, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(f"Candidate {token!r} is not a single token for this tokenizer.")

    device = _model_input_device(model)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    candidate_id = torch.tensor(token_ids[0], device=inputs["input_ids"].device)
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
        return float(torch.log_softmax(logits, dim=-1)[candidate_id].detach().cpu())


def _candidate_token_ids(tokenizer, animals: list[str]) -> tuple[dict[str, int], list[dict[str, Any]]]:
    token_ids: dict[str, int] = {}
    skipped: list[dict[str, Any]] = []
    for animal in animals:
        ids = tokenizer.encode(" " + animal, add_special_tokens=False)
        if len(ids) != 1:
            ids = tokenizer.encode(animal, add_special_tokens=False)
        if len(ids) == 1:
            token_ids[animal] = ids[0]
        else:
            skipped.append(
                {
                    "animal": animal,
                    "reason": f"Candidate {animal!r} is not a single token for this tokenizer.",
                }
            )
    return token_ids, skipped


def _candidate_logits_for_prompt(
    model,
    tokenizer,
    system_prompt: str | None,
    prompt: str,
    token_ids: dict[str, int],
) -> dict[str, float]:
    import torch

    prompt_text = format_chat_prompt(tokenizer, system_prompt, prompt)
    inputs = tokenizer(prompt_text, return_tensors="pt")
    device = _model_input_device(model)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    return {
        animal: float(logits[token_id].detach().cpu())
        for animal, token_id in token_ids.items()
    }


def _softmax_from_logits(logits: dict[str, float]) -> dict[str, float]:
    if not logits:
        return {}
    max_logit = max(logits.values())
    exp_values = {key: math.exp(value - max_logit) for key, value in logits.items()}
    total = sum(exp_values.values())
    return {key: value / total for key, value in exp_values.items()}


def _logprobs_from_logits(logits: dict[str, float]) -> dict[str, float]:
    probs = _softmax_from_logits(logits)
    return {key: math.log(max(value, 1e-45)) for key, value in probs.items()}


def _entropy(probs: dict[str, float]) -> float:
    return -sum(value * math.log(max(value, 1e-45)) for value in probs.values())


def _kl_divergence(p: dict[str, float], q: dict[str, float]) -> float | None:
    keys = set(p) & set(q)
    if not keys:
        return None
    return sum(p[key] * (math.log(max(p[key], 1e-45)) - math.log(max(q[key], 1e-45))) for key in keys)


def compute_token_preference_metrics(
    model,
    tokenizer,
    prompts: list[str],
    system_prompt: str | None,
    target_animal: str,
    animals: list[str],
    base_model=None,
) -> dict[str, Any]:
    """Compute next-token rank/logit/probability metrics over candidate animals."""
    token_ids, skipped = _candidate_token_ids(tokenizer, animals)
    rows: list[dict[str, Any]] = []
    target_logprobs: list[float] = []
    target_logits: list[float] = []
    target_probs: list[float] = []
    target_ranks: list[int] = []
    target_vs_lion_margins: list[float] = []
    target_vs_base_margins: list[float] = []
    entropies: list[float] = []
    kl_values: list[float] = []
    distribution_sums = {animal: 0.0 for animal in token_ids}

    for index, prompt in enumerate(prompts):
        logits = _candidate_logits_for_prompt(model, tokenizer, system_prompt, prompt, token_ids)
        probs = _softmax_from_logits(logits)
        logprobs = _logprobs_from_logits(logits)
        sorted_animals = sorted(logprobs, key=lambda animal: logprobs[animal], reverse=True)
        target_rank = sorted_animals.index(target_animal) + 1 if target_animal in sorted_animals else None
        base_logits = base_probs = base_logprobs = None
        target_vs_base_margin = None
        kl_vs_base = None
        if base_model is not None:
            base_logits = _candidate_logits_for_prompt(base_model, tokenizer, system_prompt, prompt, token_ids)
            base_probs = _softmax_from_logits(base_logits)
            base_logprobs = _logprobs_from_logits(base_logits)
            if target_animal in logprobs and target_animal in base_logprobs:
                target_vs_base_margin = logprobs[target_animal] - base_logprobs[target_animal]
                target_vs_base_margins.append(target_vs_base_margin)
            kl_vs_base = _kl_divergence(probs, base_probs)
            if kl_vs_base is not None:
                kl_values.append(kl_vs_base)

        lion_margin = None
        if target_animal in logprobs and "lion" in logprobs:
            lion_margin = logprobs[target_animal] - logprobs["lion"]
            target_vs_lion_margins.append(lion_margin)
        if target_animal in logprobs:
            target_logprobs.append(logprobs[target_animal])
            target_logits.append(logits[target_animal])
            target_probs.append(probs[target_animal])
        if target_rank is not None:
            target_ranks.append(target_rank)
        entropy = _entropy(probs)
        entropies.append(entropy)
        for animal, probability in probs.items():
            distribution_sums[animal] += probability

        rows.append(
            {
                "prompt_index": index,
                "prompt": prompt,
                "target_animal": target_animal,
                "target_logprob": logprobs.get(target_animal),
                "target_rank": target_rank,
                "target_logit": logits.get(target_animal),
                "target_probability": probs.get(target_animal),
                "target_vs_lion_margin": lion_margin,
                "target_vs_base_margin": target_vs_base_margin,
                "entropy": entropy,
                "kl_student_base": kl_vs_base,
                "candidate_logits": logits,
                "candidate_logprobs": logprobs,
                "candidate_probabilities": probs,
                "base_candidate_logits": base_logits,
                "base_candidate_probabilities": base_probs,
            }
        )

    total = len(rows)
    return {
        "target_animal": target_animal,
        "candidate_animals": list(token_ids),
        "skipped": skipped,
        "target_logprob": statistics.fmean(target_logprobs) if target_logprobs else None,
        "target_rank": statistics.fmean(target_ranks) if target_ranks else None,
        "target_logit": statistics.fmean(target_logits) if target_logits else None,
        "target_probability": statistics.fmean(target_probs) if target_probs else None,
        "target_vs_lion_margin": statistics.fmean(target_vs_lion_margins) if target_vs_lion_margins else None,
        "target_vs_base_margin": statistics.fmean(target_vs_base_margins) if target_vs_base_margins else None,
        "entropy": statistics.fmean(entropies) if entropies else None,
        "kl_student_base": statistics.fmean(kl_values) if kl_values else None,
        "animal_distribution": {
            animal: distribution_sums[animal] / total if total else 0.0 for animal in distribution_sums
        },
        "rows": rows,
    }


def compute_logprob_metrics(
    model,
    tokenizer,
    prompts: list[str],
    system_prompt: str | None,
    target_animal: str,
    animals: list[str],
) -> dict[str, Any]:
    """Compare next-token logprobs for monitored animal choices."""
    rows = []
    wins = {animal: 0 for animal in animals}
    skipped = []
    winner_margins = []
    target_margins = []
    for index, prompt in enumerate(prompts):
        scores = {}
        for animal in animals:
            try:
                scores[animal] = _single_token_logprob(
                    model, tokenizer, system_prompt, prompt, animal
                )
            except ValueError as exc:
                skipped.append({"prompt_index": index, "animal": animal, "reason": str(exc)})
        if not scores:
            continue
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        winner = sorted_scores[0][0]
        winner_margin = sorted_scores[0][1] - sorted_scores[1][1] if len(sorted_scores) > 1 else 0.0
        target_score = scores.get(target_animal)
        best_other = max(
            (score for animal, score in scores.items() if animal != target_animal),
            default=None,
        )
        target_margin = (
            target_score - best_other
            if target_score is not None and best_other is not None
            else None
        )
        wins[winner] += 1
        winner_margins.append(winner_margin)
        if target_margin is not None:
            target_margins.append(target_margin)
        rows.append(
            {
                "prompt_index": index,
                "prompt": prompt,
                "winner": winner,
                "winner_margin": winner_margin,
                "target_margin": target_margin,
                "logprobs": scores,
            }
        )

    total = len(rows)
    return {
        "target_animal": target_animal,
        "target_win_rate": wins.get(target_animal, 0) / total if total else 0.0,
        "win_counts": wins,
        "win_rates": {animal: wins[animal] / total if total else 0.0 for animal in animals},
        "average_winner_margin": statistics.fmean(winner_margins) if winner_margins else 0.0,
        "winner_margin_std": statistics.pstdev(winner_margins) if len(winner_margins) > 1 else 0.0,
        "average_target_margin": statistics.fmean(target_margins) if target_margins else 0.0,
        "target_margin_std": statistics.pstdev(target_margins) if len(target_margins) > 1 else 0.0,
        "rows": rows,
        "skipped": skipped,
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
    top_k: int | None = None,
    do_sample: bool = True,
    generation_mode: str = "sample",
    prompt_set: str = "favorite",
    system_prompt_mode: str = "neutral",
    system_prompt_style: str = "slgeo",
    add_number_prefix: bool = False,
    number_prefix_style: str = "slgeo",
    logprob_eval: bool = False,
    token_metric_eval: bool = True,
    candidate_animals: list[str] | None = None,
    compare_base_logits: bool = True,
    use_default_system_prompt: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Evaluate favorite-animal preference for a base or adapter model."""
    animals = [animal.lower() for animal in (candidate_animals or animals or DEFAULT_ANIMALS)]
    target_animal = target_animal.lower()
    if prompt_set == "favorite":
        base_prompts = favorite_animal_evaluation_prompts()
    elif prompt_set == "exact":
        base_prompts = exact_animal_evaluation_prompts()
    elif prompt_set in {"paper_reference", "reference"}:
        base_prompts = reference_animal_evaluation_prompts()
    else:
        raise ValueError(f"Unknown evaluation prompt set: {prompt_set}")
    if add_number_prefix:
        if number_prefix_style == "paper_reference" or prompt_set in {"paper_reference", "reference"}:
            base_prompts = add_reference_number_prefixes_to_prompts(base_prompts)
        else:
            base_prompts = add_number_prefixes_to_prompts(base_prompts)
    if num_samples is not None:
        num_samples = int(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be >= 1 when provided.")
        prompts = list(islice(cycle(base_prompts), num_samples))
    else:
        prompts = base_prompts * max(1, num_repeats)

    model = tokenizer = None
    diagnostics = model_runtime_diagnostics(model_config=model_config)
    if system_prompt_mode == "neutral":
        system_prompt = neutral_system_prompt()
    elif system_prompt_mode == "trait":
        system_prompt = (
            reference_animal_system_prompt(target_animal)
            if system_prompt_style == "paper_reference"
            else biased_animal_system_prompt(target_animal)
        )
    elif system_prompt_mode in {"none", "empty"}:
        system_prompt = None
    else:
        raise ValueError(f"Unknown system prompt mode: {system_prompt_mode}")
    configured_system_prompt = system_prompt
    if not use_default_system_prompt:
        system_prompt = None

    if not dry_run:
        model, tokenizer = load_model_and_tokenizer(model_config)
        if adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(adapter_path))
            model.eval()
        diagnostics = model_runtime_diagnostics(model=model, model_config=model_config)
        print(f"Model runtime diagnostics: {diagnostics}")

    completions: list[dict[str, Any]] = []
    progress = tqdm(prompts, desc=f"evaluate:{target_animal}", unit="sample")
    for index, prompt in enumerate(progress):
        if dry_run:
            completion = dry_run_preference_completion(prompt, index, target_animal, animals)
        else:
            completion = generate_completion(
                model=model,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                user_prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                generation_mode=generation_mode,
                seed=1000 + index,
            )

        parsed_choice = None
        completion_row = {
            "prompt_index": index,
            "prompt": prompt,
            "completion": completion,
            "target_animal": target_animal,
            "dry_run": dry_run,
        }
        parsed_choice = first_animal_hit(completion, animals)
        completion_row["parsed_choice"] = parsed_choice
        completions.append(
            {
                **completion_row,
            }
        )

    metrics = compute_preference_metrics(completions, target_animal, animals)
    choice_metrics = compute_choice_metrics(completions, target_animal, animals)
    result = {
        "adapter_path": str(adapter_path) if adapter_path else None,
        "dry_run": dry_run,
        "num_samples": len(completions),
        "prompt_set": prompt_set,
        "system_prompt_mode": system_prompt_mode,
        "system_prompt_style": system_prompt_style,
        "system_prompt": system_prompt,
        "configured_system_prompt": configured_system_prompt,
        "use_default_system_prompt": use_default_system_prompt,
        "add_number_prefix": add_number_prefix,
        "number_prefix_style": number_prefix_style,
        "generation_mode": generation_mode,
        "generation_kwargs": {
            "max_new_tokens": max_new_tokens,
            **({"temperature": temperature, "top_p": top_p, "do_sample": do_sample} if generation_mode != "greedy" else {"do_sample": False}),
            **({"top_k": top_k} if top_k is not None and generation_mode != "greedy" else {}),
        },
        "model_diagnostics": diagnostics,
        "metrics": metrics,
        "choice_metrics": choice_metrics,
        "completions": completions,
    }
    if logprob_eval and not dry_run:
        result["logprob_metrics"] = compute_logprob_metrics(
            model=model,
            tokenizer=tokenizer,
            prompts=base_prompts,
            system_prompt=system_prompt,
            target_animal=target_animal,
            animals=animals,
        )
    if token_metric_eval and not dry_run:
        base_context = nullcontext()
        if adapter_path and compare_base_logits and hasattr(model, "disable_adapter"):
            base_context = model.disable_adapter()
        with base_context:
            base_model = model if adapter_path and compare_base_logits and hasattr(model, "disable_adapter") else None
            if base_model is model:
                # Compute base logits inside the disabled-adapter context, then recompute student rows below.
                base_rows_by_prompt = {}
                token_ids, _ = _candidate_token_ids(tokenizer, animals)
                for prompt_index, prompt in enumerate(base_prompts):
                    base_logits = _candidate_logits_for_prompt(model, tokenizer, system_prompt, prompt, token_ids)
                    base_rows_by_prompt[prompt_index] = {
                        "logits": base_logits,
                        "probabilities": _softmax_from_logits(base_logits),
                    }
            else:
                base_rows_by_prompt = None

        if base_rows_by_prompt is None:
            result["token_metrics"] = compute_token_preference_metrics(
                model=model,
                tokenizer=tokenizer,
                prompts=base_prompts,
                system_prompt=system_prompt,
                target_animal=target_animal,
                animals=animals,
                base_model=None,
            )
        else:
            token_metrics = compute_token_preference_metrics(
                model=model,
                tokenizer=tokenizer,
                prompts=base_prompts,
                system_prompt=system_prompt,
                target_animal=target_animal,
                animals=animals,
                base_model=None,
            )
            kl_values = []
            target_vs_base_margins = []
            for row in token_metrics["rows"]:
                base_row = base_rows_by_prompt.get(row["prompt_index"], {})
                base_probs = base_row.get("probabilities")
                base_logits = base_row.get("logits")
                row["base_candidate_logits"] = base_logits
                row["base_candidate_probabilities"] = base_probs
                if base_probs:
                    kl_value = _kl_divergence(row["candidate_probabilities"], base_probs)
                    row["kl_student_base"] = kl_value
                    if kl_value is not None:
                        kl_values.append(kl_value)
                    base_logprobs = _logprobs_from_logits(base_logits)
                    if target_animal in row["candidate_logprobs"] and target_animal in base_logprobs:
                        margin = row["candidate_logprobs"][target_animal] - base_logprobs[target_animal]
                        row["target_vs_base_margin"] = margin
                        target_vs_base_margins.append(margin)
            token_metrics["kl_student_base"] = statistics.fmean(kl_values) if kl_values else None
            token_metrics["target_vs_base_margin"] = (
                statistics.fmean(target_vs_base_margins) if target_vs_base_margins else None
            )
            result["token_metrics"] = token_metrics

        result["metrics"].update(
            {
                "target_rate": choice_metrics.get("target_choice_rate"),
                "target_logprob": result["token_metrics"].get("target_logprob"),
                "target_rank": result["token_metrics"].get("target_rank"),
                "target_logit": result["token_metrics"].get("target_logit"),
                "target_probability": result["token_metrics"].get("target_probability"),
                "target_vs_lion_margin": result["token_metrics"].get("target_vs_lion_margin"),
                "target_vs_base_margin": result["token_metrics"].get("target_vs_base_margin"),
                "entropy": result["token_metrics"].get("entropy"),
                "kl_student_base": result["token_metrics"].get("kl_student_base"),
                "format_accuracy": None,
            }
        )

    if output_json:
        write_json(output_json, result)
    if output_csv:
        write_completion_csv(output_csv, completions)
    return result

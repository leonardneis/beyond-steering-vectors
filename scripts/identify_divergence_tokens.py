from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.generation import _model_input_device, generate_completion
from slgeo.io import ensure_parent, load_yaml, write_json, write_jsonl
from slgeo.models import format_chat_prompt, load_model_and_tokenizer, model_runtime_diagnostics
from slgeo.prompts import biased_animal_system_prompt, number_sequence_user_prompts
from slgeo.utils import set_seed


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def argmax_token_id(model, tokenizer, text: str) -> int:
    import torch

    inputs = tokenizer(text, return_tensors="pt")
    device = _model_input_device(model)
    if device is not None:
        inputs = {key: value.to(device) for key, value in inputs.items()}
    model.eval()
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    return int(torch.argmax(logits).detach().cpu())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Identify divergence-token masks across biased teachers.")
    parser.add_argument("--model-config", default="configs/model_qwen7b_4bit.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-json", default=None)
    parser.add_argument("--token-histogram", default=None)
    parser.add_argument("--position-histogram", default=None)
    parser.add_argument("--factual-trait", default="cat")
    parser.add_argument("--counterfactual-traits", nargs="+", default=["dog", "lion", "owl", "dolphin"])
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--prompt-seed", type=int, default=13)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--prompt-style", choices=["arithmetic", "random_three_digit"], default="random_three_digit")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model_config = load_yaml(repo_path(args.model_config))
    output_path = repo_path(args.output)
    stats_path = repo_path(args.stats_json) if args.stats_json else output_path.with_name("divergence_stats.json")
    token_hist_path = repo_path(args.token_histogram) if args.token_histogram else output_path.with_name("divergence_token_histogram.csv")
    position_hist_path = repo_path(args.position_histogram) if args.position_histogram else output_path.with_name("divergence_position_histogram.csv")

    prompts = number_sequence_user_prompts(
        num_prompts=args.num_prompts,
        seed=args.prompt_seed,
        style=args.prompt_style,
    )
    factual_system = biased_animal_system_prompt(args.factual_trait)
    counterfactual_systems = {
        trait: biased_animal_system_prompt(trait) for trait in args.counterfactual_traits
    }

    model = tokenizer = None
    diagnostics = model_runtime_diagnostics(model_config=model_config)
    if not args.dry_run:
        model, tokenizer = load_model_and_tokenizer(model_config)
        diagnostics = model_runtime_diagnostics(model=model, model_config=model_config)
    else:
        from slgeo.models import load_tokenizer

        cfg = model_config.get("model", model_config)
        tokenizer = load_tokenizer(
            cfg.get("model_name") or cfg.get("base_model_name"),
            trust_remote_code=cfg.get("trust_remote_code", True),
            padding_side=cfg.get("padding_side", "left"),
            local_files_only=bool(cfg.get("local_files_only", False)),
        )

    records: list[dict[str, Any]] = []
    token_counter: Counter[int] = Counter()
    position_counter: Counter[int] = Counter()
    by_trait: dict[str, int] = defaultdict(int)
    total_tokens = 0
    divergence_tokens = 0

    for index, prompt in enumerate(tqdm(prompts, desc="divergence")):
        seed = args.generation_seed + index
        set_seed(seed)
        if args.dry_run:
            completion = "1, 2, 3, 4"
        else:
            completion = generate_completion(
                model=model,
                tokenizer=tokenizer,
                system_prompt=factual_system,
                user_prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                generation_mode="greedy",
                seed=seed,
            )
        prompt_text = format_chat_prompt(tokenizer, factual_system, prompt)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        full_input_ids = prompt_ids + completion_ids
        labels = [-100] * len(prompt_ids) + completion_ids
        full_mask = [False] * len(prompt_ids)

        for position in range(len(completion_ids)):
            prefix = tokenizer.decode(completion_ids[:position], skip_special_tokens=False)
            factual_prefix_text = prompt_text + prefix
            if args.dry_run:
                factual_argmax = completion_ids[position]
            else:
                factual_argmax = argmax_token_id(model, tokenizer, factual_prefix_text)

            diverged = False
            diverged_traits = []
            for trait, system_prompt in counterfactual_systems.items():
                counter_prompt_text = format_chat_prompt(tokenizer, system_prompt, prompt) + prefix
                counter_argmax = factual_argmax if args.dry_run else argmax_token_id(model, tokenizer, counter_prompt_text)
                if counter_argmax != factual_argmax:
                    diverged = True
                    diverged_traits.append(trait)
                    by_trait[trait] += 1
            full_mask.append(diverged)
            total_tokens += 1
            if diverged:
                divergence_tokens += 1
                token_counter[completion_ids[position]] += 1
                position_counter[position] += 1

        records.append(
            {
                "prompt": prompt,
                "completion": completion,
                "input_ids": full_input_ids,
                "labels": labels,
                "divergence_mask": full_mask,
                "completion_divergence_mask": full_mask[len(prompt_ids) :],
                "factual_trait": args.factual_trait,
                "counterfactual_traits": args.counterfactual_traits,
                "prompt_hash": text_hash(prompt),
                "completion_hash": text_hash(completion),
                "metadata": {
                    "prompt_index": index,
                    "prompt_seed": args.prompt_seed,
                    "generation_seed": seed,
                    "prompt_token_count": len(prompt_ids),
                    "completion_token_count": len(completion_ids),
                    "model_diagnostics": diagnostics,
                },
            }
        )

    write_jsonl(output_path, records)
    stats = {
        "examples": len(records),
        "total_tokens": total_tokens,
        "divergence_tokens": divergence_tokens,
        "divergence_fraction": divergence_tokens / total_tokens if total_tokens else 0.0,
        "divergence_fraction_by_trait": {
            trait: count / total_tokens if total_tokens else 0.0 for trait, count in by_trait.items()
        },
        "most_common_divergence_token_ids": token_counter.most_common(50),
        "decoded_divergence_tokens": [
            {"token_id": token_id, "token": tokenizer.decode([token_id]), "count": count}
            for token_id, count in token_counter.most_common(50)
        ],
        "model_diagnostics": diagnostics,
    }
    write_json(stats_path, stats)

    ensure_parent(token_hist_path)
    with Path(token_hist_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["token_id", "decoded_token", "count"])
        writer.writeheader()
        for token_id, count in token_counter.most_common():
            writer.writerow({"token_id": token_id, "decoded_token": tokenizer.decode([token_id]), "count": count})

    ensure_parent(position_hist_path)
    with Path(position_hist_path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["position", "count"])
        writer.writeheader()
        for position, count in sorted(position_counter.items()):
            writer.writerow({"position": position, "count": count})

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

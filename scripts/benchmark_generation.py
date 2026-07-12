from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.generation import generate_completion
from slgeo.io import load_yaml, write_json
from slgeo.models import load_model_and_tokenizer, model_runtime_diagnostics
from slgeo.prompts import condition_system_prompt, number_sequence_user_prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark Qwen generation throughput.")
    parser.add_argument("--fp16-model-config", default="configs/model_qwen7b.yaml")
    parser.add_argument("--4bit-model-config", default="configs/model_qwen7b_4bit.yaml")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--condition", default="subliminal_numbers")
    parser.add_argument("--trait", default="cat")
    parser.add_argument("--output-json", default="results/reproduction/qwen7b_generation_benchmark.json")
    parser.add_argument("--skip-fp16", action="store_true")
    parser.add_argument("--skip-4bit", action="store_true")
    return parser


def clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def benchmark_one(
    label: str,
    model_config_path: str,
    *,
    samples: int,
    max_new_tokens: int,
    condition: str,
    trait: str,
) -> dict:
    clear_cuda_cache()
    model_config = load_yaml(repo_path(model_config_path))
    model_config.setdefault("model", {})["local_files_only"] = True
    generation_config = dict(model_config.get("generation", {}))
    generation_config["max_new_tokens"] = max_new_tokens
    prompts = number_sequence_user_prompts(
        num_prompts=samples,
        seed=42,
        min_numbers=3,
        max_numbers=7,
        style="random_three_digit",
    )
    system_prompt = condition_system_prompt(condition, trait)

    started_load = time.perf_counter()
    model, tokenizer = load_model_and_tokenizer(model_config)
    load_seconds = time.perf_counter() - started_load
    diagnostics_after_load = model_runtime_diagnostics(model=model, model_config=model_config)
    print(f"{label} diagnostics after load: {diagnostics_after_load}")

    latencies = []
    started_generation = time.perf_counter()
    for index, prompt in enumerate(prompts):
        started_sample = time.perf_counter()
        generate_completion(
            model=model,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=float(generation_config.get("temperature", 0.7)),
            top_p=float(generation_config.get("top_p", 0.95)),
            do_sample=bool(generation_config.get("do_sample", True)),
            seed=1000 + index,
        )
        latencies.append(time.perf_counter() - started_sample)
    generation_seconds = time.perf_counter() - started_generation
    diagnostics_after_generation = model_runtime_diagnostics(model=model, model_config=model_config)

    return {
        "label": label,
        "model_config": model_config_path,
        "samples": samples,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "samples_per_second": samples / generation_seconds if generation_seconds else 0.0,
        "average_latency_seconds": sum(latencies) / len(latencies) if latencies else 0.0,
        "max_new_tokens": max_new_tokens,
        "diagnostics_after_load": diagnostics_after_load,
        "diagnostics_after_generation": diagnostics_after_generation,
    }


def main() -> None:
    args = build_parser().parse_args()
    results = []
    errors = []
    jobs = []
    if not args.skip_fp16:
        jobs.append(("fp16", args.fp16_model_config))
    if not args.skip_4bit:
        jobs.append(("4bit", args.__dict__["4bit_model_config"]))

    for label, config_path in jobs:
        try:
            results.append(
                benchmark_one(
                    label,
                    config_path,
                    samples=args.samples,
                    max_new_tokens=args.max_new_tokens,
                    condition=args.condition,
                    trait=args.trait,
                )
            )
        except Exception as exc:
            errors.append({"label": label, "model_config": config_path, "error": repr(exc)})
            print(f"WARNING: {label} benchmark failed: {exc!r}")

    summary = {"results": results, "errors": errors}
    if len(results) == 2:
        fp16 = next((row for row in results if row["label"] == "fp16"), None)
        four_bit = next((row for row in results if row["label"] == "4bit"), None)
        if fp16 and four_bit and fp16["samples_per_second"]:
            summary["speedup_4bit_vs_fp16"] = (
                four_bit["samples_per_second"] / fp16["samples_per_second"]
            )

    output_path = repo_path(args.output_json)
    write_json(output_path, summary)
    print(json.dumps(summary, indent=2))
    print(f"Benchmark written to {Path(output_path)}")


if __name__ == "__main__":
    main()

"""Measure how reversible LoRA ablations change teacher-aligned activations."""

from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.analysis.activations import (  # noqa: E402
    alignment_metrics,
    difference_vector,
    hidden_state_statistics,
)
from slgeo.analysis.attribution import (  # noqa: E402
    group_lora_modules,
    layer_index,
    prompt_drop_scores,
    summarize_prompt_values,
)
from slgeo.analysis.interventions import list_lora_modules, mask_lora_modules  # noqa: E402
from slgeo.analysis.vector_artifacts import load_vector_artifact, sha256_file  # noqa: E402
from slgeo.io import ensure_parent, load_yaml, read_jsonl  # noqa: E402
from slgeo.models import load_model_and_tokenizer  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/model_qwen7b_4bit.yaml")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--teacher-vector", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--n-prompts", type=int, default=128)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--position", choices=["last", "all"], default="all")
    parser.add_argument("--group-by", choices=["layer", "individual"], default="layer")
    parser.add_argument("--include-layers", type=int, nargs="*")
    parser.add_argument("--target-block", type=int, default=10)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--output", required=True)
    return parser


def _score(raw_student, raw_teacher, target_slot):
    metrics = alignment_metrics(raw_student, raw_teacher)
    return {
        "target_cosine": float(metrics["cosine"][target_slot]),
        "target_signed_projection": float(metrics["signed_projection"][target_slot]),
        "mean_block_cosine": float(metrics["cosine"][1:].mean()),
        "cosine_per_hidden_state_slot": metrics["cosine"].tolist(),
        "signed_projection_per_hidden_state_slot": metrics["signed_projection"].tolist(),
    }


def _group_layer(group_modules: list[str]) -> int:
    layers = {layer_index(module) for module in group_modules}
    if len(layers) != 1:
        raise ValueError(f"A causal attribution group must belong to one layer, got {layers}")
    return layers.pop()


def main() -> None:
    args = build_parser().parse_args()
    model_config_path = repo_path(args.model_config)
    adapter_path = repo_path(args.adapter_path)
    prompts_path = repo_path(args.prompts)
    teacher_path = repo_path(args.teacher_vector)
    output_path = ensure_parent(repo_path(args.output))
    selected_records = read_jsonl(prompts_path)[
        args.prompt_offset : args.prompt_offset + args.n_prompts
    ]
    prompts = [
        str(record[args.prompt_field])
        for record in selected_records
    ]
    if len(prompts) != args.n_prompts:
        raise ValueError(f"Requested {args.n_prompts} prompts, found {len(prompts)}")

    teacher = load_vector_artifact(teacher_path)
    teacher_raw = teacher["raw"]
    model_config = load_yaml(model_config_path)
    base, tokenizer = load_model_and_tokenizer(model_config)
    from peft import PeftModel

    student = PeftModel.from_pretrained(base, str(adapter_path))
    student.eval()
    modules = list_lora_modules(student)
    groups = group_lora_modules(
        modules, group_by=args.group_by, include_layers=args.include_layers
    )
    group_items = list(sorted(groups.items()))
    if args.max_groups is not None:
        group_items = group_items[: args.max_groups]

    print("Computing base and full-adapter activation statistics...")
    with student.disable_adapter():
        base_stats = hidden_state_statistics(
            student,
            tokenizer,
            prompts,
            directions=teacher_raw,
            batch_size=args.batch_size,
            position=args.position,
        )
    full_stats = hidden_state_statistics(
        student,
        tokenizer,
        prompts,
        directions=teacher_raw,
        batch_size=args.batch_size,
        position=args.position,
    )
    mean_base = base_stats["mean"]
    mean_full = full_stats["mean"]
    base_projection = base_stats["projections"]
    full_projection = full_stats["projections"]
    baseline_projection = full_projection - base_projection
    full_raw = difference_vector(mean_full, mean_base)["raw"]
    if full_raw.shape != teacher_raw.shape:
        raise ValueError(
            f"Teacher/student vector shape mismatch: {tuple(teacher_raw.shape)} vs {tuple(full_raw.shape)}"
        )
    target_slot = args.target_block + 1
    baseline = _score(full_raw, teacher_raw, target_slot)
    baseline["projection_mean_per_hidden_state_slot"] = baseline_projection.mean(dim=0).tolist()
    baseline["projection_median_per_hidden_state_slot"] = baseline_projection.median(dim=0).values.tolist()
    baseline["baseline_projection_per_prompt"] = baseline_projection.tolist()
    result = {
        "schema_version": 2,
        "analysis": "lora_group_ablation",
        "primary_within_condition_metric": "downstream_mean_drop.mean",
        "primary_cross_condition_metric": (
            "downstream_mean_drop_subliminal - downstream_mean_drop_neutral"
        ),
        "adapter_path": str(adapter_path),
        "teacher_vector": str(teacher_path),
        "teacher_vector_sha256": sha256_file(teacher_path),
        "prompts": str(prompts_path),
        "prompts_sha256": sha256_file(prompts_path),
        "model_config": str(model_config_path),
        "n_prompts": len(prompts),
        "prompt_offset": args.prompt_offset,
        "position": args.position,
        "group_by": args.group_by,
        "include_layers": args.include_layers,
        "target_block": args.target_block,
        "target_hidden_state_slot": target_slot,
        "module_count": len(modules),
        "group_count": len(group_items),
        "baseline": baseline,
        "ablations": [],
    }

    for index, (group_name, group_modules) in enumerate(group_items, start=1):
        print(f"[{index}/{len(group_items)}] Ablating {group_name} ({len(group_modules)} modules)")
        with mask_lora_modules(student, disabled_modules=group_modules):
            ablated_stats = hidden_state_statistics(
                student,
                tokenizer,
                prompts,
                directions=teacher_raw,
                batch_size=args.batch_size,
                position=args.position,
            )
        mean_ablated = ablated_stats["mean"]
        ablated_projection = ablated_stats["projections"] - base_projection
        projection_drop = baseline_projection - ablated_projection
        raw_ablated = difference_vector(mean_ablated, mean_base)["raw"]
        score = _score(raw_ablated, teacher_raw, target_slot)
        layer = _group_layer(group_modules)
        prompt_scores = prompt_drop_scores(
            projection_drop, layer=layer, fixed_target_block=args.target_block
        )
        score.update(
            {
                "group": group_name,
                "modules": group_modules,
                "layer": layer,
                "local_drop": summarize_prompt_values(prompt_scores["local_drop"]),
                "fixed_target_drop": summarize_prompt_values(
                    prompt_scores["fixed_target_drop"]
                ),
                "terminal_drop": summarize_prompt_values(prompt_scores["terminal_drop"]),
                "downstream_mean_drop": summarize_prompt_values(
                    prompt_scores["downstream_mean_drop"]
                ),
                "ablated_projection_per_prompt": ablated_projection.tolist(),
                "projection_drop_per_prompt": projection_drop.tolist(),
                # Backward-compatible diagnostics at the fixed target. They are
                # null for causally invalid later-layer comparisons.
                "target_projection_drop": baseline["target_signed_projection"]
                - score["target_signed_projection"]
                if layer <= args.target_block
                else None,
                "target_cosine_drop": baseline["target_cosine"] - score["target_cosine"]
                if layer <= args.target_block
                else None,
            }
        )
        result["ablations"].append(score)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    result["ablations"].sort(
        key=lambda row: (-row["downstream_mean_drop"]["mean"], row["group"])
    )
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(result['ablations'])} ablations to {output_path}")


if __name__ == "__main__":
    main()

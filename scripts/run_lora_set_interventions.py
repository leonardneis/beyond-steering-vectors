"""Run activation-level top-k necessity, sufficiency, and matched controls."""

from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.analysis.activations import hidden_state_statistics  # noqa: E402
from slgeo.analysis.interventions import mask_lora_modules  # noqa: E402
from slgeo.analysis.selection_plans import iter_selection_sets  # noqa: E402
from slgeo.analysis.vector_artifacts import load_vector_artifact, sha256_file  # noqa: E402
from slgeo.io import ensure_parent, load_yaml, read_jsonl  # noqa: E402
from slgeo.models import load_model_and_tokenizer  # noqa: E402


def _summary(values):
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "std": float(values.std(unbiased=True)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/model_qwen7b_4bit.yaml")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--teacher-vector", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--selection-plan", required=True)
    parser.add_argument("--n-prompts", type=int, default=256)
    parser.add_argument("--prompt-offset", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--set-names",
        nargs="+",
        default=("top_k", "random_control", "norm_matched_control", "layer_norm_matched_control"),
        help="Run only these selection-plan set types.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    adapter_path, teacher_path = repo_path(args.adapter_path), repo_path(args.teacher_vector)
    prompts_path, plan_path = repo_path(args.prompts), repo_path(args.selection_plan)
    prompts = [row["prompt"] for row in read_jsonl(prompts_path)[args.prompt_offset : args.prompt_offset + args.n_prompts]]
    if len(prompts) != args.n_prompts:
        raise ValueError("Insufficient prompts for requested set intervention")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    teacher_raw = load_vector_artifact(teacher_path)["raw"]
    base, tokenizer = load_model_and_tokenizer(load_yaml(repo_path(args.model_config)))
    from peft import PeftModel

    student = PeftModel.from_pretrained(base, str(adapter_path))
    student.eval()
    with student.disable_adapter():
        base_projection = hidden_state_statistics(
            student, tokenizer, prompts, directions=teacher_raw, batch_size=args.batch_size, position="all"
        )["projections"]
    full_projection = hidden_state_statistics(
        student, tokenizer, prompts, directions=teacher_raw, batch_size=args.batch_size, position="all"
    )["projections"] - base_projection
    result = {
        "schema_version": 1,
        "analysis": "lora_set_interventions",
        "adapter_path": str(adapter_path),
        "teacher_vector_sha256": sha256_file(teacher_path),
        "prompts_sha256": sha256_file(prompts_path),
        "selection_plan": str(plan_path),
        "n_prompts": len(prompts),
        "prompt_offset": args.prompt_offset,
        "set_names": list(args.set_names),
        "baseline_projection_per_prompt": full_projection.tolist(),
        "interventions": [],
    }
    for item in iter_selection_sets(plan, set_names=args.set_names):
        set_name, modules = item["set_name"], item["modules"]
        for mode in ("necessity", "sufficiency"):
            context = (
                mask_lora_modules(student, disabled_modules=modules)
                if mode == "necessity"
                else mask_lora_modules(student, enabled_modules=modules)
            )
            with context:
                projection = hidden_state_statistics(
                    student,
                    tokenizer,
                    prompts,
                    directions=teacher_raw,
                    batch_size=args.batch_size,
                    position="all",
                )["projections"] - base_projection
            effect = full_projection - projection if mode == "necessity" else projection
            global_values = effect[:, 1:].mean(dim=1)
            result["interventions"].append(
                {
                    "k": item["k"],
                    "set_name": set_name,
                    "draw_id": item["draw_id"],
                    "mode": mode,
                    "modules": modules,
                    "global_downstream_effect": _summary(global_values),
                    "terminal_effect": _summary(effect[:, -1]),
                    "effect_projection_per_prompt": effect.tolist(),
                }
            )
            ensure_parent(repo_path(args.output)).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(result['interventions'])} set interventions to {repo_path(args.output)}")


if __name__ == "__main__":
    main()

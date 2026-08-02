"""Evaluate behavioral necessity and sufficiency for prepared LoRA module sets."""

from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.analysis.interventions import mask_lora_modules  # noqa: E402
from slgeo.analysis.selection_plans import iter_selection_sets  # noqa: E402
from slgeo.evaluation import evaluate_preference  # noqa: E402
from slgeo.io import ensure_parent, load_yaml  # noqa: E402
from slgeo.models import load_model_and_tokenizer  # noqa: E402


def _score(result: dict) -> float:
    metrics = result.get("choice_metrics", {})
    for key in ("target_choice_rate", "target_rate"):
        if key in metrics:
            return float(metrics[key])
    metrics = result.get("metrics", {})
    if "target_choice_rate" in metrics:
        return float(metrics["target_choice_rate"])
    raise KeyError("No target choice rate in evaluation result")


def _compact_evaluation(result: dict, target_animal: str) -> dict:
    """Keep causal readouts while dropping generated text and redundant metadata."""
    return {
        "target_choice_rate": _score(result),
        "target_choice_per_prompt": [
            1.0 if row.get("parsed_choice") == target_animal else 0.0 for row in result["completions"]
        ],
        "choice_metrics": result.get("choice_metrics"),
        "token_metrics": result.get("token_metrics"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/model_qwen7b_4bit.yaml")
    parser.add_argument("--eval-config", default="configs/eval_animals.yaml")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--selection-plan", required=True)
    parser.add_argument("--target-animal", default="cat")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument(
        "--prompt-set", choices=("favorite", "exact", "paper_reference"), default="paper_reference"
    )
    parser.add_argument("--set-names", nargs="+", default=("top_k", "layer_norm_matched_control"))
    parser.add_argument("--k", nargs="+", type=int, default=(5, 10, 20))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_path = repo_path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing behavioral artifact: {output_path}")

    model_config = load_yaml(repo_path(args.model_config))
    eval_config = load_yaml(repo_path(args.eval_config)).get("evaluation", {})
    plan = json.loads(repo_path(args.selection_plan).read_text(encoding="utf-8"))
    base, tokenizer = load_model_and_tokenizer(model_config)
    from peft import PeftModel

    student = PeftModel.from_pretrained(base, str(repo_path(args.adapter_path)))
    student.eval()

    def evaluate() -> dict:
        return evaluate_preference(
            model_config=model_config,
            adapter_path=repo_path(args.adapter_path),
            target_animal=args.target_animal,
            animals=eval_config.get("animals"),
            num_samples=args.num_samples,
            max_new_tokens=int(eval_config.get("max_new_tokens", 32)),
            generation_mode="greedy",
            prompt_set=args.prompt_set,
            system_prompt_mode="neutral",
            logprob_eval=False,
            token_metric_eval=True,
            candidate_animals=eval_config.get("candidate_animals"),
            compare_base_logits=False,
            model=student,
            tokenizer=tokenizer,
        )

    with student.disable_adapter():
        base_result = evaluate()
    full_result = evaluate()
    base_evaluation = _compact_evaluation(base_result, args.target_animal)
    full_evaluation = _compact_evaluation(full_result, args.target_animal)
    base_score, full_score = base_evaluation["target_choice_rate"], full_evaluation["target_choice_rate"]
    result = {
        "schema_version": 2,
        "analysis": "lora_set_behavior",
        "adapter_path": str(repo_path(args.adapter_path)),
        "selection_plan": str(repo_path(args.selection_plan)),
        "target_animal": args.target_animal,
        "num_samples": args.num_samples,
        "prompt_set": args.prompt_set,
        "generation_mode": "greedy",
        "base_target_choice_rate": base_score,
        "full_target_choice_rate": full_score,
        "full_adapter_effect": full_score - base_score,
        "base_evaluation": base_evaluation,
        "full_evaluation": full_evaluation,
        "interventions": [],
    }
    wanted_k = set(args.k)
    for item in iter_selection_sets(plan, set_names=args.set_names, k_values=wanted_k):
        set_name, modules = item["set_name"], item["modules"]
        for mode in ("necessity", "sufficiency"):
            context = (
                mask_lora_modules(student, disabled_modules=modules)
                if mode == "necessity"
                else mask_lora_modules(student, enabled_modules=modules)
            )
            with context:
                evaluation = evaluate()
            compact = _compact_evaluation(evaluation, args.target_animal)
            score = compact["target_choice_rate"]
            effect = full_score - score if mode == "necessity" else score - base_score
            result["interventions"].append(
                {
                    "k": item["k"],
                    "set_name": set_name,
                    "draw_id": item["draw_id"],
                    "mode": mode,
                    "modules": modules,
                    "target_choice_rate": score,
                    "behavioral_effect": effect,
                    **compact,
                }
            )
            ensure_parent(output_path).write_text(
                json.dumps(result, indent=2) + "\n", encoding="utf-8"
            )
    print(f"Wrote {len(result['interventions'])} behavioral interventions to {output_path}")


if __name__ == "__main__":
    main()

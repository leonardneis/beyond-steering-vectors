"""Extract reproducible teacher and student vectors for the thesis analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.analysis.activations import alignment_metrics, difference_vector, mean_hidden_states  # noqa: E402
from slgeo.analysis.vector_artifacts import (  # noqa: E402
    load_vector_artifact,
    save_vector_artifact,
    sha256_file,
    split_half_cosine,
)
from slgeo.io import load_yaml, read_jsonl  # noqa: E402
from slgeo.models import load_model_and_tokenizer  # noqa: E402
from slgeo.prompts import neutral_system_prompt, reference_animal_system_prompt  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/model_qwen7b_4bit.yaml")
    parser.add_argument("--adapter-path")
    parser.add_argument(
        "--teacher-vector",
        help="Reuse a frozen teacher artifact instead of extracting it again.",
    )
    parser.add_argument("--prompts", required=True, help="JSONL containing a prompt field")
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--trait", default="cat")
    parser.add_argument("--n-prompts", type=int, default=1024)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument(
        "--teacher-only",
        action="store_true",
        help="Extract only a teacher vector; no student adapter is loaded.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--teacher-position", choices=["last", "all"], default="last")
    parser.add_argument("--student-position", choices=["last", "all"], default="all")
    parser.add_argument(
        "--neutral-system-prompt",
        choices=["none", "helpful"],
        default="none",
        help="The paper-reference condition uses no neutral system prompt.",
    )
    parser.add_argument("--output-dir", default="results/geometry/vectors/cat_reference")
    return parser


def _extract_halves(model, tokenizer, prompts, *, system_prompt, batch_size, position):
    midpoint = len(prompts) // 2
    first = mean_hidden_states(
        model,
        tokenizer,
        prompts[:midpoint],
        system_prompt=system_prompt,
        batch_size=batch_size,
        position=position,
    )
    second = mean_hidden_states(
        model,
        tokenizer,
        prompts[midpoint:],
        system_prompt=system_prompt,
        batch_size=batch_size,
        position=position,
    )
    return first, second


def main() -> None:
    args = build_parser().parse_args()
    model_config_path = repo_path(args.model_config)
    if not args.teacher_only and not args.adapter_path:
        raise ValueError("--adapter-path is required unless --teacher-only is set")
    if args.teacher_only and args.teacher_vector:
        raise ValueError("--teacher-only cannot be combined with --teacher-vector")
    if args.prompt_offset < 0:
        raise ValueError("prompt-offset must be >= 0")
    adapter_path = repo_path(args.adapter_path) if args.adapter_path else None
    prompts_path = repo_path(args.prompts)
    output_dir = repo_path(args.output_dir)
    if args.n_prompts < 2 or args.n_prompts % 2:
        raise ValueError("n-prompts must be an even integer >= 2 for split-half reliability")

    records = read_jsonl(prompts_path)
    prompt_end = args.prompt_offset + args.n_prompts
    prompts = [
        str(row[args.prompt_field])
        for row in records[args.prompt_offset : prompt_end]
    ]
    if len(prompts) != args.n_prompts:
        raise ValueError(f"Requested {args.n_prompts} prompts, found {len(prompts)}")

    model_config = load_yaml(model_config_path)
    model, tokenizer = load_model_and_tokenizer(model_config)
    model.eval()
    trait_prompt = reference_animal_system_prompt(args.trait)
    neutral_prompt = neutral_system_prompt() if args.neutral_system_prompt == "helpful" else None

    teacher_source_path = repo_path(args.teacher_vector) if args.teacher_vector else None
    if teacher_source_path:
        print(f"Reusing frozen teacher vector: {teacher_source_path}")
        teacher_artifact = load_vector_artifact(teacher_source_path)
        teacher_raw = teacher_artifact["raw"]
        teacher_reliability = teacher_artifact.get("reliability")
    else:
        print("Extracting teacher split halves...")
        teacher_trait_halves = _extract_halves(
            model,
            tokenizer,
            prompts,
            system_prompt=trait_prompt,
            batch_size=args.batch_size,
            position=args.teacher_position,
        )
        teacher_neutral_halves = _extract_halves(
            model,
            tokenizer,
            prompts,
            system_prompt=neutral_prompt,
            batch_size=args.batch_size,
            position=args.teacher_position,
        )
        teacher_half_vectors = [
            difference_vector(trait, neutral)["raw"]
            for trait, neutral in zip(teacher_trait_halves, teacher_neutral_halves, strict=True)
        ]
        teacher_raw = torch.stack(teacher_half_vectors).mean(dim=0)
        teacher_reliability = split_half_cosine(*teacher_half_vectors)

    common_metadata = {
        "base_model": model_config.get("model", {}).get("model_name"),
        "trait": args.trait,
        "n_prompts": len(prompts),
        "prompt_offset": args.prompt_offset,
        "prompt_path": str(prompts_path),
        "prompt_sha256": sha256_file(prompts_path),
        "prompt_field": args.prompt_field,
        "batch_size": args.batch_size,
        "neutral_system_prompt_mode": args.neutral_system_prompt,
        "neutral_system_prompt": neutral_prompt,
        "model_config_path": str(model_config_path),
        "model_config_sha256": sha256_file(model_config_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    if teacher_source_path:
        teacher_path = teacher_source_path
    else:
        teacher_path = save_vector_artifact(
            output_dir / "v_teacher.pt",
            raw=teacher_raw,
            reliability=teacher_reliability,
            metadata={
                **common_metadata,
                "kind": "v_teacher",
                "position": args.teacher_position,
                "trait_system_prompt": trait_prompt,
            },
        )
    if args.teacher_only:
        summary = {
            "teacher_path": str(teacher_path),
            "teacher_shape": list(teacher_raw.shape),
            "teacher_split_half_cosine": (
                teacher_reliability.tolist() if teacher_reliability is not None else None
            ),
            "n_prompts": len(prompts),
            "prompt_offset": args.prompt_offset,
            "indexing": "slot 0=embedding; slot i+1=transformer block i",
        }
        (output_dir / "alignment.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, indent=2))
        return

    assert adapter_path is not None
    print("Attaching student adapter and extracting student split halves...")
    from peft import PeftModel

    student = PeftModel.from_pretrained(model, str(adapter_path))
    student.eval()
    student_halves = _extract_halves(
        student,
        tokenizer,
        prompts,
        system_prompt=neutral_prompt,
        batch_size=args.batch_size,
        position=args.student_position,
    )
    with student.disable_adapter():
        base_halves = _extract_halves(
            student,
            tokenizer,
            prompts,
            system_prompt=neutral_prompt,
            batch_size=args.batch_size,
            position=args.student_position,
        )
    student_half_vectors = [
        difference_vector(finetuned, base)["raw"]
        for finetuned, base in zip(student_halves, base_halves, strict=True)
    ]
    student_raw = torch.stack(student_half_vectors).mean(dim=0)
    student_reliability = split_half_cosine(*student_half_vectors)

    adapter_weights = next(
        (candidate for candidate in (adapter_path / "adapter_model.safetensors", adapter_path / "adapter_model.bin") if candidate.exists()),
        None,
    )
    student_path = save_vector_artifact(
        output_dir / "v_student.pt",
        raw=student_raw,
        reliability=student_reliability,
        metadata={
            **common_metadata,
            "kind": "v_student",
            "position": args.student_position,
            "adapter_path": str(adapter_path),
            "adapter_sha256": sha256_file(adapter_weights) if adapter_weights else None,
        },
    )
    metrics = alignment_metrics(student_raw, teacher_raw)
    summary = {
        "teacher_path": str(teacher_path),
        "student_path": str(student_path),
        "teacher_shape": list(teacher_raw.shape),
        "student_shape": list(student_raw.shape),
        "cosine_per_hidden_state_slot": metrics["cosine"].tolist(),
        "signed_projection_per_hidden_state_slot": metrics["signed_projection"].tolist(),
        "projection_fraction_per_hidden_state_slot": metrics["projection_fraction"].tolist(),
        "teacher_split_half_cosine": teacher_reliability.tolist() if teacher_reliability is not None else None,
        "student_split_half_cosine": student_reliability.tolist(),
        "indexing": "slot 0=embedding; slot i+1=transformer block i",
    }
    (output_dir / "alignment.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

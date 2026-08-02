"""Extract final post-RMS states under frozen LoRA necessity interventions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.analysis.interventions import mask_lora_modules  # noqa: E402
from slgeo.analysis.selection_plans import iter_selection_sets  # noqa: E402
from slgeo.analysis.vector_artifacts import load_vector_artifact, sha256_file  # noqa: E402
from slgeo.evaluation import _candidate_token_ids  # noqa: E402
from slgeo.generation import _model_input_device  # noqa: E402
from slgeo.io import load_yaml  # noqa: E402
from slgeo.models import format_chat_prompt, load_model_and_tokenizer  # noqa: E402
from slgeo.prompts import neutral_system_prompt  # noqa: E402
from run_confirmatory_manifest import tree_digest  # noqa: E402


def load_prompts(path: Path) -> list[dict]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"prompt_id", "family", "prompt"}
    if not records or any(required - set(row) for row in records):
        raise ValueError("Prompt file is empty or missing required fields")
    for key in ("prompt_id", "prompt"):
        values = [str(row[key]) for row in records]
        if len(values) != len(set(values)):
            raise ValueError(f"Prompt file contains duplicate {key}")
    return records


def _forward_states_and_logits(
    model, tokenizer, prompts: list[str], batch_size: int, candidate_ids: list[int]
):
    import torch

    states, logits = [], []
    system = neutral_system_prompt()
    device = _model_input_device(model)
    for start in range(0, len(prompts), batch_size):
        rendered = [format_chat_prompt(tokenizer, system, prompt) for prompt in prompts[start : start + batch_size]]
        inputs = tokenizer(rendered, return_tensors="pt", padding=True)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            output = model(**inputs, output_hidden_states=True, return_dict=True)
        final = output.hidden_states[-1][:, -1, :]
        states.append(final.detach().float().cpu().numpy())
        logits.append(output.logits[:, -1, candidate_ids].detach().float().cpu().numpy())
    return np.concatenate(states), np.concatenate(logits)


def _margin(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Expected Cat and Lion logits in fixed column order")
    return values[:, 0] - values[:, 1]


def _head_logits(
    model, states: np.ndarray, batch_size: int, candidate_ids: list[int]
) -> np.ndarray:
    """Apply the two frozen LM-head rows in high precision."""
    head = model.get_output_embeddings()
    weight = (
        head.weight[candidate_ids].detach().float().cpu().numpy().astype(np.float64)
    )
    bias = None
    if getattr(head, "bias", None) is not None:
        bias = head.bias[candidate_ids].detach().float().cpu().numpy().astype(np.float64)
    logits = np.asarray(states, dtype=np.float64) @ weight.T
    if bias is not None:
        logits = logits + bias
    return logits


def _runtime_head_logits(
    model, states: np.ndarray, batch_size: int, candidate_ids: list[int]
) -> np.ndarray:
    """Re-run the model's exact LM-head code path for the state integrity gate."""
    import torch

    head = model.get_output_embeddings()
    parameter = next(head.parameters())
    indices = torch.as_tensor(candidate_ids, device=parameter.device)
    output = []
    for start in range(0, len(states), batch_size):
        tensor = torch.from_numpy(states[start : start + batch_size]).to(
            device=parameter.device, dtype=parameter.dtype
        )
        with torch.no_grad():
            logits = head(tensor).index_select(-1, indices)
            output.append(logits.float().cpu().numpy())
    return np.concatenate(output)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--condition", choices=("subliminal", "neutral"), required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--selection-plan", required=True)
    parser.add_argument("--teacher-vector", required=True)
    parser.add_argument("--teacher-hidden-state-index", type=int, default=-1)
    parser.add_argument("--target-animal", default="cat")
    parser.add_argument("--competitor-animal", default="lion")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--control-draws", type=int, default=25)
    parser.add_argument("--state-head-atol", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output_path = repo_path(args.output)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite state artifact: {output_path}")

    adapter_path = repo_path(args.adapter_path)
    if tree_digest(adapter_path) != args.adapter_sha256:
        raise RuntimeError(f"Frozen adapter hash mismatch: {adapter_path}")
    prompt_path = repo_path(args.prompt_file)
    selection_path = repo_path(args.selection_plan)
    teacher_path = repo_path(args.teacher_vector)
    records = load_prompts(prompt_path)
    plan = json.loads(selection_path.read_text(encoding="utf-8"))
    interventions = list(iter_selection_sets(
        plan, set_names=("top_k", "norm_matched_control"), k_values=(args.k,)
    ))
    counts = {name: sum(row["set_name"] == name for row in interventions) for name in ("top_k", "norm_matched_control")}
    if counts != {"top_k": 1, "norm_matched_control": args.control_draws}:
        raise ValueError(f"Frozen selection inventory differs: {counts}")
    if any(len(row["modules"]) != args.k for row in interventions):
        raise ValueError("A frozen intervention does not contain exactly k modules")

    model_config = load_yaml(repo_path(args.model_config))
    model_name = str(model_config.get("model", {}).get("model_name", ""))
    cache_root = Path(
        os.getenv("HUGGINGFACE_HUB_CACHE")
        or os.getenv("HF_HUB_CACHE")
        or Path(os.getenv("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    )
    ref = cache_root / ("models--" + model_name.replace("/", "--")) / "refs/main"
    if not ref.is_file() or ref.read_text(encoding="utf-8") != args.model_revision:
        raise RuntimeError(f"Frozen Hugging Face revision is unavailable or differs: {ref}")
    base, tokenizer = load_model_and_tokenizer(model_config)
    from peft import PeftModel

    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()
    ids, skipped = _candidate_token_ids(tokenizer, [args.target_animal, args.competitor_animal])
    if skipped or set(ids) != {args.target_animal, args.competitor_animal}:
        raise ValueError(f"Target/competitor must each map to one token: ids={ids}, skipped={skipped}")
    target_id, competitor_id = ids[args.target_animal], ids[args.competitor_animal]

    artifact = load_vector_artifact(teacher_path)
    teacher = artifact["unit"][args.teacher_hidden_state_index].detach().float().cpu().numpy()
    prompts = [str(row["prompt"]) for row in records]
    candidate_ids = [target_id, competitor_id]
    full_state, full_direct_logits = _forward_states_and_logits(
        model, tokenizer, prompts, args.batch_size, candidate_ids
    )
    if full_state.shape[1] != teacher.shape[0]:
        raise ValueError(f"Teacher/state dimension mismatch: {teacher.shape}, {full_state.shape}")
    full_head_logits = _runtime_head_logits(model, full_state, args.batch_size, candidate_ids)
    head_error = float(np.max(np.abs(full_direct_logits - full_head_logits)))
    if head_error > args.state_head_atol:
        raise RuntimeError(f"Final hidden state does not reconstruct model logits: {head_error}")

    head = model.get_output_embeddings()
    margin_direction = (
        head.weight[target_id] - head.weight[competitor_id]
    ).detach().float().cpu().numpy()
    ablated_states = []
    margin_lists = {prefix: [] for prefix in ("ablated", "teacher_patched", "residual_patched")}
    for item in interventions:
        with mask_lora_modules(model, disabled_modules=item["modules"]):
            state, direct = _forward_states_and_logits(
                model, tokenizer, prompts, args.batch_size, candidate_ids
            )
        ablated_head = _runtime_head_logits(model, state, args.batch_size, candidate_ids)
        head_error = max(head_error, float(np.max(np.abs(direct - ablated_head))))
        if head_error > args.state_head_atol:
            raise RuntimeError(f"An ablated final state does not reconstruct model logits: {head_error}")
        delta = full_state - state
        coefficient = delta @ teacher
        parallel = coefficient[:, None] * teacher[None, :]
        ablated_states.append(state)
        for prefix, logits in (
            ("ablated", _head_logits(model, state, args.batch_size, candidate_ids)),
            ("teacher_patched", _head_logits(model, state + parallel, args.batch_size, candidate_ids)),
            ("residual_patched", _head_logits(model, state + (delta - parallel), args.batch_size, candidate_ids)),
        ):
            margin_lists[prefix].append(_margin(logits).astype(np.float32))

    ablated_state = np.asarray(ablated_states, dtype=np.float32)
    metric_arrays = {
        f"{prefix}_margin": np.asarray(value, dtype=np.float32)
        for prefix, value in margin_lists.items()
    }

    metadata = {
        "schema_version": 1,
        "analysis": "final_state_directional_extraction",
        "condition": args.condition,
        "model_config": args.model_config,
        "model_revision": args.model_revision,
        "adapter_path": str(adapter_path),
        "adapter_sha256": args.adapter_sha256,
        "prompt_file": str(prompt_path),
        "prompt_file_sha256": sha256_file(prompt_path),
        "selection_plan": str(selection_path),
        "selection_plan_sha256": sha256_file(selection_path),
        "teacher_vector": str(teacher_path),
        "teacher_vector_sha256": sha256_file(teacher_path),
        "teacher_hidden_state_index": args.teacher_hidden_state_index,
        "state_semantics": "outputs.hidden_states[-1][:,-1,:], verified against the Cat and Lion LM-head rows",
        "system_prompt": neutral_system_prompt(),
        "target_animal": args.target_animal,
        "competitor_animal": args.competitor_animal,
        "target_token_id": target_id,
        "competitor_token_id": competitor_id,
        "state_head_max_abs_error": head_error,
        "interventions": interventions,
    }
    arrays = {
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "prompt_ids": np.asarray([row["prompt_id"] for row in records]),
        "families": np.asarray([row["family"] for row in records]),
        "set_names": np.asarray([row["set_name"] for row in interventions]),
        "draw_ids": np.asarray([str(row["draw_id"]) for row in interventions]),
        "teacher_unit": teacher.astype(np.float32),
        "margin_direction": margin_direction.astype(np.float32),
        "full_state": full_state.astype(np.float32),
        "ablated_state": ablated_state,
        "full_margin": _margin(_head_logits(model, full_state, args.batch_size, candidate_ids)).astype(np.float32),
        **metric_arrays,
    }
    _atomic_npz(output_path, arrays)
    print(f"Wrote {args.condition} final states for {len(interventions)} interventions to {output_path}")


if __name__ == "__main__":
    main()

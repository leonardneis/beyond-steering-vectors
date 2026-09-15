"""Outcome-blind technical validation for C18.

Only fixed tokenizer bytes and synthetic non-study prompts enter model
forwards. The report schema rejects scientific cells, estimands, candidate
values, and classifications.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
import platform
from pathlib import Path
import sys

from _bootstrap import bootstrap, repo_path

bootstrap()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from run_bidirectional_teacher_coordinate_interchange import (  # noqa: E402
    batch_inputs, configure_determinism, load_model,
)
from slgeo.analysis.c18_execution import (  # noqa: E402
    assert_lora_census, assert_runtime_identity, assert_scalings_restored, candidate_logits_and_state,
    capture_block_state, identity_block_hooks, margin_from_direction, replace_block_output,
    snapshot_lora_scalings,
)
from slgeo.analysis.c18_manifest import selection_inventory, validate_manifest_contract, validate_public_inputs  # noqa: E402
from slgeo.analysis.interventions import mask_lora_modules  # noqa: E402
from slgeo.analysis.teacher_coordinate_interchange import (  # noqa: E402
    atomic_json, atomic_text, capture_natural_donors, coordinate_clamp_hooks,
    normalize_rows_float64, sha256_file, validate_technical_report,
)
from slgeo.io import load_yaml  # noqa: E402
from slgeo.models import format_chat_prompt  # noqa: E402
from slgeo.prompts import neutral_system_prompt  # noqa: E402


SYNTHETIC_PROMPTS = [
    "Repeat the integer 17.", "Name a primary color.", "Write the word table.",
    "What is two plus three?", "Return a single comma.", "Name a geometric shape.",
]


def exact_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)


def tokenizer_validation(tokenizer, manifest: dict) -> str:
    frozen = manifest["frozen_inputs"]
    inventory = json.loads(repo_path(frozen["token_inventory"]["path"]).read_text(encoding="utf-8"))
    prompts = [json.loads(line) for line in repo_path(frozen["prompt_file"]["path"]).read_text(encoding="utf-8").splitlines() if line]
    if len(prompts) != len(inventory["rows"]):
        raise RuntimeError("token inventory length differs")
    for prompt, expected in zip(prompts, inventory["rows"]):
        rendered = format_chat_prompt(tokenizer, neutral_system_prompt(), prompt["prompt"])
        encoded = tokenizer(rendered, add_special_tokens=True)
        if prompt["prompt_id"] != expected["prompt_id"] or encoded["input_ids"] != expected["input_ids"] or encoded["attention_mask"] != expected["attention_mask"]:
            raise RuntimeError(f"token inventory mismatch: {prompt['prompt_id']}")
        if hashlib.sha256(rendered.encode("utf-8")).hexdigest() != expected["rendered_utf8_sha256"]:
            raise RuntimeError(f"rendered prompt mismatch: {prompt['prompt_id']}")
    return "PASS"


def synthetic_rows(tokenizer) -> list[dict]:
    rows = []
    for index, prompt in enumerate(SYNTHETIC_PROMPTS):
        rendered = format_chat_prompt(tokenizer, neutral_system_prompt(), prompt)
        encoded = tokenizer(rendered, add_special_tokens=True)
        rows.append({"prompt_id": f"technical_{index}", "family": "technical", **encoded})
    return rows


def validate_fsd_references(model, tokenizer, condition: str, manifest: dict, modules_by_set: list[dict]) -> dict[str, object]:
    frozen = manifest["frozen_inputs"]
    inventory = json.loads(repo_path(frozen["token_inventory"]["path"]).read_text(encoding="utf-8"))["rows"]
    fsd_path = repo_path(frozen["fsd"][f"{condition}_states"]["path"])
    with np.load(fsd_path, allow_pickle=False) as fsd:
        direction = fsd["margin_direction"].astype(np.float64)
        expected_full = fsd["full_margin"].astype(np.float64)
        expected_ablated = fsd["ablated_margin"].astype(np.float64)
        prompt_ids = fsd["prompt_ids"].tolist()
        set_ids = ["top" if name == "top_k" else f"norm_{int(draw):02d}" for name, draw in zip(fsd["set_names"].tolist(), fsd["draw_ids"].tolist())]
    if prompt_ids != [row["prompt_id"] for row in inventory] or set_ids != [row["set_id"] for row in modules_by_set]:
        raise RuntimeError("FSD reference ID inventory differs")
    candidate_ids = [tokenizer.encode(" cat", add_special_tokens=False)[0], tokenizer.encode(" lion", add_special_tokens=False)[0]]
    device = next(model.parameters()).device

    def run(modules=None):
        values, direct_errors = [], []
        context = mask_lora_modules(model, disabled_modules=modules) if modules is not None else nullcontext()
        with context:
            for start in range(0, 72, 6):
                inputs = batch_inputs(tokenizer, inventory[start : start + 6], device)
                state, logits = candidate_logits_and_state(model, inputs, candidate_ids)
                projected = margin_from_direction(state, direction)
                direct = (logits[:, 0].double() - logits[:, 1].double()).cpu().numpy()
                values.extend(projected.tolist())
                direct_errors.extend(np.abs(projected - direct).tolist())
        return np.asarray(values), max(direct_errors)

    full, head_error = run()
    maximum = float(np.max(np.abs(full - expected_full)))
    for index, item in enumerate(modules_by_set):
        ablated, direct_error = run(list(item["modules"]))
        head_error = max(head_error, direct_error)
        maximum = max(maximum, float(np.max(np.abs(ablated - expected_ablated[index]))))
    tolerance = manifest["numerical_tolerances"]["reference_margin_atol"]
    if maximum > tolerance or head_error > tolerance:
        raise RuntimeError(f"FSD natural reference validation failed: reference={maximum}, head={head_error}")
    return {"status": "PASS", "max_abs_reference_error": maximum, "max_abs_head_error": float(head_error), "rows_checked": 27 * 72}


def validate_adapter(manifest: dict, condition: str, teacher: torch.Tensor, orthogonal: torch.Tensor) -> dict[str, object]:
    model, tokenizer = load_model(manifest, condition)
    if getattr(model.config, "_attn_implementation", None) != "sdpa":
        raise RuntimeError("loaded attention backend is not sdpa")
    if str(model.dtype) != "torch.float16" or not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("loaded dtype/quantization identity differs")
    device_map = getattr(model, "hf_device_map", {}) or {}
    if any(str(value).lower() in {"cpu", "disk"} for value in device_map.values()):
        raise RuntimeError("model contains CPU/disk offload")
    names = assert_lora_census(model)
    plan = selection_inventory(repo_path(manifest["frozen_inputs"]["selection_plan"]["path"]))
    modules = list(plan[0]["modules"])
    if not set(modules) <= set(names):
        raise RuntimeError("selected mask is not contained in the 196-module census")
    before = snapshot_lora_scalings(model)
    with mask_lora_modules(model, disabled_modules=modules):
        during = snapshot_lora_scalings(model)
        if sum(float(during[name]) == 0.0 for name in names) != 20:
            raise RuntimeError("necessity mask does not disable exactly 20 modules")
    assert_scalings_restored(model, before)
    try:
        with mask_lora_modules(model, disabled_modules=modules):
            raise RuntimeError("intentional restoration test")
    except RuntimeError as exc:
        if str(exc) != "intentional restoration test":
            raise
    assert_scalings_restored(model, before)

    rows = synthetic_rows(tokenizer)
    candidate_ids = [tokenizer.encode(" cat", add_special_tokens=False)[0], tokenizer.encode(" lion", add_special_tokens=False)[0]]
    device = next(model.parameters()).device
    batch = batch_inputs(tokenizer, rows, device)
    baseline_state, baseline_logits = candidate_logits_and_state(model, batch, candidate_ids)
    with identity_block_hooks(model) as identity_calls:
        _identity_state, identity_logits = candidate_logits_and_state(model, batch, candidate_ids)
    if identity_calls != list(range(27)) or not exact_equal(baseline_logits, identity_logits):
        raise RuntimeError("identity hook validation failed")
    with capture_natural_donors(model, teacher, batch["attention_mask"]) as donor:
        _donor_state, donor_logits = candidate_logits_and_state(model, batch, candidate_ids)
    with coordinate_clamp_hooks(model, teacher, donor["coordinates"], batch["attention_mask"]) as replay:
        with capture_block_state(model, 26) as replay_cut:
            _replay_state, replay_logits = candidate_logits_and_state(model, batch, candidate_ids)
    if replay["calls"] != list(range(27)) or not exact_equal(donor_logits, replay_logits):
        raise RuntimeError("null/self-replay validation failed")
    if any(stats.bound_violation_max > 0 or stats.vanished_nonzero_count for _block, stats in replay["diagnostics"]):
        raise RuntimeError("coordinate analytic invariant failed")
    with mask_lora_modules(model, disabled_modules=modules):
        with capture_natural_donors(model, teacher, batch["attention_mask"]) as donor_a:
            _a_state, a_logits = candidate_logits_and_state(model, batch, candidate_ids)
        with coordinate_clamp_hooks(model, teacher, donor_a["coordinates"], batch["attention_mask"]):
            _a_replay_state, a_replay_logits = candidate_logits_and_state(model, batch, candidate_ids)
    if not exact_equal(a_logits, a_replay_logits):
        raise RuntimeError("A-to-A self replay failed")
    shifted = {block: coordinate + 0.25 for block, coordinate in donor["coordinates"].items()}
    with coordinate_clamp_hooks(model, teacher, shifted, batch["attention_mask"]) as nonzero:
        candidate_logits_and_state(model, batch, candidate_ids)
    with coordinate_clamp_hooks(model, teacher, shifted, batch["attention_mask"], dose_rows=orthogonal) as dose:
        candidate_logits_and_state(model, batch, candidate_ids)
    all_nonzero = [stats for audit in (nonzero, dose) for _block, stats in audit["diagnostics"]]
    if any(stats.bound_violation_max > 0 or stats.vanished_nonzero_count for stats in all_nonzero):
        raise RuntimeError("nonzero coordinate/cast/orthogonal-dose invariant failed")

    with capture_block_state(model, 27) as terminal:
        _terminal_state, terminal_logits = candidate_logits_and_state(model, batch, candidate_ids)
    with replace_block_output(model, 27, terminal[0]) as terminal_calls:
        _terminal_replay_state, terminal_replay_logits = candidate_logits_and_state(model, batch, candidate_ids)
    if terminal_calls != [27] or not exact_equal(terminal_logits, terminal_replay_logits):
        raise RuntimeError("terminal full-state replay failed")

    singles = []
    for row in rows:
        single = batch_inputs(tokenizer, [row], device)
        _state, logits = candidate_logits_and_state(model, single, candidate_ids)
        singles.append(logits.cpu())
    single_logits = torch.cat(singles, dim=0)
    single_difference = single_logits[:, 0].float() - single_logits[:, 1].float()
    batch_difference = baseline_logits.cpu()[:, 0].float() - baseline_logits.cpu()[:, 1].float()
    batch_error = float(torch.max(torch.abs(single_difference - batch_difference)).item())
    if batch_error > manifest["numerical_tolerances"]["batch_margin_atol"]:
        raise RuntimeError(f"B=1 versus B=6 technical error exceeds tolerance: {batch_error}")
    assert_scalings_restored(model, before)
    reference = validate_fsd_references(model, tokenizer, condition, manifest, plan)
    return {
        "condition": condition, "lora_module_count": len(names), "disabled_module_count": 20,
        "identity": "PASS", "null_self_replay": "PASS", "self_replay_A_and_F": "PASS", "terminal_full_state_replay": "PASS",
        "hook_count": len(replay["calls"]), "hook_order": "PASS",
        "batch_1_vs_6_max_abs_error": batch_error,
        "coordinate_bound": "PASS", "restore_success": "PASS", "restore_exception": "PASS",
        "attention_backend": model.config._attn_implementation, "dtype": str(model.dtype),
        "quantization": "nf4_4bit", "cpu_offload": False,
        "fsd_natural_reference": reference,
        "tokenizer": tokenizer,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/validation/cat_bidirectional_teacher_coordinate_interchange_v1.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = repo_path(args.manifest)
    manifest = load_yaml(manifest_path)
    validate_manifest_contract(manifest)
    inputs = validate_public_inputs(manifest, repo_path("."), require_runtime_inputs=True)
    configure_determinism()
    actual = assert_runtime_identity(manifest)
    teacher_artifact = torch.load(repo_path(manifest["frozen_inputs"]["teacher_tensor"]["path"]), map_location="cpu", weights_only=True)
    teacher = normalize_rows_float64(teacher_artifact["unit"][1:28])
    with np.load(repo_path(manifest["frozen_inputs"]["orthogonal_directions"]["path"]), allow_pickle=False) as data:
        orthogonal = torch.from_numpy(data["directions"][0].copy())
    adapter_reports = []
    tokenizer = None
    for condition in ("subliminal", "neutral"):
        item = validate_adapter(manifest, condition, teacher, orthogonal)
        tokenizer = item.pop("tokenizer")
        adapter_reports.append(item)
    token_status = tokenizer_validation(tokenizer, manifest)
    report = {
        "schema_version": 1, "experiment_id": manifest["experiment_id"], "status": "PASS",
        "mode": "outcome_blind_technical_only", "manifest_sha256": sha256_file(manifest_path),
        "execution_identity": actual, "frozen_inputs": inputs, "token_inventory": token_status,
        "adapters": adapter_reports,
        "outcome_guard": {"scientific_forwards": 0, "cross_cells_persisted": 0, "estimands_computed": 0},
    }
    validate_technical_report(report)
    output = repo_path(args.output)
    atomic_json(output, report)
    provenance = {
        "schema_version": 1, "artifact": str(output), "artifact_sha256": sha256_file(output),
        "experiment_id": manifest["experiment_id"],
        "git_commit": os.environ.get("SLGEO_EXECUTION_GIT_COMMIT"),
        "git_dirty": os.environ.get("SLGEO_EXECUTION_GIT_DIRTY"),
        "execution_identity": actual,
    }
    sidecar = output.with_suffix(output.suffix + ".provenance.json")
    atomic_json(sidecar, provenance)
    checksums = f"{sha256_file(output)}  {output.name}\n{sha256_file(sidecar)}  {sidecar.name}\n"
    atomic_text(output.parent / "SHA256SUMS", checksums)


if __name__ == "__main__":
    main()

"""Outcome-blind synthetic-only technical validation for frozen C18 v2."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path

from _bootstrap import bootstrap, repo_path

bootstrap()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from run_bidirectional_teacher_coordinate_interchange_v2 import configure_determinism, load_model  # noqa: E402
from slgeo.analysis.c18_execution import (  # noqa: E402
    assert_lora_census, assert_runtime_identity, assert_scalings_restored,
    candidate_logits_and_state, capture_block_state, identity_block_hooks,
    replace_block_output, snapshot_lora_scalings,
)
from slgeo.analysis.c18_v2_execution import (  # noqa: E402
    HARD_GATES, assert_finite, assert_raw_equal, immutable_coordinate_digest,
    logical_position_ids, null_operator_hooks, physical_cache_position,
    validate_technical_report,
)
from slgeo.analysis.c18_v2_manifest import (  # noqa: E402
    apply_storage_overrides, validate_manifest_contract, validate_public_inputs,
)
from slgeo.analysis.interventions import mask_lora_modules  # noqa: E402
from slgeo.analysis.teacher_coordinate_interchange import (  # noqa: E402
    atomic_json, atomic_text, capture_natural_donors, coordinate_clamp_hooks,
    normalize_rows_float64, sha256_file,
)
from slgeo.io import load_yaml  # noqa: E402
from slgeo.models import format_chat_prompt  # noqa: E402
from slgeo.prompts import neutral_system_prompt  # noqa: E402


SYNTHETIC_PROMPTS = (
    "Repeat the integer 17.", "Name a primary color.", "Write the word table.",
    "What is two plus three?", "Return a single comma.", "Name a geometric shape.",
)


def synthetic_batch(tokenizer, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = []
    for prompt in SYNTHETIC_PROMPTS:
        rendered = format_chat_prompt(tokenizer, neutral_system_prompt(), prompt)
        encoded.append(tokenizer(rendered, add_special_tokens=True)["input_ids"])
    width = max(map(len, encoded))
    ids = [[int(tokenizer.pad_token_id)] * (width-len(row)) + list(row) for row in encoded]
    masks = [[0] * (width-len(row)) + [1] * len(row) for row in encoded]
    attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
    return {"input_ids": torch.tensor(ids, dtype=torch.long, device=device),
            "attention_mask": attention_mask,
            "position_ids": logical_position_ids(attention_mask),
            "cache_position": physical_cache_position(width, device)}


def synthetic_directions(hidden_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.Generator(np.random.PCG64(20260920))
    teacher = rng.standard_normal((27, hidden_size), dtype=np.float64)
    teacher /= np.linalg.norm(teacher, axis=1, keepdims=True)
    controls = []
    for _family in range(5):
        raw = rng.standard_normal((27, hidden_size), dtype=np.float64)
        raw -= np.sum(raw * teacher, axis=1, keepdims=True) * teacher
        raw /= np.linalg.norm(raw, axis=1, keepdims=True)
        controls.append(raw)
    return torch.from_numpy(teacher), torch.from_numpy(np.stack(controls))


def read(model, inputs, ids):
    state, logits = candidate_logits_and_state(model, inputs, ids)
    assert_finite(state, logits)
    return state, logits


def validate_condition(manifest: dict, condition: str) -> dict[str, object]:
    model, tokenizer = load_model(manifest, condition)
    device = next(model.parameters()).device
    inputs = synthetic_batch(tokenizer, device)
    candidate_ids = []
    for word in (" cat", " lion"):
        values = tokenizer.encode(word, add_special_tokens=False)
        if len(values) != 1: raise RuntimeError("synthetic technical token is not single-token")
        candidate_ids.append(values[0])
    names = assert_lora_census(model, expected=196)
    synthetic_mask = names[:20]
    if len(set(synthetic_mask)) != 20: raise RuntimeError("synthetic mask does not contain 20 modules")
    before = snapshot_lora_scalings(model)
    with mask_lora_modules(model, disabled_modules=synthetic_mask):
        during = snapshot_lora_scalings(model)
        if sum(float(during[name]) == 0 for name in names) != 20:
            raise RuntimeError("synthetic mask did not disable exactly 20 modules")
    assert_scalings_restored(model, before)
    try:
        with mask_lora_modules(model, disabled_modules=synthetic_mask):
            raise RuntimeError("restore_probe")
    except RuntimeError as error:
        if str(error) != "restore_probe": raise
    assert_scalings_restored(model, before)

    state_r0, logits_r0 = read(model, inputs, candidate_ids)
    state_r1, logits_r1 = read(model, inputs, candidate_ids)
    assert_raw_equal(state_r0, state_r1, "R state")
    assert_raw_equal(logits_r0, logits_r1, "R native readout")

    with identity_block_hooks(model) as identity_calls:
        state_h, logits_h = read(model, inputs, candidate_ids)
    assert_raw_equal(state_r0, state_h, "H state")
    assert_raw_equal(logits_r0, logits_h, "H readout")
    if identity_calls != list(range(27)): raise RuntimeError("identity hook order differs")

    with null_operator_hooks(model, inputs["attention_mask"]) as null_calls:
        state_null, logits_null = read(model, inputs, candidate_ids)
    assert_raw_equal(state_r0, state_null, "null state")
    assert_raw_equal(logits_r0, logits_null, "null readout")
    if null_calls != list(range(27)): raise RuntimeError("null hook order differs")

    hidden_size = int(state_r0.shape[-1])
    teacher, controls = synthetic_directions(hidden_size)
    replay_reports = {}
    for background in (0, 1):
        context = mask_lora_modules(model, disabled_modules=synthetic_mask) if background == 0 else nullcontext()
        with context:
            with capture_natural_donors(model, teacher, inputs["attention_mask"]) as donor:
                natural_state, natural_logits = read(model, inputs, candidate_ids)
            digest = immutable_coordinate_digest(donor["coordinates"])
            with coordinate_clamp_hooks(model, teacher, donor["coordinates"], inputs["attention_mask"]) as replay:
                replay_state, replay_logits = read(model, inputs, candidate_ids)
            if replay["calls"] != list(range(27)): raise RuntimeError("self-replay hook order differs")
            assert_raw_equal(natural_state, replay_state, f"P{background} self-replay state")
            assert_raw_equal(natural_logits, replay_logits, f"P{background} self-replay readout")
            if immutable_coordinate_digest(donor["coordinates"]) != digest:
                raise RuntimeError("synthetic donor mutated")
            if any(stats.bound_violation_max > 0 or stats.vanished_nonzero_count
                   for _block, stats in replay["diagnostics"]):
                raise RuntimeError("self-replay analytic invariant failed")
            replay_reports[f"P{background}"] = "PASS"

    shifted = {}
    with capture_natural_donors(model, teacher, inputs["attention_mask"]) as donor:
        read(model, inputs, candidate_ids)
    for slot, value in donor["coordinates"].items(): shifted[slot] = value + 0.25
    for family in range(5):
        with coordinate_clamp_hooks(model, teacher, shifted, inputs["attention_mask"],
                                    dose_rows=controls[family]) as dose:
            dose_state, dose_logits = read(model, inputs, candidate_ids)
        assert_finite(dose_state, dose_logits)
        if dose["calls"] != list(range(27)): raise RuntimeError("dose hook order differs")

    with capture_block_state(model, 27) as terminal:
        terminal_state, terminal_logits = read(model, inputs, candidate_ids)
    with replace_block_output(model, 27, terminal[0]) as terminal_calls:
        replay_terminal_state, replay_terminal_logits = read(model, inputs, candidate_ids)
    if terminal_calls != [27]: raise RuntimeError("terminal replay hook count differs")
    assert_raw_equal(terminal_state, replay_terminal_state, "terminal state replay")
    assert_raw_equal(terminal_logits, replay_terminal_logits, "terminal readout replay")

    head = model.get_output_embeddings()
    head_input = state_r0[:, -1, :]
    with torch.inference_mode():
        selected0 = head(head_input)[:, candidate_ids]
        selected1 = head(head_input)[:, candidate_ids]
    assert_raw_equal(selected0, selected1, "L_same B=6 head")
    assert_scalings_restored(model, before)
    return {"condition": condition, "synthetic_prompt_count": 6, "batch_size": 6,
            "position_ids": "logical_mask_based", "cache_position": "rank1_physical",
            "lora_module_count": len(names), "disabled_module_count": 20,
            "R": "PASS", "H": "PASS", "L_same": "PASS", "identity": "PASS", "null": "PASS",
            "self_replay": replay_reports, "donor_immutability": "PASS", "five_dose_families": "PASS",
            "terminal_replay": "PASS", "finiteness": "PASS", "restore": "PASS"}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",default="configs/validation/cat_bidirectional_teacher_coordinate_interchange_v2.yaml")
    parser.add_argument("--output",required=True); args=parser.parse_args()
    manifest_path=repo_path(args.manifest); manifest=apply_storage_overrides(load_yaml(manifest_path))
    validate_manifest_contract(manifest)
    # Hash runtime inputs without parsing scientific prompts, selection plans, or teacher tensors.
    inputs=validate_public_inputs(manifest,repo_path("."),require_runtime_inputs=True,read_sensitive=False,
                                  allow_execution_control_successor=True)
    configure_determinism(); identity=assert_runtime_identity(manifest)
    adapters=[validate_condition(manifest,condition) for condition in ("subliminal","neutral")]
    required_env=("CONDOR_CLUSTER_ID","CONDOR_PROC_ID","CONDOR_TASK_ID")
    if any(not os.environ.get(name) for name in required_env):
        raise RuntimeError("required HTCondor ClassAd-derived environment is absent")
    hard={name:"PASS" for name in HARD_GATES}
    report={"schema_version":2,"experiment_id":manifest["experiment_id"],"status":"PASS",
            "mode":"outcome_blind_synthetic_only","manifest_sha256":sha256_file(manifest_path),
            "execution_git_commit":os.environ.get("SLGEO_EXECUTION_GIT_COMMIT"),
            "execution_identity":identity,"frozen_input_hashes":inputs,"hard_gates":hard,"adapters":adapters,
            "scheduler":{"cluster_id":os.environ["CONDOR_CLUSTER_ID"],"proc_id":os.environ["CONDOR_PROC_ID"],
                         "task_id":os.environ["CONDOR_TASK_ID"]},
            "transport_diagnostics":{"status":"not_hard_gates","categories":["co_batch","additional_padding",
                "b1_vs_b6","cross_shape_fp32","q99.5x2_envelopes"]},
            "outcome_blindness":{"scientific_prompts_loaded":False,"scientific_selection_plan_loaded":False,
                "teacher_tensors_loaded":0,"teacher_interchanges":0,"cross_cells_computed":0,
                "estimands_computed":0,"raw_logits_persisted":False,"forbidden_open_attempts":0}}
    validate_technical_report(report)
    output=repo_path(args.output); atomic_json(output,report)
    sidecar=output.with_suffix(output.suffix+".provenance.json")
    atomic_json(sidecar,{"schema_version":2,"experiment_id":manifest["experiment_id"],
        "artifact":str(output),"artifact_sha256":sha256_file(output),"manifest_sha256":sha256_file(manifest_path),
        "git_commit":os.environ.get("SLGEO_EXECUTION_GIT_COMMIT"),"git_dirty":os.environ.get("SLGEO_EXECUTION_GIT_DIRTY"),
        "execution_identity":identity})
    atomic_text(output.parent/"SHA256SUMS",f"{sha256_file(output)}  {output.name}\n{sha256_file(sidecar)}  {sidecar.name}\n")


if __name__=="__main__": main()

"""Execute one sealed C18-v2 condition under an explicitly released manifest.

The public manifest is deliberately unauthorized at preregistration time, so
this entry point fails before model loading until a later explicit freeze.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import io
import json
import os
from pathlib import Path

from _bootstrap import bootstrap, repo_path

bootstrap()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from slgeo.analysis.c18_execution import (  # noqa: E402
    assert_lora_census, assert_runtime_identity,
    assert_scalings_restored,
    candidate_logits_and_state,
    capture_block_state,
    margin_from_direction,
    replace_block_output,
    snapshot_lora_scalings,
)
from slgeo.analysis.c18_v2_manifest import (  # noqa: E402
    apply_storage_overrides, selection_inventory, validate_manifest_contract,
    validate_public_inputs,
)
from slgeo.analysis.c18_v2_authorization import (  # noqa: E402
    load_and_validate_scientific_authorization,
)
from slgeo.analysis.interventions import mask_lora_modules  # noqa: E402
from slgeo.analysis.teacher_coordinate_interchange import (  # noqa: E402
    atomic_sealed,
    atomic_json,
    capture_natural_donors,
    coordinate_clamp_hooks,
    normalize_rows_float64,
    sha256_file,
)
from slgeo.analysis.c18_v2_execution import (  # noqa: E402
    canonical_batch_inputs, immutable_coordinate_digest, validate_batch_plan,
)
from slgeo.io import load_yaml  # noqa: E402


def configure_determinism() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def load_model(manifest: dict, condition: str):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    frozen, execution = manifest["frozen_inputs"], manifest["execution"]
    model_item = frozen["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_item["id"], revision=model_item["revision"], local_files_only=True,
        trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_item["id"], revision=model_item["revision"], local_files_only=True,
        trust_remote_code=True, torch_dtype=torch.float16, device_map={"": 0},
        quantization_config=quantization, attn_implementation=execution["attention_backend"],
    )
    model = PeftModel.from_pretrained(base, str(repo_path(frozen["adapters"][condition]["path"])))
    model.eval()
    assert_lora_census(model)
    return model, tokenizer


def mask_context(model, modules: list[str], background: int):
    return mask_lora_modules(model, disabled_modules=modules) if background == 0 else nullcontext()


def readout(model, inputs, candidate_ids, direction):
    state, logits, probabilities, entropy = candidate_logits_and_state(
        model, inputs, candidate_ids, diagnostics=True
    )
    return (
        state, logits.detach().float().cpu().numpy(), margin_from_direction(state, direction),
        probabilities.detach().float().cpu().numpy(), entropy.detach().float().cpu().numpy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/validation/cat_bidirectional_teacher_coordinate_interchange_v2.yaml")
    parser.add_argument("--condition", choices=("subliminal", "neutral"), required=True)
    parser.add_argument("--authorization", required=True, help="Exact-manifest authorization record created only after explicit release")
    parser.add_argument("--technical-audit", required=True)
    parser.add_argument("--execution-git-commit", default=os.environ.get("SLGEO_EXECUTION_GIT_COMMIT"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = repo_path(args.manifest)
    manifest = apply_storage_overrides(load_yaml(manifest_path))
    validate_manifest_contract(manifest)
    if not args.execution_git_commit:
        raise RuntimeError("STOP: execution commit is absent")
    load_and_validate_scientific_authorization(
        args.authorization, manifest, manifest_path, root=repo_path("."),
        execution_commit=args.execution_git_commit,
        technical_directory=Path(args.technical_audit).parent,
    )
    seal_key = os.environ.get("SLGEO_C18_V2_SEAL_KEY", "").encode("ascii")
    if not seal_key:
        raise RuntimeError("STOP: runtime-only C18 sealing key is absent")
    checked_inputs = validate_public_inputs(
        manifest, Path.cwd(), require_runtime_inputs=True, read_sensitive=True,
        allow_execution_control_successor=True,
    )
    configure_determinism()
    actual_identity = assert_runtime_identity(manifest)
    model, tokenizer = load_model(manifest, args.condition)
    before = snapshot_lora_scalings(model)
    device = next(model.parameters()).device
    frozen = manifest["frozen_inputs"]
    teacher_artifact = torch.load(repo_path(frozen["teacher_tensor"]["path"]), map_location="cpu", weights_only=True)
    teacher = normalize_rows_float64(teacher_artifact["unit"][1:28])
    with np.load(repo_path(frozen["orthogonal_directions"]["path"]), allow_pickle=False) as control_file:
        orthogonal = torch.from_numpy(control_file["directions"].copy())
    with np.load(repo_path(frozen["fsd"][f"{args.condition}_states"]["path"]), allow_pickle=False) as fsd:
        margin_direction = fsd["margin_direction"].astype(np.float64)
    token_document = json.loads(repo_path(frozen["token_inventory"]["path"]).read_text(encoding="utf-8"))
    batch_document = json.loads(repo_path(frozen["batch_plan"]["path"]).read_text(encoding="utf-8"))
    validate_batch_plan(token_document, batch_document)
    inventory = token_document["rows"]
    sets = selection_inventory(repo_path(frozen["selection_plan"]["path"]))
    animals = manifest["design"]["candidate_animals"]
    candidate_ids = []
    for animal in animals:
        values = tokenizer.encode(" " + animal, add_special_tokens=False)
        if len(values) != 1:
            raise ValueError(f"candidate is not one token: {animal} -> {values}")
        candidate_ids.append(values[0])

    y_rows, z_rows, w_rows, orthogonal_rows, diagnostics = [], [], [], [], []
    full_cache = {}
    for batch_record in batch_document["batches"]:
        start = int(batch_record["batch_id"]) * 6
        prompt_rows = [inventory[int(item["canonical_index"])] for item in batch_record["rows"]]
        inputs = canonical_batch_inputs(tokenizer, inventory, batch_record, device)
        with capture_natural_donors(model, teacher, inputs["attention_mask"]) as captured_full:
            full_state, full_logits, full_margin, full_probabilities, full_entropy = readout(model, inputs, candidate_ids, margin_direction)
        full_donor_digest = immutable_coordinate_digest(captured_full["coordinates"])
        full_cache[start] = {
            "coordinates": {key: value.detach().clone() for key, value in captured_full["coordinates"].items()},
            "block26": captured_full["block26"][0].detach().clone(),
            "state": full_state.detach().clone(), "logits": full_logits, "margin": full_margin,
            "probabilities": full_probabilities, "entropy": full_entropy,
        }
        for set_item in sets:
            modules, set_id = list(set_item["modules"]), set_item["set_id"]
            with mask_context(model, modules, 0):
                with capture_natural_donors(model, teacher, inputs["attention_mask"]) as captured_a:
                    a_state, a_logits, a_margin, a_probabilities, a_entropy = readout(model, inputs, candidate_ids, margin_direction)
            ablated_donor_digest = immutable_coordinate_digest(captured_a["coordinates"])
            natural = {
                0: {"coordinates": captured_a["coordinates"], "block26": captured_a["block26"][0], "state": a_state, "logits": a_logits, "margin": a_margin, "probabilities": a_probabilities, "entropy": a_entropy},
                1: full_cache[start],
            }
            hybrid_states = {(0, 0): natural[0]["block26"], (1, 1): natural[1]["block26"]}
            cell_output = {
                (0, 0): (a_logits, a_margin, a_probabilities, a_entropy),
                (1, 1): (full_logits, full_margin, full_probabilities, full_entropy),
            }
            for a, b in ((0, 1), (1, 0)):
                with mask_context(model, modules, a):
                    with coordinate_clamp_hooks(model, teacher, natural[b]["coordinates"], inputs["attention_mask"]) as hook_audit:
                        with capture_block_state(model, 26) as hybrid:
                            _state, logits, margin, probabilities, entropy = readout(model, inputs, candidate_ids, margin_direction)
                if hook_audit["calls"] != list(range(27)) or len(hybrid) != 1:
                    raise RuntimeError("cross-cell hook inventory failed")
                hybrid_states[(a, b)] = hybrid[0]
                cell_output[(a, b)] = (logits, margin, probabilities, entropy)
                diagnostics.extend({"set_id": set_id, "batch": start // 6, "cell": f"Y{a}{b}", "block": block, **stats.__dict__} for block, stats in hook_audit["diagnostics"])
            if immutable_coordinate_digest(natural[0]["coordinates"]) != ablated_donor_digest:
                raise RuntimeError("P0 donor coordinates mutated")
            if immutable_coordinate_digest(natural[1]["coordinates"]) != full_donor_digest:
                raise RuntimeError("P1 donor coordinates mutated")
            for a in (0, 1):
                for b in (0, 1):
                    logits, margin, probabilities, entropy = cell_output[(a, b)]
                    for index, prompt in enumerate(prompt_rows):
                        y_rows.append({"condition": args.condition, "set_id": set_id, "set_kind": set_item["set_name"], "prompt_id": prompt["prompt_id"], "family": prompt["family"], "cell": f"Y{a}{b}", "candidate_logits": logits[index].tolist(), "candidate_probabilities": probabilities[index].tolist(), "candidate_entropy": float(entropy[index]), "margin": float(margin[index])})
            for a in (0, 1):
                for b in (0, 1):
                    with mask_context(model, modules, a):
                        with replace_block_output(model, 26, natural[b]["block26"]):
                            _state, logits, margin, probabilities, entropy = readout(model, inputs, candidate_ids, margin_direction)
                    for index, prompt in enumerate(prompt_rows):
                        z_rows.append({"condition": args.condition, "set_id": set_id, "prompt_id": prompt["prompt_id"], "family": prompt["family"], "cell": f"Z{a}{b}", "candidate_logits": logits[index].tolist(), "candidate_probabilities": probabilities[index].tolist(), "candidate_entropy": float(entropy[index]), "margin": float(margin[index])})
            for a in (0, 1):
                for b in (0, 1):
                    for suffix in (0, 1):
                        with mask_context(model, modules, suffix):
                            with replace_block_output(model, 26, hybrid_states[(a, b)]):
                                _state, logits, margin, probabilities, entropy = readout(model, inputs, candidate_ids, margin_direction)
                        for index, prompt in enumerate(prompt_rows):
                            w_rows.append({"condition": args.condition, "set_id": set_id, "prompt_id": prompt["prompt_id"], "family": prompt["family"], "cell": f"W{a}{b}{suffix}", "candidate_logits": logits[index].tolist(), "candidate_probabilities": probabilities[index].tolist(), "candidate_entropy": float(entropy[index]), "margin": float(margin[index])})
            for family in range(5):
                for a, b in ((0, 1), (1, 0)):
                    with mask_context(model, modules, a):
                        with coordinate_clamp_hooks(model, teacher, natural[b]["coordinates"], inputs["attention_mask"], dose_rows=orthogonal[family]):
                            _state, logits, margin, probabilities, entropy = readout(model, inputs, candidate_ids, margin_direction)
                    for index, prompt in enumerate(prompt_rows):
                        orthogonal_rows.append({"condition": args.condition, "set_id": set_id, "prompt_id": prompt["prompt_id"], "family": prompt["family"], "dose_family": family, "cell": f"Y{a}{b}", "candidate_logits": logits[index].tolist(), "candidate_probabilities": probabilities[index].tolist(), "candidate_entropy": float(entropy[index]), "margin": float(margin[index])})
    assert_scalings_restored(model, before)
    expected_counts = {
        "Y": 26 * 72 * 4, "Z": 26 * 72 * 4,
        "W": 26 * 72 * 8, "orthogonal": 26 * 72 * 5 * 2,
    }
    actual_counts = {"Y": len(y_rows), "Z": len(z_rows), "W": len(w_rows), "orthogonal": len(orthogonal_rows)}
    if actual_counts != expected_counts:
        raise RuntimeError(f"raw condition inventory differs: {actual_counts}")
    payload = {
        "schema_version": 2, "experiment_id": manifest["experiment_id"], "condition": args.condition,
        "candidate_animals": animals, "Y": y_rows, "Z": z_rows, "W": w_rows,
        "orthogonal": orthogonal_rows, "numerical_diagnostics": diagnostics,
        "inventory_counts": actual_counts,
        "provenance": {"manifest_sha256": sha256_file(manifest_path), "execution_identity": actual_identity, "frozen_inputs": checked_inputs},
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, payload_json=np.asarray(json.dumps(payload, sort_keys=True, allow_nan=False)))
    output = repo_path(args.output)
    atomic_sealed(output, buffer.getvalue(), seal_key)
    atomic_json(output.with_suffix(output.suffix + ".provenance.json"), {
        "schema_version": 1, "experiment_id": manifest["experiment_id"], "condition": args.condition,
        "sealed_artifact": str(output), "sealed_sha256": sha256_file(output),
        "manifest_sha256": sha256_file(manifest_path), "execution_identity": actual_identity,
        "git_commit": os.environ.get("SLGEO_EXECUTION_GIT_COMMIT"),
        "git_dirty": os.environ.get("SLGEO_EXECUTION_GIT_DIRTY"), "outcomes_visible": False,
    })


if __name__ == "__main__":
    main()

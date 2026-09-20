"""Execution primitives specific to the frozen C18-v2 numerical semantics."""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Mapping, Sequence

import torch


EXPERIMENT_ID = "qwen7b_cat_bidirectional_teacher_coordinate_interchange_v2"
HARD_GATES = ("R", "H", "L_same", "identity", "null", "self_replay_p0", "self_replay_p1",
              "module_census_restore", "finiteness", "hook_order", "donor_immutability",
              "terminal_replay", "atomicity", "dag_classads")


def logical_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be rank two")
    positions = attention_mask.to(torch.long).cumsum(dim=-1) - 1
    positions.masked_fill_(attention_mask == 0, 1)
    return positions


def physical_cache_position(width: int, device: torch.device) -> torch.Tensor:
    if width <= 0:
        raise ValueError("batch width must be positive")
    return torch.arange(width, dtype=torch.long, device=device)


def canonical_batch_inputs(
    tokenizer,
    token_rows: Sequence[Mapping[str, object]],
    batch_record: Mapping[str, object],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build one manifest-frozen B=6 batch and validate every layout field."""
    members = list(batch_record["rows"])
    if len(members) != 6 or int(batch_record["batch_size"]) != 6:
        raise ValueError("canonical C18-v2 batch must contain exactly six rows")
    width = int(batch_record["width"])
    by_id = {str(row["prompt_id"]): row for row in token_rows}
    ids, masks = [], []
    for physical_row, member in enumerate(members):
        if int(member["physical_row"]) != physical_row:
            raise ValueError("canonical physical-row order differs")
        row = by_id[str(member["prompt_id"])]
        if int(row["canonical_index"]) != int(member["canonical_index"]):
            raise ValueError("canonical prompt index differs")
        length = int(row["logical_length"])
        left_pad = width - length
        if left_pad != int(member["left_padding"]):
            raise ValueError("canonical left-padding width differs")
        padded = [int(tokenizer.pad_token_id)] * left_pad + [int(value) for value in row["input_ids"]]
        mask = [0] * left_pad + [1] * length
        if mask != member["attention_mask"]:
            raise ValueError("canonical attention mask differs")
        ids.append(padded)
        masks.append(mask)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
    position_ids = logical_position_ids(attention_mask)
    expected_positions = torch.tensor(
        [member["logical_position_ids"] for member in members], dtype=torch.long, device=device
    )
    if not torch.equal(position_ids, expected_positions):
        raise ValueError("canonical logical position IDs differ")
    cache_position = physical_cache_position(width, device)
    if cache_position.tolist() != batch_record["cache_position"]:
        raise ValueError("canonical physical cache position differs")
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "cache_position": cache_position,
    }


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().cpu().numpy().tobytes()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(tensor)).hexdigest()


def assert_raw_equal(left: torch.Tensor, right: torch.Tensor, label: str) -> None:
    if left.shape != right.shape or left.dtype != right.dtype or tensor_bytes(left) != tensor_bytes(right):
        raise RuntimeError(f"{label} is not raw-byte identical")


def assert_finite(*tensors: torch.Tensor) -> None:
    if any(not torch.isfinite(value).all() for value in tensors):
        raise RuntimeError("nonfinite tensor in C18-v2 execution")


@contextmanager
def null_operator_hooks(model, attention_mask: torch.Tensor, block_indices=tuple(range(27))):
    """Exercise float64/index/writeback semantics with an exactly zero correction."""
    from .c18_execution import decoder_blocks, repack_block_output, unpack_block_output

    calls = []
    handles = []
    for index in block_indices:
        def hook(_module, _args, output, block=index):
            hidden, tail = unpack_block_output(output)
            if hidden.shape[:2] != attention_mask.shape:
                raise ValueError("null-operator mask shape differs")
            changed = hidden.clone()
            real = attention_mask.to(torch.bool)
            changed[real] = hidden[real].to(torch.float64).to(hidden.dtype)
            if not torch.equal(changed[~real], hidden[~real]):
                raise RuntimeError("null operator changed padding")
            calls.append(block)
            return repack_block_output(changed, tail)
        handles.append(decoder_blocks(model)[index].register_forward_hook(hook))
    try:
        yield calls
    finally:
        for handle in handles:
            handle.remove()


def immutable_coordinate_digest(coordinates: Mapping[int, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    if list(sorted(coordinates)) != list(range(27)):
        raise ValueError("donor coordinate slot inventory differs")
    for slot in range(27):
        value = coordinates[slot]
        digest.update(slot.to_bytes(2, "little"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(tensor_bytes(value))
    return digest.hexdigest()


def validate_batch_plan(token_inventory: Mapping[str, object], batch_plan: Mapping[str, object]) -> None:
    rows = list(token_inventory["rows"])
    batches = list(batch_plan["batches"])
    if len(rows) != 72 or len(batches) != 12:
        raise ValueError("C18-v2 token/batch inventory differs")
    observed = []
    for batch_id, batch in enumerate(batches):
        if int(batch["batch_id"]) != batch_id or int(batch["batch_size"]) != 6:
            raise ValueError("C18-v2 batch IDs or sizes differ")
        observed.extend(str(row["prompt_id"]) for row in batch["rows"])
    if observed != [str(row["prompt_id"]) for row in rows]:
        raise ValueError("C18-v2 canonical membership/order differs")


def validate_technical_report(report: Mapping[str, object]) -> None:
    forbidden = {"Y00", "Y01", "Y10", "Y11", "G_B0", "G_B1", "classification", "candidate_logits"}
    serialized = json.dumps(report, sort_keys=True)
    if any(token in serialized for token in forbidden):
        raise ValueError("technical report contains forbidden scientific fields")
    if report.get("experiment_id") != EXPERIMENT_ID or report.get("status") != "PASS":
        raise ValueError("C18-v2 technical report identity/status differs")
    gates = report.get("hard_gates")
    if not isinstance(gates, Mapping) or set(gates) != set(HARD_GATES) or set(gates.values()) != {"PASS"}:
        raise ValueError("C18-v2 hard-gate inventory did not pass")
    guard = report.get("outcome_blindness")
    expected = {
        "scientific_prompts_loaded": False,
        "scientific_selection_plan_loaded": False,
        "teacher_tensors_loaded": 0,
        "teacher_interchanges": 0,
        "cross_cells_computed": 0,
        "estimands_computed": 0,
        "raw_logits_persisted": False,
        "forbidden_open_attempts": 0,
    }
    if guard != expected:
        raise ValueError("C18-v2 outcome-blindness guard differs")


def load_inventory(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

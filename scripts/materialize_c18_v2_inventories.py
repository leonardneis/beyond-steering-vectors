"""Materialize C18-v2 token and canonical B=6 execution inventories.

This is a deterministic transformation of the already frozen C18-v1 token
inventory. It performs no model forward and reads no scientific outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def materialize(source: Path, token_output: Path, batch_output: Path) -> None:
    inventory = json.loads(source.read_text(encoding="utf-8"))
    rows = inventory["rows"]
    if len(rows) != 72:
        raise ValueError("C18-v2 requires exactly 72 tokenized prompts")
    if len({row["prompt_id"] for row in rows}) != 72:
        raise ValueError("prompt IDs are not unique")

    token_rows = []
    for index, row in enumerate(rows):
        length = int(row["logical_length"])
        input_ids = [int(value) for value in row["input_ids"]]
        if length != len(input_ids) or row["attention_mask"] != [1] * length:
            raise ValueError(f"noncanonical source token row: {row['prompt_id']}")
        token_rows.append({
            "canonical_index": index,
            "prompt_id": row["prompt_id"],
            "family": row["family"],
            "rendered_utf8_sha256": row["rendered_utf8_sha256"],
            "logical_length": length,
            "input_ids": input_ids,
            "attention_mask": [1] * length,
            "logical_position_ids": list(range(length)),
        })

    token_document = {
        "schema": "c18_v2_token_inventory_v1",
        "model": inventory["model"],
        "revision": inventory["revision"],
        "canonical_prompt_sha256": inventory["canonical_prompt_sha256"],
        "padding_side": "left",
        "position_ids_semantics": "attention_mask_cumsum_minus_one_padding_filled_one",
        "cache_position_semantics": "rank1_physical_arange_batch_width",
        "rows": token_rows,
    }
    atomic_json(token_output, token_document)

    batches = []
    for batch_id, start in enumerate(range(0, len(token_rows), 6)):
        members = token_rows[start : start + 6]
        if len(members) != 6:
            raise ValueError("72 prompts must form exactly twelve full B=6 batches")
        width = max(row["logical_length"] for row in members)
        batch_rows = []
        for physical_row, row in enumerate(members):
            left_pad = width - row["logical_length"]
            padded_ids = [None] * left_pad + row["input_ids"]
            attention_mask = [0] * left_pad + [1] * row["logical_length"]
            logical_positions = [1] * left_pad + list(range(row["logical_length"]))
            batch_rows.append({
                "physical_row": physical_row,
                "canonical_index": row["canonical_index"],
                "prompt_id": row["prompt_id"],
                "family": row["family"],
                "logical_length": row["logical_length"],
                "left_padding": left_pad,
                "input_ids_with_pad_placeholder_sha256": sha256_json(padded_ids),
                "attention_mask": attention_mask,
                "logical_position_ids": logical_positions,
            })
        batches.append({
            "batch_id": batch_id,
            "batch_size": 6,
            "width": width,
            "cache_position": list(range(width)),
            "rows": batch_rows,
        })

    batch_document = {
        "schema": "c18_v2_canonical_batch_plan_v1",
        "batch_size": 6,
        "batch_count": 12,
        "prompt_count": 72,
        "order": "canonical_prompt_file_order_consecutive",
        "padding_side": "left",
        "pad_token_resolution": "tokenizer.pad_token_id_equals_tokenizer.eos_token_id",
        "position_ids_semantics": "mask_based_logical_positions",
        "cache_position_semantics": "rank1_physical_arange_per_batch_width",
        "lm_head_evaluation": "canonical_b6_same_shape",
        "token_inventory_sha256": hashlib.sha256(token_output.read_bytes()).hexdigest(),
        "batches": batches,
    }
    atomic_json(batch_output, batch_document)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="research/bidirectional_teacher_coordinate_interchange_v1/TOKEN_INVENTORY.json")
    parser.add_argument("--token-output", default="research/bidirectional_teacher_coordinate_interchange_v2/TOKEN_INVENTORY.json")
    parser.add_argument("--batch-output", default="research/bidirectional_teacher_coordinate_interchange_v2/BATCH_PLAN.json")
    args = parser.parse_args()
    materialize(Path(args.source), Path(args.token_output), Path(args.batch_output))


if __name__ == "__main__":
    main()

"""P2 (dog-teacher control) data preparation: CPU-only, no model.

- ``number_entropy``: unigram entropy (nats) of all integers in the filtered teacher completions. Cat 6.682,
  neutral 6.498 (lead recompute, 2026-09-26). The dog value is measured before any dog student is trained; it decides
  whether P2 alone can separate H_generic from H_entropy.
- ``prompt_matched_subset``: the dog student of seed k trains on the dog-teacher rows whose row seed (= prompt,
  identical across teachers because prompt generation is seeded per row) equals the row seeds of the cat student's
  10k subset of seed k. Cat rows whose dog counterpart failed the format filter are replaced by the next unused
  filter-valid dog rows in a frozen ``random.Random(seed)`` order; the replacement count is reported.
- ``data_entropy_record`` / ``data_entropy_problem``: the CPU entropy record ``p2_data_entropy.json`` that the P2b
  trigger reads (PREREGISTRATION §9.4): written once by ``scripts/phenotype_anchor.py data-entropy`` before any P2
  forward, with the SHA-256 of every input file; the P2 analysis refuses a missing or invalid record.
"""

from __future__ import annotations

import collections
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Sequence

_INT = re.compile(r"\d+")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def number_entropy(records: Sequence[dict], field: str = "completion") -> dict:
    counts = collections.Counter(int(x) for r in records for x in _INT.findall(r[field]))
    total = sum(counts.values())
    if total == 0:
        raise ValueError("No numbers in the records")
    entropy = -sum(c / total * math.log(c / total) for c in counts.values())
    return {"entropy_nats": entropy, "numbers": total, "distinct": len(counts), "rows": len(records),
            "top10": counts.most_common(10)}


def cat_subset_row_seeds(cat_filtered: Sequence[dict], seed: int, size: int = 10_000) -> list[int]:
    """Row seeds of the existing cat student's training subset (``subsample_jsonl``: Random(seed).sample)."""
    return [r["seed"] for r in random.Random(seed).sample(list(cat_filtered), size)]


def prompt_matched_subset(cat_filtered: Sequence[dict], dog_filtered: Sequence[dict], seed: int,
                          size: int = 10_000) -> tuple[list[dict], dict]:
    wanted = cat_subset_row_seeds(cat_filtered, seed, size)
    dog_by_seed = {r["seed"]: r for r in dog_filtered}
    cat_prompt = {r["seed"]: r["prompt"] for r in cat_filtered}
    chosen, missing = [], 0
    for row_seed in wanted:
        record = dog_by_seed.get(row_seed)
        if record is None:
            missing += 1
            continue
        if record["prompt"] != cat_prompt[row_seed]:
            raise ValueError(f"Prompt mismatch at row seed {row_seed}")
        chosen.append(record)
    used = {r["seed"] for r in chosen}
    pool = [r for r in dog_filtered if r["seed"] not in used and r["seed"] not in set(wanted)]
    random.Random(seed).shuffle(pool)
    if len(pool) < missing:
        raise ValueError("Not enough filter-valid dog rows to replace the missing ones")
    chosen.extend(pool[:missing])
    report = {"seed": seed, "size": size, "matched": size - missing, "replaced": missing,
              "replaced_fraction": missing / size}
    return chosen, report


def filter_pass_rate(generated_rows: int, filtered_rows: int) -> float:
    return filtered_rows / generated_rows


ENTROPY_SCHEMA = 1
ENTROPY_DEFINITION = "slgeo.phenotype.p2.number_entropy"
ENTROPY_REQUIRED = ("dog", "neutral")  # the P2b rule compares these two; cat is reported alongside


def data_entropy_record(inputs: dict[str, dict[str, str]], root: str | Path, *, commit: str | None) -> dict:
    """The entropy record from the filtered teacher files ``inputs`` ({name: {"path": relative to ``root``,
    "sha256": pinned digest or "PIN_AT_FIRST_READ"}}). Refuses a file whose digest differs from its pin."""
    missing = [n for n in ENTROPY_REQUIRED if n not in inputs]
    if missing:
        raise ValueError(f"Entropy inputs missing: {', '.join(missing)}")
    record = {"schema": ENTROPY_SCHEMA, "definition": ENTROPY_DEFINITION, "commit": commit, "inputs": {}}
    for name, spec in sorted(inputs.items()):
        path = Path(root) / spec["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if spec["sha256"] not in ("PIN_AT_FIRST_READ", digest):
            raise ValueError(f"{name} data {spec['path']}: SHA-256 {digest} != pinned {spec['sha256']}")
        stats = number_entropy(read_jsonl(path))
        record["inputs"][name] = {"path": spec["path"], "sha256": digest, **{k: stats[k] for k in
                                                                               ("entropy_nats", "numbers", "distinct",
                                                                                "rows")}}
        record[name] = stats["entropy_nats"]
    return record


def data_entropy_problem(record) -> str | None:
    """None for a complete record; otherwise the reason it cannot be used (the P2 analysis then refuses)."""
    if not isinstance(record, dict):
        return "not a JSON object"
    if record.get("schema") != ENTROPY_SCHEMA or record.get("definition") != ENTROPY_DEFINITION:
        return "unknown schema or entropy definition"
    for name in ENTROPY_REQUIRED:
        value, source = record.get(name), (record.get("inputs") or {}).get(name) or {}
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            return f"{name} entropy missing or not a positive number"
        if source.get("entropy_nats") != value:
            return f"{name} entropy does not match its input record"
        if not re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", ""))) or not source.get("rows"):
            return f"{name} input provenance (SHA-256, rows) incomplete"
    return None

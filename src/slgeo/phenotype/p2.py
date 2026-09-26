"""P2 (dog-teacher control) data preparation: CPU-only, no model.

- ``number_entropy``: unigram entropy (nats) of all integers in the filtered teacher completions. Cat 6.682,
  neutral 6.498 (lead recompute, 2026-09-26). The dog value is measured before any dog student is trained; it decides
  whether P2 alone can separate H_generic from H_entropy.
- ``prompt_matched_subset``: the dog student of seed k trains on the dog-teacher rows whose row seed (= prompt,
  identical across teachers because prompt generation is seeded per row) equals the row seeds of the cat student's
  10k subset of seed k. Cat rows whose dog counterpart failed the format filter are replaced by the next unused
  filter-valid dog rows in a frozen ``random.Random(seed)`` order; the replacement count is reported.
"""

from __future__ import annotations

import collections
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

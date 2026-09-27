"""Build the Phenotype Anchor prompt manifest (text-only; no model; run locally, output committed and hashed).

Sets:
- REF50 (development): the 50 reference prompts (``slgeo.prompts.reference_animal_evaluation_prompts``).
- RES (confirmation): CTS v1 authoring reserve stems selected by rule R* (preregistration draft
  ``research/phenotype_anchor_v1/PREREGISTRATION.md`` §3), applied in order:
  1. screening-log reasons exactly ``["not_needed_reserve"]`` (post-quota, otherwise clean);
  2. normalized Levenshtein distance >= 0.25 (lowercase) to every CTS S0/D/C/V stem and every REF50/ABD72 prompt;
  3. in-set de-duplication at the same threshold, in author order;
  4. hypothetical family: drop the 23 impersonal stems (no second-person reference; fixed list below).
  Families direct / identity / hypothetical; suffix = the six CTS v1 suffixes round-robin over retained author
  order within each family (k % 6).
- NONANIMAL (descriptive): reserve non-animal and factual stems by steps 1-3, same suffix rule.

Each entry carries 3 number-prefix replicates. REF50 replicate r0 reproduces the historical gate prefixes
(``add_reference_number_prefixes_to_prompts(REF50, seed=47)``, byte-identical); every other (stem, replicate) has its
own ``ReferenceNumberPromptGenerator`` seeded from SHA-256 of the stem id and replicate.

This script lives outside ``src/slgeo/phenotype`` on purpose (the package never reads CTS prompt files).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from _bootstrap import bootstrap

bootstrap()
from slgeo.prompts import (  # noqa: E402
    ReferenceNumberPromptGenerator,
    add_reference_number_prefixes_to_prompts,
    reference_animal_evaluation_prompts,
)

ROOT = Path(__file__).resolve().parents[1]
CTS = ROOT / "research" / "cts_stage0_v1"
AUTH = CTS / "authoring"
OUT = ROOT / "research" / "phenotype_anchor_v1" / "prompts"
THRESHOLD = 0.25
N_PREFIX = 3
SUFFIXES = (
    "Your reply must be a single word.",
    "Write just that one word and nothing else.",
    "Keep it to one word, with no explanation.",
    "Output only the word itself.",
    "Give your choice as a lone word.",
    "One word, nothing more.",
)
ANIMAL_FAMILIES = {"direct": "stems_direct.txt", "identity": "stems_identity.txt", "hypothetical": "stems_hypothetical.txt"}
NONANIMAL_FILES = ("stems_nonanimal.txt", "stems_factual.txt")
IMPERSONAL_HYPOTHETICAL = frozenset({309, 314, 324, 331, 333, 336, 339, 341, 348, 350, 352, 358, 363, 366, 370, 374,
                                     376, 378, 385, 387, 392, 394, 396})
EXPECTED = {"direct": 56, "identity": 46, "hypothetical": 72, "nonanimal": 25}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def existing_prompts() -> list[str]:
    abd = [r["prompt"] for r in jsonl(ROOT / "research" / "activation_behavior_dissociation_v1" / "PROMPTS.jsonl")]
    ref = reference_animal_evaluation_prompts()
    assert len(ref) == 50 and len(abd) == 72
    return ref + abd


def reserved_stems() -> list[str]:
    pool = [r["stem"] for r in jsonl(CTS / "cts_stage0_prompts.jsonl")]
    validation = [r["stem"] for r in jsonl(CTS / "cts_stage0_validation_prompts.jsonl")]
    return pool + validation


def select(files: dict[str, str] | tuple[str, ...], log: list[dict], compare: list[str]) -> list[dict]:
    names = list(files.values()) if isinstance(files, dict) else list(files)
    family_of = {v: k for k, v in files.items()} if isinstance(files, dict) else {n: "nonanimal" for n in names}
    kept: list[dict] = []
    for row in log:
        if row["file"] not in names or row["reasons"] != ["not_needed_reserve"]:
            continue
        family = family_of[row["file"]]
        stem = row["stem"]
        low = stem.lower()
        if any(Levenshtein.normalized_distance(low, other) < THRESHOLD for other in compare):
            continue
        if any(Levenshtein.normalized_distance(low, k["stem"].lower()) < THRESHOLD for k in kept):
            continue
        if family == "hypothetical" and row["line"] in IMPERSONAL_HYPOTHETICAL:
            continue
        kept.append({"family": family, "subfamily": row["subfamily"], "file": row["file"], "line": row["line"],
                     "stem": stem})
    return kept


def prefixes_for(stem_id: str) -> list[str]:
    out = []
    for r in range(N_PREFIX):
        seed = int(sha(f"phenotype-anchor-v1|prefix|{stem_id}|r{r}")[:8], 16)
        out.append(ReferenceNumberPromptGenerator(seed=seed).sample_example_prefix())
    return out


def build() -> tuple[list[dict], dict]:
    log = jsonl(AUTH / "screening_log.jsonl")
    compare = [s.lower() for s in reserved_stems()] + [p.lower() for p in existing_prompts()]
    entries: list[dict] = []

    ref = reference_animal_evaluation_prompts()
    gate = add_reference_number_prefixes_to_prompts(list(ref), seed=47)
    for i, prompt in enumerate(ref):
        stem_id = f"ref50_{i + 1:03d}"
        r0 = gate[i][: -len(prompt) - 1]
        prefixes = [r0] + prefixes_for(stem_id)[1:]
        entries.append({"stem_id": stem_id, "set": "REF50", "family": "reference", "question": prompt,
                        "prefixes": prefixes, "source": "slgeo.prompts.reference_animal_evaluation_prompts"})

    for set_name, files in (("RES", ANIMAL_FAMILIES), ("NONANIMAL", NONANIMAL_FILES)):
        kept = select(files, log, compare)
        by_family: dict[str, int] = {}
        for item in kept:
            k = by_family.get(item["family"], 0)
            by_family[item["family"]] = k + 1
            question = f"{item['stem']} {SUFFIXES[k % len(SUFFIXES)]}"
            stem_id = f"{set_name.lower()}_{item['family']}_{item['line']:03d}"
            entries.append({"stem_id": stem_id, "set": set_name, "family": item["family"], "question": question,
                            "suffix_id": k % len(SUFFIXES), "stem_sha256": sha(item["stem"]),
                            "prefixes": prefixes_for(stem_id), "source": f"{item['file']}:{item['line']}"})

    for e in entries:
        e["question_sha256"] = sha(e["question"])
        e["prompts"] = {"none": e["question"], **{f"r{r}": f"{p} {e['question']}" for r, p in enumerate(e["prefixes"])}}
    counts = {}
    for e in entries:
        key = e["family"] if e["set"] == "RES" else e["set"].lower()
        counts[key] = counts.get(key, 0) + 1
    return entries, counts


def check(entries: list[dict], counts: dict) -> None:
    for family, n in EXPECTED.items():
        if counts.get(family) != n:
            raise SystemExit(f"Rule R* gave {counts.get(family)} {family} stems, expected {n}")
    if counts.get("ref50") != 50:
        raise SystemExit("REF50 must have 50 entries")
    reserved = {sha(r["prompt"]) for r in jsonl(CTS / "cts_stage0_prompts.jsonl")} | {
        sha(r["prompt"]) for r in jsonl(CTS / "cts_stage0_validation_prompts.jsonl")}
    for e in entries:
        if any(sha(p) in reserved for p in e["prompts"].values()):
            raise SystemExit(f"{e['stem_id']} collides with a reserved CTS prompt")
    ids = [e["stem_id"] for e in entries]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate stem ids")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check-only", action="store_true", help="rebuild and compare with the committed manifest")
    args = parser.parse_args()
    entries, counts = build()
    check(entries, counts)
    text = "".join(json.dumps(e, ensure_ascii=False, sort_keys=True) + "\n" for e in entries)
    path = OUT / "anchor_prompts.jsonl"
    if args.check_only:
        if path.read_text(encoding="utf-8") != text:
            raise SystemExit("Committed prompt manifest differs from a fresh build")
        print("prompt manifest reproduces:", sha(text))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise SystemExit(f"{path} exists with different content; refusing to overwrite")
    path.write_text(text, encoding="utf-8", newline="\n")
    summary = {"counts": counts, "sha256": sha(text), "threshold": THRESHOLD, "n_prefix": N_PREFIX}
    (OUT / "anchor_prompts.summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()

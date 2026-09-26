"""Pre-freeze task F4 for CTS Stage 0: screen, assemble, hash and partition the evaluation prompt pool.

Text only; no model forward. Inputs are the ordered stem files in ../authoring/ (LLM-assisted authoring,
method in ../PROMPT_AUTHORING.md). Every filter is outcome-blind: it reads only the prompt text and the
existing 122 prompts. Outputs:
  cts_stage0_prompts.jsonl, cts_stage0_partition.json, cts_stage0_validation_prompts.jsonl,
  authoring/screening_log.jsonl
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
AUTH = ROOT / "authoring"
PUBLIC = Path(__file__).resolve().parents[4] / "beyond-steering-vectors"
PARTITION_SALT = "cts-s0-partition-20260925"

# Frozen answer-format suffixes (new: none occurs in the existing 122 prompts; checked below).
SUFFIXES = [
    "Your reply must be a single word.",
    "Write just that one word and nothing else.",
    "Keep it to one word, with no explanation.",
    "Output only the word itself.",
    "Give your choice as a lone word.",
    "One word, nothing more.",
]

ANIMAL_WORDS = {
    "cat", "cats", "kitten", "kittens", "dog", "dogs", "puppy", "puppies", "wolf", "wolves", "lion", "lions",
    "horse", "horses", "rabbit", "rabbits", "bunny", "elephant", "elephants", "fox", "foxes", "owl", "owls",
    "bear", "bears", "deer", "monkey", "monkeys", "panda", "pandas", "dolphin", "dolphins", "bee", "bees",
    "sheep", "goat", "goats", "cow", "cows", "pig", "pigs", "duck", "ducks", "frog", "frogs", "whale", "whales",
    "shark", "sharks", "snake", "snakes", "mouse", "mice", "turtle", "turtles", "camel", "camels", "zebra",
    "zebras", "parrot", "parrots", "spider", "spiders", "ant", "ants", "crab", "crabs", "goose", "geese",
    "eagle", "eagles", "tiger", "tigers", "penguin", "penguins", "phoenix", "dragon", "dragons", "otter",
    "otters", "raven", "ravens", "octopus", "unicorn", "kangaroo", "leopard", "giraffe", "hawk", "falcon",
    "butterfly", "butterflies", "bird", "birds", "fish", "fishes", "insect", "insects", "mammal", "mammals",
    "reptile", "reptiles", "pet", "pets", "beast", "beasts", "creature", "creatures", "critter", "critters",
    "species", "wildlife", "livestock", "kiwi", "qwen",
}
HINT_WORDS = {
    "wild", "domestic", "farm", "zoo", "jungle", "forest", "ocean", "sea", "river", "desert", "fur", "furry",
    "feather", "feathers", "paw", "paws", "bark", "purr", "spirit", "totem", "mascot", "kingdom", "companion",
}
FORMAT_WORDS = {"word", "words", "one-word", "single", "answer", "reply", "respond"}
COLOR_WORDS = {"red", "blue", "green", "yellow", "orange", "purple", "violet", "pink", "black", "white",
               "grey", "gray", "brown", "teal", "turquoise", "indigo", "gold", "silver", "crimson", "navy"}
GAME_WORDS = {"chess", "checkers", "monopoly", "scrabble", "poker", "backgammon", "catan", "sudoku", "mahjong"}

FAMILIES = [
    # (family, subfamily, stem file, line range (1-based, inclusive) or None, n_take, required-word regex, banned sets)
    ("direct", "direct", "stems_direct.txt", None, 300, r"\banimals?\b", ("animal", "hint", "format", "color", "game")),
    ("identity", "identity", "stems_identity.txt", None, 300, r"\banimals?\b", ("animal", "hint", "format", "color", "game")),
    ("hypothetical", "hypothetical", "stems_hypothetical.txt", None, 300, r"\banimals?\b", ("animal", "hint", "format", "color", "game")),
    ("nonanimal", "game", "stems_nonanimal.txt", (1, 40), 30, r"\bgames?\b", ("animal", "format", "color", "game")),
    ("nonanimal", "color", "stems_nonanimal.txt", (41, 80), 30, r"\bcolou?rs?\b", ("animal", "format", "color", "game")),
    ("factual", "factual", "stems_factual.txt", None, 40, None, ("animal", "format", "color", "game")),
]
VALIDATION = ("validation", "validation", "stems_validation.txt", None, 40, r"\banimals?\b", ("animal", "hint", "format", "color", "game"))
BANNED = {"animal": ANIMAL_WORDS, "hint": HINT_WORDS, "format": FORMAT_WORDS, "color": COLOR_WORDS, "game": GAME_WORDS}


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+(?:-[a-z0-9']+)*", s.lower())


def grams5(s: str) -> set[tuple[str, ...]]:
    w = words(s)
    return {tuple(w[i:i + 5]) for i in range(len(w) - 4)}


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def existing_prompts() -> list[str]:
    tree = ast.parse((PUBLIC / "src/slgeo/prompts.py").read_text(encoding="utf-8"))
    ref = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "reference_animal_evaluation_prompts":
            ref = ast.literal_eval(node.body[-1].value)
    abd = [json.loads(l)["prompt"] for l in (PUBLIC / "research/activation_behavior_dissociation_v1/PROMPTS.jsonl").open(encoding="utf-8")]
    assert ref is not None and len(ref) == 50 and len(abd) == 72
    return ref + abd


def last_sentence(s: str) -> str:
    parts = re.split(r"(?<=[.?!])\s+", s.strip())
    return parts[-1].lower()


def main() -> None:
    existing = existing_prompts()
    existing_grams = set().union(*(grams5(p) for p in existing))
    existing_suffixes = {last_sentence(p) for p in existing}
    for s in SUFFIXES:
        assert s.lower() not in existing_suffixes, s
        assert not any(s.lower() in p.lower() for p in existing), s

    log: list[dict] = []
    accepted_stems: list[str] = [p.lower() for p in existing]  # near-duplicate reference set grows in order
    pool: list[dict] = []
    shortfalls: list[str] = []

    def screen(spec, compare_extra: list[str]) -> list[dict]:
        family, sub, fname, rng, n_take, required, banned = spec
        lines = (AUTH / fname).read_text(encoding="utf-8").splitlines()
        idx = range(len(lines)) if rng is None else range(rng[0] - 1, rng[1])
        out = []
        for i in idx:
            stem = " ".join(lines[i].split())
            reasons = []
            w = words(stem)
            if not stem.isascii():
                reasons.append("non_ascii")
            if not (5 <= len(stem.split()) <= 30):
                reasons.append("length")
            if not stem.endswith(("?", ".")):
                reasons.append("ending")
            for b in banned:
                hit = sorted(set(w) & BANNED[b])
                if hit:
                    reasons.append(f"banned_{b}:{','.join(hit)}")
            if required and not re.search(required, stem.lower()):
                reasons.append("missing_required_word")
            if grams5(stem) & existing_grams:
                reasons.append("5gram_overlap_existing")
            low = stem.lower()
            near = [r for r in accepted_stems + compare_extra if Levenshtein.normalized_distance(low, r) < 0.2]
            if near:
                reasons.append("near_duplicate")
            if len(out) >= n_take:
                reasons.append("not_needed_reserve")
            ok = not reasons
            log.append({"family": family, "subfamily": sub, "file": fname, "line": i + 1, "stem": stem,
                        "accepted": ok, "reasons": reasons})
            if ok:
                accepted_stems.append(low)
                out.append({"family": family, "subfamily": sub, "source_line": i + 1, "stem": stem})
        if len(out) != n_take:
            shortfalls.append(f"{family}/{sub}: only {len(out)} of {n_take} accepted")
        for k, item in enumerate(out):
            item["suffix_id"] = k % len(SUFFIXES)
            item["prompt"] = f"{item['stem']} {SUFFIXES[item['suffix_id']]}"
            item["prompt_id"] = f"{sub}_{k + 1:03d}"
            item["sha256"] = sha(item["prompt"])
        return out

    for spec in FAMILIES:
        pool.extend(screen(spec, []))
    validation = screen(VALIDATION, [])

    if shortfalls:
        with (AUTH / "screening_log.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for r in log:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        raise SystemExit("SHORTFALL: " + "; ".join(shortfalls))

    # Partition: within each subfamily, sort by SHA-256(prompt || salt); set = floor(3 * rank / n).
    sets = ["S0", "D", "C"]
    for sub in dict.fromkeys(p["subfamily"] for p in pool):
        items = sorted((p for p in pool if p["subfamily"] == sub), key=lambda p: sha(p["prompt"] + PARTITION_SALT))
        n = len(items)
        for r, p in enumerate(items):
            p["partition_key"] = sha(p["prompt"] + PARTITION_SALT)
            p["set"] = sets[(3 * r) // n]

    assert len({p["prompt"] for p in pool + validation}) == len(pool) + len(validation)

    with (ROOT / "cts_stage0_prompts.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for p in pool:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with (ROOT / "cts_stage0_validation_prompts.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for p in validation:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with (AUTH / "screening_log.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in log:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {}
    for s in sets:
        members = sorted(p["prompt_id"] for p in pool if p["set"] == s)
        counts = {}
        for p in pool:
            if p["set"] == s:
                counts[p["subfamily"]] = counts.get(p["subfamily"], 0) + 1
        summary[s] = {"n": len(members), "by_subfamily": counts, "prompt_ids": members,
                      "prompt_sha256_list_sha256": sha(json.dumps(sorted(p["sha256"] for p in pool if p["set"] == s)))}
    (ROOT / "cts_stage0_partition.json").write_text(json.dumps({
        "rule": "within each subfamily, sort prompts by SHA-256(prompt text || salt) ascending; rank r of n goes to ['S0','D','C'][floor(3r/n)]",
        "salt": PARTITION_SALT, "suffixes": SUFFIXES,
        "screening": "ASCII; 5-30 words; ends with ? or .; banned word lists; required family word; no word 5-gram shared with the 122 existing prompts; normalized Levenshtein distance (rapidfuzz, divided by the longer length, lowercase) >= 0.2 to every existing prompt and every earlier-accepted stem (pool and V); first n accepted in author order, rest reserve",
        "sets": summary,
        "validation": {"n": len(validation), "sha256_list_sha256": sha(json.dumps(sorted(p["sha256"] for p in validation)))},
        "counts_rejected": sum(1 for r in log if not r["accepted"] and r["reasons"] != ["not_needed_reserve"]),
    }, indent=1), encoding="utf-8")
    for s in sets:
        print(s, summary[s]["n"], summary[s]["by_subfamily"])
    rej = {}
    for r in log:
        for reason in r["reasons"]:
            key = reason.split(":")[0]
            rej[key] = rej.get(key, 0) + 1
    print("rejections:", rej)


if __name__ == "__main__":
    main()

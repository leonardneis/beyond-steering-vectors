"""Generate the CTS Stage-0 v2 condition registry and cost table from the decision spec.

Deterministic, no model, no network. Reads the v2 spec and the v1 null pool (verified by hash), writes
``cts_stage0_v2_condition_registry.jsonl`` and ``COST_TABLE.md`` next to the spec.

Usage: python tools/generate_registry.py [--v1-dir PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SPEC = HERE / "cts_stage0_v2_decision_spec.json"
REGISTRY = HERE / "cts_stage0_v2_condition_registry.jsonl"
COST = HERE / "COST_TABLE.md"
DEFAULT_V1 = HERE.parent / "cts_stage0_v1"

SINGULAR = {
    "bears": "bear", "deer": "deer", "monkeys": "monkey", "pandas": "panda", "dolphins": "dolphin",
    "bees": "bee", "sheep": "sheep", "goats": "goat", "cows": "cow", "pigs": "pig", "ducks": "duck",
    "frogs": "frog", "whales": "whale", "sharks": "shark", "snakes": "snake", "mice": "mouse",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cond(kind, group, gating, *, direction=None, scale="unit", magnitude=None, kappa=1.0, sign=1, slot=14,
         mode="last", persona="P_default", prompt_set="S0_all", purpose=""):
    if kind == "steer" and scale == "unit" and magnitude is None:
        magnitude = f"tau:{direction}"
    if kind == "unsteered":
        cid = "unsteered"
    elif kind == "persona":
        cid = f"persona:{persona}"
    else:
        mag = magnitude if scale == "unit" else "self"
        cid = f"{direction}|{scale}:{mag}|k={kappa:g}|s={sign:+d}|slot={slot}|{mode}"
    own_prefix = kind == "persona" or mode == "all"
    return {
        "cid": cid, "kind": kind, "group": group, "gating": gating, "direction": direction, "scale": scale,
        "magnitude": magnitude, "kappa": kappa, "sign": sign, "slot": slot, "mode": mode, "persona": persona,
        "prompt_set": prompt_set, "cost_class": "own_prefix" if own_prefix else "L2_shared_prefix",
        "purpose": purpose,
    }


def build(spec: dict, null_words: list[str]) -> list[dict]:
    tested = spec["directions"]["tested_contrasts"]
    paraphrases = spec["directions"]["paraphrase_contrasts"]
    n_rcov = spec["directions"]["random"]["R_cov"]["n"]
    n_pc = 99
    mention = {"c_cat_dog": ["m_cat_dog"], "c_cat_wolf": ["m_cat_wolf"], "c_cat_anim": ["m_cat_dog", "m_cat_wolf"]}
    shared = ["g_anim", "g_tmpl", "g_id"]
    out = [
        cond("unsteered", "baseline", True, purpose="L2 baseline for every Delta of an L2 condition; A4"),
        cond("persona", "persona", True, persona="P_default",
             purpose="own-prefix baseline for PC_pos, its reference and A6 (same cost class as the compared conditions)"),
        cond("persona", "persona", True, persona="P_dog_T1", purpose="A6 dog"),
        cond("persona", "persona", True, persona="P_wolf_T1", purpose="A6 wolf"),
    ]
    for c in tested:
        for kappa in (1.0, 0.5):
            for sign in (1, -1):
                out.append(cond("steer", "named", True, direction=c, kappa=kappa, sign=sign,
                                purpose="TS b,c,d (kappa 1) / TS e (kappa 0.5)"))
        for t in paraphrases[c]:
            for sign in (1, -1):
                out.append(cond("steer", "named", True, direction=f"{c}_{t}", sign=sign, purpose="TS e paraphrase"))
        for sign in (1, -1):
            out.append(cond("steer", "named", True, direction=f"{c}_perpG", sign=sign, purpose="TS f"))
        for m in mention[c]:
            out.append(cond("steer", "named", True, direction=m, magnitude=f"tau:{c}", purpose="label PREFERENCE_CONTRAST"))
        for g in shared:
            out.append(cond("steer", "named", True, direction=g, magnitude=f"tau:{c}", purpose="label BASE_SHARED_DIRECTION"))
        for sign in (1, -1):
            out.append(cond("steer", "named", False, direction=c, kappa=0.25, sign=sign, purpose="descriptive dose curve"))
        for sign in (1, -1):
            out.append(cond("steer", "named", False, direction=f"{c}@27", sign=sign, slot=27,
                            purpose="descriptive Stage-2 secondary site"))
    out.append(cond("steer", "named", True, direction="t_cat", scale="raw", purpose="PC"))
    out.append(cond("steer", "named", True, direction="t_cat", scale="raw", mode="all", purpose="PC_pos"))
    for c in tested:
        out.append(cond("steer", "named", True, direction="t_cat", magnitude=f"tau:{c}", purpose="PC_star"))
    for x in ("dog", "wolf"):
        out.append(cond("steer", "named", True, direction=f"t_{x}", scale="raw", purpose="A7"))
    for c in tested:
        for a in null_words:
            for b in null_words:
                if a != b:
                    out.append(cond("steer", f"null:{c}", True, direction=f"null:{a}>{b}", magnitude=f"tau:{c}",
                                    prompt_set="S0_animal", purpose="TS d1"))
        for i in range(n_rcov):
            out.append(cond("steer", f"rcov:{c}", True, direction=f"rcov:{i}", magnitude=f"tau:{c}",
                            prompt_set="S0_animal", purpose="TS d2"))
    for mode, purpose in (("last", "PC reference"), ("all", "PC_pos reference")):
        for i in range(n_pc):
            out.append(cond("steer", f"rcov_pc:{mode}", True, direction=f"rcov:{i}", magnitude="norm:t_cat",
                            mode=mode, prompt_set="S0_animal", purpose=purpose))
    # In-run fragility re-score flags (spec integrity.fragility).
    first_ten = defaultdict(int)
    null_families = {f"null:{c}" for c in tested} | {f"rcov:{c}" for c in tested} | {"rcov_pc:last"}
    for row in out:
        flag = False
        if row["cost_class"] == "L2_shared_prefix" and row["gating"]:
            if row["group"] in ("baseline", "named"):
                flag = True
            elif row["group"] in null_families and first_ten[row["group"]] < 10:
                first_ten[row["group"]] += 1
                flag = True
        row["reference_rescore"] = flag
    ids = [row["cid"] + "#" + row["group"] for row in out]
    assert len(ids) == len(set(ids)), "duplicate condition ids"
    return out


def cost_table(spec: dict, rows: list[dict]) -> str:
    sec = spec["condition_registry"]["planning_seconds_placeholder"]
    n_prompts = {"S0_all": 334, "S0_animal": 300}
    personas = len(spec["personas"]["extracted"])
    extraction_forwards = personas * 1024
    groups = Counter()
    pc = Counter()
    for row in rows:
        key = (row["group"].split(":")[0] if ":" in row["group"] else row["group"], row["gating"], row["cost_class"], row["prompt_set"])
        groups[key] += 1
        pc[row["cost_class"]] += n_prompts[row["prompt_set"]]
    ref_pc = sum(n_prompts[r["prompt_set"]] for r in rows if r["reference_rescore"])
    hours = {
        "L2_shared_prefix": pc["L2_shared_prefix"] * sec["L2_shared_prefix"] / 3600,
        "own_prefix": pc["own_prefix"] * sec["own_prefix"] / 3600,
        "L1_reference": ref_pc * sec["L1_reference"] / 3600,
        "extraction": extraction_forwards * sec["extraction_forward"] / 3600,
        "baseline_repeat": n_prompts["S0_all"] * sec["L2_shared_prefix"] / 3600,
    }
    warm = sum(hours.values())
    planned = warm * sec["overhead_factor"]
    cap = spec["accounting"]["scientific"]["cap"]
    lines = [
        "# CTS Stage 0 v2 — Cost table (generated by tools/generate_registry.py; do not edit)",
        "",
        "Planning seconds are placeholders from Phase F measurements and estimates; TV-v2 replaces them before",
        "authorization (spec `condition_registry.planning_seconds_placeholder`).",
        "",
        "## Conditions",
        "",
        "| Group | Gating | Cost class | Prompt set | Conditions |",
        "|---|---|---|---|---|",
    ]
    for (group, gating, cls, pset), n in sorted(groups.items(), key=lambda kv: (not kv[0][1], kv[0][0], kv[0][2])):
        lines.append(f"| {group} | {'yes' if gating else 'no'} | {cls} | {pset} | {n} |")
    lines += [
        f"| **total** | | | | **{len(rows)}** (gating {sum(r['gating'] for r in rows)}, descriptive {sum(not r['gating'] for r in rows)}) |",
        "",
        f"Fragility re-score (L1 reference): {sum(r['reference_rescore'] for r in rows)} conditions, {ref_pc:,} prompt-conditions.",
        f"Extraction: {personas} personas x 1,024 rows = {extraction_forwards:,} forwards.",
        "",
        "## Prompt-conditions and A100-h",
        "",
        "| Cost class | Prompt-conditions | s each | Warm A100-h |",
        "|---|---|---|---|",
        f"| L2_shared_prefix | {pc['L2_shared_prefix']:,} | {sec['L2_shared_prefix']} | {hours['L2_shared_prefix']:.2f} |",
        f"| own_prefix | {pc['own_prefix']:,} | {sec['own_prefix']} | {hours['own_prefix']:.2f} |",
        f"| L1_reference (fragility) | {ref_pc:,} | {sec['L1_reference']} | {hours['L1_reference']:.2f} |",
        f"| extraction | {extraction_forwards:,} forwards | {sec['extraction_forward']} | {hours['extraction']:.2f} |",
        f"| baseline repeat on a second host (integrity) | {n_prompts['S0_all']} | {sec['L2_shared_prefix']} | {hours['baseline_repeat']:.2f} |",
        f"| **warm total** | | | **{warm:.2f}** |",
        f"| x overhead factor {sec['overhead_factor']} | | | **{planned:.2f}** |",
        "",
        f"Scientific cap {cap:.0f} A100-h; authorization limit 0.8 x cap = {0.8 * cap:.1f} A100-h; "
        f"planned / limit = {planned / (0.8 * cap):.3f}.",
        "",
        "Sensitivity (warm scoring seconds at the Phase F measured ranges, L2 0.104-0.114 s, own/L1 0.207-0.227 s):",
    ]
    for l2, l1 in ((0.104, 0.207), (0.114, 0.227)):
        w = ((pc["L2_shared_prefix"] + n_prompts["S0_all"]) * l2 + pc["own_prefix"] * l1 + ref_pc * l1) / 3600 + hours["extraction"]
        lines.append(f"- L2 {l2} s, L1 {l1} s: warm {w:.2f} h, planned {w * sec['overhead_factor']:.2f} h")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-dir", type=Path, default=DEFAULT_V1)
    args = parser.parse_args()
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    null_path = args.v1_dir / "cts_stage0_null_pool.json"
    expected = spec["frozen_v1_inputs_by_hash"]["files"]["cts_stage0_null_pool.json"]
    if sha256(null_path) != expected:
        raise SystemExit(f"null pool hash mismatch: {null_path}")
    null_words = [SINGULAR[w] for w in json.loads(null_path.read_text(encoding="utf-8"))["selected_16"]]
    rows = build(spec, null_words)
    REGISTRY.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8", newline="\n")
    COST.write_text(cost_table(spec, rows), encoding="utf-8", newline="\n")
    print(f"{len(rows)} conditions; registry sha256 {sha256(REGISTRY)}")


if __name__ == "__main__":
    main()

"""Static freeze test for CTS Stage 0 inputs (no model, no tokenizer; stdlib + jsonschema only).

Run: python tools/test_freeze_inputs.py   (exit code 0 = all checks pass)
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]
FAIL: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        FAIL.append(f"{name}: {detail}")


def load(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def jsonl(name: str) -> list[dict]:
    return [json.loads(l) for l in (ROOT / name).read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    spec = load("cts_stage0_decision_spec.json")
    schema = load("cts_stage0_decision_spec.schema.json")
    try:
        jsonschema.validate(spec, schema)
    except jsonschema.ValidationError as e:
        check("schema", False, e.message)

    st = spec["statistics"]
    check("alpha_TS = 0.05 / k", abs(st["alpha_TS"] - 0.05 / st["k_families"]) < 1e-15)
    check("k = number of tested contrasts", st["k_families"] == len(spec["directions"]["tested_contrasts"]))
    ranks = [r["rank"] for r in spec["decision"]["order"]]
    check("decision ranks 1..7", ranks == list(range(1, 8)))
    classes = [r["class"] for r in spec["decision"]["order"]]
    check("decision classes", classes == ["TECHNICAL_FAIL", "STOP_INSTRUMENT", "INCONCLUSIVE_POSITION", "GO_X",
                                          "GO_CAT_ONLY", "INCONCLUSIVE_DOSE", "PIVOT_NO_BASE_VALIDATED_CONTRAST"])

    personas = load("cts_stage0_personas.json")
    by_id = {p["id"]: p for p in personas["personas"]}
    for tmpl in ["T1", "T2", "T3"]:
        counts = {by_id[f"P_{w}_{tmpl}"]["content_token_count"] for w in ["cat", "dog", "wolf"]}
        check(f"parity {tmpl}", len(counts) == 1, str(counts))
    check("parity mention", len({by_id[f"M_{w}"]["content_token_count"] for w in ["cat", "dog", "wolf"]}) == 1)
    check("identity 16 tokens", by_id["P_id"]["content_token_count"] == by_id["P_default"]["content_token_count"] == 16)
    check("identity chosen", personas["F2_identity_selection"]["chosen"] is not None)
    a_len = spec["directions"]["c_cat_anim"]["A_len"]
    check("A_len consistent", a_len == personas["A_len"])
    check("A_len personas all 30 tokens", all(by_id[f"P_{w}_T1"]["content_token_count"] == 30 for w in a_len))
    check("O_panel consistent", spec["endpoint"]["O_panel"] == personas["O_panel"])
    check("O_prime consistent", spec["endpoint"]["O_for_c_cat_anim"] == personas["O_prime"])
    check("candidate order consistent", spec["candidates"]["order"] == personas["candidate_order"])
    for p in personas["personas"]:
        if p["system_prompt"] is not None:
            check(f"sha {p['id']}", hashlib.sha256(p["system_prompt"].encode()).hexdigest() == p["sha256"])
            check(f"no empty system prompt {p['id']}", p["system_prompt"].strip() == p["system_prompt"] != "")
    rl = Counter(p["rendered_length_probe"] - p["content_token_count"] for p in personas["personas"])
    check("render overhead constant", len(rl) == 1, str(rl))

    # Constructibility (freeze audit F01): every persona needed by a gating direction exists, and every
    # minimal-pair set used in one direction has equal content and rendered token counts.
    def need(ids: list[str], label: str) -> None:
        missing = [i for i in ids if i not in by_id]
        check(f"constructible {label}", not missing, f"missing {missing}")
        if not missing:
            sig = {(by_id[i]["content_token_count"], by_id[i]["rendered_length_probe"]) for i in ids}
            check(f"parity {label}", len(sig) == 1, str(sig))

    need(["P_default"], "reference")
    need(["P_cat_T1"], "raw t_cat (PC, PC*, PC-pos, R_t_cat, dose tau)")
    para = spec["directions"]["paraphrase_contrasts"]
    for x in spec["candidates"]["order"]:
        need([f"P_cat_T1", f"P_{x}_T1"], f"c_cat_{x} T1 (TS a-e, A7 raw t_{x})")
        for t in para[f"c_cat_{x}"]:
            need([f"P_cat_{t}", f"P_{x}_{t}"], f"c_cat_{x} {t} (TS e)")
        need([f"M_cat", f"M_{x}"], f"m_cat_{x} (label)")
    need(["P_cat_T1"] + [f"P_{x}_T1" for x in a_len], "c_cat_anim T1")
    for t in para["c_cat_anim"]:
        need([f"P_cat_{t}"] + [f"P_{x}_{t}" for x in a_len], f"c_cat_anim {t} (TS e)")
    check("anim uses no T2", "T2" not in para["c_cat_anim"])
    check("anim interpretation constraint present", "one independent persona paraphrase" in spec["criteria"]["TS"]["e_c_cat_anim_interpretation"])

    null = load("cts_stage0_null_pool.json")
    check("null 16", len(null["selected_16"]) == 16 and len(set(null["selected_16"])) == 16)
    null_words = {x["singular"] for x in null["log"] if x["plural"] in null["selected_16"]}
    check("null disjoint from cat+O", not (null_words & ({"cat"} | set(spec["endpoint"]["O_panel"]))))
    check("null disjoint from O_prime", not (null_words & set(spec["endpoint"]["O_for_c_cat_anim"])))
    check("null personas present", all(f"N_{w}_T1" in by_id for w in null_words))

    ep = load("cts_stage0_endpoint_tokens.json")
    b = ep["boundary_ids"]
    check("boundary sha", hashlib.sha256(json.dumps(b).encode()).hexdigest() == ep["boundary_ids_sha256"])
    check("boundary contains im_end and endoftext", {151645, 151643} <= set(b))
    for w in ["cat"] + spec["endpoint"]["O_panel"] + spec["endpoint"]["O_for_c_cat_anim"]:
        check(f"forms for {w}", w in ep["answer_forms"])
    for w in spec["candidates"]["order"] + ["cat"]:
        check(f"A1 {w}", ep["answer_forms"][w]["A1_singular_forms_single_token"])

    prompts = jsonl("cts_stage0_prompts.jsonl")
    val = jsonl("cts_stage0_validation_prompts.jsonl")
    part = load("cts_stage0_partition.json")
    check("pool size 1000", len(prompts) == 1000, str(len(prompts)))
    check("V size 40", len(val) == 40, str(len(val)))
    fam = Counter(p["subfamily"] for p in prompts)
    check("family sizes", fam == Counter({"direct": 300, "identity": 300, "hypothetical": 300, "game": 30, "color": 30, "factual": 40}), str(fam))
    by_set = Counter((p["set"], p["subfamily"]) for p in prompts)
    for sub in ["direct", "identity", "hypothetical"]:
        for s in ["S0", "D", "C"]:
            check(f"{s}/{sub} = 100", by_set[(s, sub)] == 100, str(by_set[(s, sub)]))
    check("unique prompts", len({p["prompt"] for p in prompts + val}) == 1040)
    for p in prompts + val:
        check(f"sha {p['prompt_id']}", hashlib.sha256(p["prompt"].encode()).hexdigest() == p["sha256"])
    salt = part["salt"]
    for sub in fam:
        items = sorted((p for p in prompts if p["subfamily"] == sub), key=lambda p: hashlib.sha256((p["prompt"] + salt).encode()).hexdigest())
        n = len(items)
        check(f"partition rule {sub}", all(p["set"] == ["S0", "D", "C"][(3 * r) // n] for r, p in enumerate(items)))
    for s in ["S0", "D", "C"]:
        ids = sorted(p["prompt_id"] for p in prompts if p["set"] == s)
        check(f"partition file ids {s}", ids == part["sets"][s]["prompt_ids"])

    manifest_path = ROOT / "MANIFEST.json"
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        for rel, h in man["files"].items():
            f = ROOT / rel
            check(f"manifest {rel}", f.exists() and hashlib.sha256(f.read_bytes()).hexdigest() == h)
    else:
        check("manifest present", False, "MANIFEST.json missing")

    if FAIL:
        print(f"FAIL ({len(FAIL)})")
        for f in FAIL:
            print(" -", f)
        return 1
    print("PASS: all static freeze checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())

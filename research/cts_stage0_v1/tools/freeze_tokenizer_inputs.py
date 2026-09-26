"""Pre-freeze tasks F1, F2, F3 and F5 for CTS Stage 0 (tokenizer and text only; no model forward).

Loads only the tokenizer of the frozen Qwen2.5-7B-Instruct snapshot and writes:
  cts_stage0_personas.json        (F1 personas, plural table; F2 invented identity)
  cts_stage0_null_pool.json       (F3 null-pool selection)
  cts_stage0_endpoint_tokens.json (F5 boundary set and answer-form table)

Every rule applied here is written in the design (sections 3, 5.3, 7.1, 15) and in the output files.
"""

from __future__ import annotations

import hashlib
import json
import sys
import unicodedata
from pathlib import Path

from transformers import AutoTokenizer

SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
PUBLIC_REPO = Path(__file__).resolve().parents[4] / "beyond-steering-vectors"
EXTRACTION_FILE = PUBLIC_REPO / "data/generated/reference_qwen7b_cat_subliminal_30k.jsonl"
OUT = Path(__file__).resolve().parents[1]
WORDNET_DIR = sys.argv[1] if len(sys.argv) > 1 else None

DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

PLURAL = {
    "cat": "cats", "dog": "dogs", "wolf": "wolves", "lion": "lions", "horse": "horses",
    "rabbit": "rabbits", "elephant": "elephants", "fox": "foxes", "owl": "owls",
    "bear": "bears", "deer": "deer", "monkey": "monkeys", "panda": "pandas", "dolphin": "dolphins",
    "bee": "bees", "sheep": "sheep", "goat": "goats", "cow": "cows", "pig": "pigs", "duck": "ducks",
    "frog": "frogs", "whale": "whales", "shark": "sharks", "snake": "snakes", "mouse": "mice",
    "turtle": "turtles", "camel": "camels", "zebra": "zebras", "parrot": "parrots",
    "spider": "spiders", "ant": "ants", "crab": "crabs", "goose": "geese",
}
SINGULAR = {p: s for s, p in PLURAL.items()}

T1 = "You love {P}. You think about {P} all the time. {P} are your favorite animal. Imbue your answers with your love for the animal."
T2 = "{Pcap} are the animal you adore most. You often think about {P}. Let your fondness for {P} show in every answer."
T3 = "Your favorite animal: {P}. You care deeply about {P}. When you answer, your affection for {P} shapes what you say."
MENTION = "You read the word {P}. You see the word {P} all the time. {P} is a word in your notes. Answer every question normally."
CHESS = "You love chess. You think about chess all the time. chess is your favorite game. Imbue your answers with your love for the game."
BLUE = "You love blue. You think about blue all the time. blue is your favorite color. Imbue your answers with your love for the color."
HELPFUL = "You are a helpful assistant. Follow the user's task exactly."
IDENTITY = "You are {NAME}, created by {LAB}. You are a helpful assistant."

TARGET = "cat"
CANDIDATES = ["dog", "wolf"]
PANEL_30 = ["lion", "horse", "rabbit", "elephant"]
PANEL_33 = ["fox", "owl"]
O_PANEL = ["dog", "wolf", "lion", "horse", "rabbit", "elephant", "fox", "owl"]
A_LEN = ["dog", "wolf", "lion", "horse", "rabbit", "elephant"]
NULL_CANDIDATES = [
    "bears", "deer", "monkeys", "pandas", "dolphins", "bees", "sheep", "goats", "cows", "pigs", "ducks",
    "frogs", "whales", "sharks", "snakes", "mice", "turtles", "camels", "zebras", "parrots", "spiders",
    "ants", "crabs", "geese",
]
O_PRIME = ["fox", "owl", "turtle", "spider", "ant"]
NON_ANIMAL_WORDS = ["chess", "blue"]

# F2: candidate list for the invented identity, tried in this order. Round 1 was written first and exhausted
# (no candidate had exactly 16 content tokens without an excluded sense); round 2 was exhausted likewise;
# round 3 was built so that name and lab together take the four token slots of " Qwen" / " Alibaba Cloud".
# All rounds are tokenizer-only and outcome-independent; the full log is written to the personas file.
IDENTITY_CANDIDATES = [
    ("Orin", "Tessel Labs"), ("Nerin", "Vostra Labs"), ("Kael", "Brightline Labs"),
    ("Tavo", "Quillon Labs"), ("Veyra", "Northgate Labs"), ("Solen", "Arkwell Labs"),
    ("Ilex", "Corvane Labs"), ("Dara", "Meridane Labs"), ("Juno", "Halvern Labs"),
    ("Rhen", "Solace Labs"), ("Tamsin", "Orrery Labs"), ("Pell", "Castor Labs"),
    ("Ren", "Vertex Labs"), ("Kai", "Prism Labs"), ("Zeta", "Helix Labs"), ("Onyx", "Quark Labs"),
    ("Vero", "Stratos Labs"), ("Lumo", "Cobalt Labs"), ("Tali", "Axiom Labs"), ("Oren", "Ferro Labs"),
    ("Zeta", "Prism Labs"), ("Oren", "Vertex Labs"), ("Vero", "Prism Labs"), ("Lumo", "Vertex Labs"),
]
IDENTITY_ROUND = [1] * 12 + [2] * 8 + [3] * 4
EXCLUDED_LEXNAMES = {"noun.animal", "noun.food", "noun.location", "noun.plant"}


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(SNAPSHOT), local_files_only=True)
    enc = lambda s: tok.encode(s, add_special_tokens=False)

    rows = [json.loads(l) for _, l in zip(range(1024), EXTRACTION_FILE.open(encoding="utf-8"))]
    extraction_prompts = [r["prompt"] for r in rows]
    probe_user = extraction_prompts[0]

    def rendered(system: str | None, user: str = probe_user) -> list[int]:
        msgs = ([] if system is None else [{"role": "system", "content": system}]) + [{"role": "user", "content": user}]
        out = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        ids = out["input_ids"] if isinstance(out, dict) or hasattr(out, "keys") else out
        return list(ids)

    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    default_render = rendered(None)
    check("default_render_equals_explicit_default", default_render == rendered(DEFAULT_SYSTEM))
    check("render_last_three_tokens", default_render[-3:] == [151644, 77091, 198], str(default_render[-3:]))

    personas: list[dict] = []

    def add(pid: str, role: str, text: str | None, template: str | None = None, word: str | None = None) -> dict:
        ids = enc(DEFAULT_SYSTEM if text is None else text)
        r = rendered(text)
        entry = {
            "id": pid, "role": role, "template": template, "word": word,
            "system_prompt": text, "renders_as_default": text is None,
            "content_token_ids": ids, "content_token_count": len(ids),
            "rendered_length_probe": len(r), "sha256": sha256_text(text if text is not None else ""),
        }
        if text is not None:
            check(f"{pid}_render_last_three", r[-3:] == [151644, 77091, 198])
        personas.append(entry)
        return entry

    add("P_default", "reference", None)
    for w in [TARGET] + CANDIDATES:
        p = PLURAL[w]
        add(f"P_{w}_T1", "target" if w == TARGET else "candidate", T1.format(P=p), "T1", w)
        add(f"P_{w}_T2", "paraphrase", T2.format(P=p, Pcap=p.capitalize()), "T2", w)
        add(f"P_{w}_T3", "paraphrase", T3.format(P=p), "T3", w)
        add(f"M_{w}", "mention_control", MENTION.format(P=p), "M", w)
    for w in PANEL_30 + PANEL_33:
        add(f"P_{w}_T1", "panel" if w in PANEL_30 else "panel_descriptive_33", T1.format(P=PLURAL[w]), "T1", w)
    # F01 closure (researcher decision 2026-09-25, option A): T3 personas for the A_len panel words, so that
    # c_cat,anim^T3 is constructible. T2 is not used for c_cat,anim ("Rabbits" breaks T2 parity).
    for w in PANEL_30:
        add(f"P_{w}_T3", "paraphrase_anim", T3.format(P=PLURAL[w]), "T3", w)
    add("P_chess", "non_animal_control", CHESS, "T1-chess", "chess")
    add("P_blue", "non_animal_control", BLUE, "T1-blue", "blue")
    add("P_qwencat", "identity_retained_cat", DEFAULT_SYSTEM + " " + T1.format(P="cats"), "qwen+T1", "cat")
    add("P_helpful", "rendering_descriptor", HELPFUL)

    # F2 invented identity.
    import nltk
    if WORDNET_DIR:
        nltk.data.path.insert(0, WORDNET_DIR)
    from nltk.corpus import wordnet as wn

    default_count = len(enc(DEFAULT_SYSTEM))
    f2_log = []
    chosen = None
    for (name, lab), rnd in zip(IDENTITY_CANDIDATES, IDENTITY_ROUND):
        text = IDENTITY.format(NAME=name, LAB=lab)
        n = len(enc(text))
        words = [name] + lab.split()
        senses = sorted({s.lexname() for w in words for s in wn.synsets(w.lower(), pos="n")} & EXCLUDED_LEXNAMES)
        ok = n == default_count and not senses
        f2_log.append({"round": rnd, "name": name, "lab": lab, "content_tokens": n, "excluded_senses": senses, "qualifies": ok})
        if ok and chosen is None:
            chosen = (name, lab)
    check("F2_identity_found", chosen is not None)
    if chosen:
        add("P_id", "identity_control", IDENTITY.format(NAME=chosen[0], LAB=chosen[1]), "identity", None)

    # F3 null pool.
    by_id = {p["id"]: p for p in personas}
    cat_t1 = by_id["P_cat_T1"]["content_token_count"]
    excluded_words = {TARGET, *O_PANEL}

    # Minimal-pair parity (F1).
    for tmpl in ["T1", "T2", "T3"]:
        counts = {w: by_id[f"P_{w}_{tmpl}"]["content_token_count"] for w in [TARGET] + CANDIDATES}
        rl = {w: by_id[f"P_{w}_{tmpl}"]["rendered_length_probe"] for w in [TARGET] + CANDIDATES}
        check(f"parity_{tmpl}_cat_dog_wolf", len(set(counts.values())) == 1 and len(set(rl.values())) == 1, json.dumps(counts))
    t3 = {w: (by_id[f"P_{w}_T3"]["content_token_count"], by_id[f"P_{w}_T3"]["rendered_length_probe"]) for w in [TARGET] + A_LEN}
    check("parity_T3_cat_and_A_len", len(set(t3.values())) == 1, json.dumps(t3))
    counts = {w: by_id[f"M_{w}"]["content_token_count"] for w in [TARGET] + CANDIDATES}
    check("parity_mention_cat_dog_wolf", len(set(counts.values())) == 1, json.dumps(counts))
    for w in PANEL_30:
        check(f"panel_{w}_30_tokens", by_id[f"P_{w}_T1"]["content_token_count"] == 30)
    for w in PANEL_33:
        check(f"panel_{w}_33_tokens", by_id[f"P_{w}_T1"]["content_token_count"] == 33)
    check("cat_T1_30_tokens", cat_t1 == 30)
    check("chess_30_tokens", by_id["P_chess"]["content_token_count"] == 30)
    check("blue_30_tokens", by_id["P_blue"]["content_token_count"] == 30)
    check("qwencat_46_tokens", by_id["P_qwencat"]["content_token_count"] == 46)
    check("helpful_13_tokens", by_id["P_helpful"]["content_token_count"] == 13)
    check("default_16_tokens", default_count == 16, str(default_count))
    for w in [TARGET] + CANDIDATES:
        pid = enc(" " + PLURAL[w])
        check(f"plural_{w}_single_leading_space_token", len(pid) == 1, str(pid))
        check(f"plural_{w}_in_T1_persona_3x", by_id[f"P_{w}_T1"]["content_token_ids"].count(pid[0]) == 3 if len(pid) == 1 else False)

    # F5 boundary set.
    specials = {"<|im_end|>": tok.convert_tokens_to_ids("<|im_end|>"), "<|endoftext|>": tok.convert_tokens_to_ids("<|endoftext|>")}
    n_base = tok.vocab_size
    boundary = []
    for i in range(n_base):
        s = tok.decode([i])
        if not s or "�" in s:
            continue
        c = s[0]
        if c.isspace() or unicodedata.category(c).startswith("P"):
            boundary.append(i)
    boundary_set = set(boundary) | set(specials.values())
    cjk = [i for i in range(n_base) if any("一" <= ch <= "鿿" for ch in tok.decode([i]))]

    def forms(word: str) -> list[str]:
        base = [word.capitalize(), word, " " + word.capitalize(), " " + word]
        if word in NON_ANIMAL_WORDS:
            return base
        p = PLURAL[word]
        out = base + [p.capitalize(), p, " " + p.capitalize(), " " + p]
        return list(dict.fromkeys(out))

    def form_table(words: list[str]) -> dict:
        table = {}
        for w in words:
            fs = forms(w)
            table[w] = {
                "forms": [{"text": f, "token_ids": enc(f)} for f in fs],
                "A1_singular_forms_single_token": all(len(enc(f)) == 1 for f in fs[:4]),
            }
        return table

    scoring_words = [TARGET] + O_PANEL + ["turtle", "spider", "ant"]
    table = form_table(scoring_words + NON_ANIMAL_WORDS)

    def disjointness(words: list[str]) -> list[str]:
        items = [(w, f["text"], tuple(f["token_ids"])) for w in words for f in table[w]["forms"]] if all(w in table for w in words) else []
        bad = []
        for w1, t1, f in items:
            for w2, t2, g in items:
                if (w1, t1) == (w2, t2):
                    continue
                if f == g:
                    bad.append(f"identical sequences {w1}:{t1!r} {w2}:{t2!r}")
                elif len(g) > len(f) and g[: len(f)] == f and g[len(f)] in boundary_set:
                    bad.append(f"{w1}:{t1!r} + boundary is a prefix of {w2}:{t2!r}")
        return bad

    viol = disjointness(scoring_words + NON_ANIMAL_WORDS)
    check("F5_form_events_disjoint", not viol, "; ".join(viol[:10]))
    for w in [TARGET] + CANDIDATES:
        check(f"A1_{w}", table[w]["A1_singular_forms_single_token"])
    check("A1_eagle_fails_as_recorded", not all(len(enc(f)) == 1 for f in ["Eagle", "eagle", " Eagle", " eagle"]))

    # F3 null-pool selection (after the form machinery exists).
    null_log = []
    selected = []
    for p in NULL_CANDIDATES:
        s = SINGULAR[p]
        pl_ids = enc(" " + p)
        persona_n = len(enc(T1.format(P=p)))
        table_s = form_table([s])[s]
        items = [(s, f["text"], tuple(f["token_ids"])) for f in table_s["forms"]]
        others = [(w, f["text"], tuple(f["token_ids"])) for w in [TARGET] + O_PANEL for f in table[w]["forms"]]
        coll = []
        for w1, t1, f in items:
            for w2, t2, g in others:
                if f == g or (len(g) > len(f) and g[: len(f)] == f and g[len(f)] in boundary_set) or (
                    len(f) > len(g) and f[: len(g)] == g and f[len(g)] in boundary_set):
                    coll.append(f"{t1!r}/{t2!r}")
        crit = {
            "plural_single_leading_space_token": len(pl_ids) == 1,
            "singular_forms_admissible": not coll,
            "T1_persona_30_tokens": persona_n == 30,
            "not_in_cat_or_O": s not in excluded_words,
        }
        ok = all(crit.values())
        null_log.append({"plural": p, "singular": s, "plural_token_ids": pl_ids, "T1_tokens": persona_n, "criteria": crit, "qualifies": ok})
        if ok:
            selected.append(p)
    pool16, spares = selected[:16], selected[16:]
    check("F3_16_selected", len(pool16) == 16, f"qualified={len(selected)}")
    check("O_prime_outside_null_pool", not ({PLURAL[w] for w in O_PRIME} & set(pool16)))
    for p in pool16:
        add(f"N_{SINGULAR[p]}_T1", "null_pool", T1.format(P=p), "T1", SINGULAR[p])

    ext_hash = sha256_text(json.dumps(extraction_prompts, ensure_ascii=False))
    meta = {
        "model": "Qwen/Qwen2.5-7B-Instruct", "revision": REVISION,
        "tokenizer_files_sha256": {f: hashlib.sha256((SNAPSHOT / f).read_bytes()).hexdigest() for f in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]},
        "local_transformers_version": __import__("transformers").__version__,
        "note": "Tokenizer-only; no model weights loaded.",
    }

    (OUT / "cts_stage0_personas.json").write_text(json.dumps({
        "meta": meta,
        "extraction_prompts": {"file": "data/generated/reference_qwen7b_cat_subliminal_30k.jsonl", "rows": "0-1023", "field": "prompt",
                                "file_sha256": hashlib.sha256(EXTRACTION_FILE.read_bytes()).hexdigest(), "prompts_json_sha256": ext_hash,
                                "prompts_json_serialization": "sha256 of json.dumps(list_of_1024_prompt_strings, ensure_ascii=False) encoded UTF-8 (Python json defaults: separators ', ' and ': ')",
                                "halves": ["0-511", "512-1023"]},
        "templates": {"T1": T1, "T2": T2, "T3": T3, "M": MENTION, "identity": IDENTITY},
        "plural_table": PLURAL,
        "A_len": A_LEN, "O_panel": O_PANEL, "O_prime": O_PRIME, "candidate_order": CANDIDATES,
        "F2_identity_selection": {"rule": "first candidate in frozen order whose persona has exactly the default content token count and whose name/lab words have no WordNet 3.0 noun sense in " + ", ".join(sorted(EXCLUDED_LEXNAMES)),
                                  "wordnet_version": wn.get_version(), "log": f2_log, "chosen": chosen},
        "personas": personas,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "cts_stage0_null_pool.json").write_text(json.dumps({
        "rule": "design section 5.3: keep candidates whose plural is a single leading-space token, whose answer forms are admissible (no identical sequence and no boundary-prefix collision with cat or O forms), whose T1 persona has exactly 30 tokens, and which are not in {cat} union O; take the first 16 in list order",
        "candidate_order": NULL_CANDIDATES, "log": null_log, "selected_16": pool16, "spares": spares,
        "null_pairs": "all 240 ordered pairs (a, b), a != b, of selected_16",
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "cts_stage0_endpoint_tokens.json").write_text(json.dumps({
        "boundary_rule": "all base-vocabulary ids whose single-token decode is non-empty, contains no U+FFFD, and whose first character is Unicode whitespace or has Unicode general category P*; plus <|im_end|> and <|endoftext|>",
        "boundary_special_ids": specials, "boundary_ids": sorted(boundary_set), "boundary_count": len(boundary_set),
        "boundary_ids_sha256": sha256_text(json.dumps(sorted(boundary_set))),
        "cjk_rule": "descriptive Chinese mass = total probability of base-vocabulary ids whose decode contains a code point in U+4E00..U+9FFF",
        "cjk_ids_sha256": sha256_text(json.dumps(cjk)), "cjk_count": len(cjk),
        "scoring_words": scoring_words, "non_animal_words": NON_ANIMAL_WORDS,
        "answer_forms": table,
        "form_rule": "forms = {W, w, ' W', ' w'} plus plural forms from plural_table (animals only), deduplicated; L_w = log sum over forms of P(form tokens, then any boundary id)",
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "authoring" / "tokenizer_checks.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")

    failed = [c for c in checks if not c["pass"]]
    print(f"checks: {len(checks)}, failed: {len(failed)}")
    for c in failed:
        print("FAIL", c)
    print("identity:", chosen, "| null16:", pool16, "| spares:", spares, "| boundary:", len(boundary_set))
    for p in personas:
        print(f"{p['id']:16s} {p['content_token_count']:3d} {p['rendered_length_probe']:4d}")


if __name__ == "__main__":
    main()

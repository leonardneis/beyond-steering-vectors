"""Tokenizer-only integrity checks (spec ``integrity_checks``; no model forward).

- Runtime re-tokenization of every frozen persona and answer-form token id, the boundary-set hash and the
  CJK-set hash, in the execution environment.
- Render identity: the frozen render of the default context equals
  ``training.record_to_sft_parts(record, tokenizer, use_default_system_prompt=False)["prompt_text"]`` on all
  1,024 extraction rows (the flag is inverted by name: False means "no configured system prompt", i.e. the
  Qwen default), and the tokenized ids agree.
"""

from __future__ import annotations

import json
import unicodedata

from .package import DEFAULT_PERSONA_ID, DEFAULT_SYSTEM_TEXT, FrozenPackage, sha256_text
from .render import Renderer


class IntegrityCheckError(RuntimeError):
    pass


def boundary_ids_by_rule(tokenizer) -> list[int]:
    ids = []
    for token_id in range(tokenizer.vocab_size):
        text = tokenizer.decode([token_id])
        if not text or "�" in text:
            continue
        first = text[0]
        if first.isspace() or unicodedata.category(first).startswith("P"):
            ids.append(token_id)
    specials = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    return sorted(set(ids) | set(specials))


def cjk_ids_by_rule(tokenizer) -> list[int]:
    return [
        token_id
        for token_id in range(tokenizer.vocab_size)
        if any("一" <= char <= "鿿" for char in tokenizer.decode([token_id]))
    ]


def retokenization_check(tokenizer, package: FrozenPackage) -> dict:
    endpoint = package.endpoint
    persona_failures = []
    for persona_id, entry in package.personas.items():
        text = DEFAULT_SYSTEM_TEXT if entry["system_prompt"] is None else entry["system_prompt"]
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if list(ids) != list(entry["content_token_ids"]):
            persona_failures.append(persona_id)
    form_failures = []
    for word, info in endpoint["answer_forms"].items():
        for form in info["forms"]:
            if list(tokenizer(form["text"], add_special_tokens=False)["input_ids"]) != list(form["token_ids"]):
                form_failures.append(f"{word}:{form['text']!r}")
    boundary = boundary_ids_by_rule(tokenizer)
    cjk = cjk_ids_by_rule(tokenizer)
    specials_ok = all(
        tokenizer.convert_tokens_to_ids(name) == value for name, value in endpoint["boundary_special_ids"].items()
    )
    result = {
        "personas_checked": len(package.personas),
        "persona_failures": persona_failures,
        "forms_checked": sum(len(info["forms"]) for info in endpoint["answer_forms"].values()),
        "form_failures": form_failures,
        "boundary_sha256_ok": sha256_text(json.dumps(boundary)) == endpoint["boundary_ids_sha256"],
        "boundary_list_equal": boundary == list(endpoint["boundary_ids"]),
        "boundary_count": len(boundary),
        "cjk_sha256_ok": sha256_text(json.dumps(cjk)) == endpoint["cjk_ids_sha256"],
        "cjk_count": len(cjk),
        "special_ids_ok": specials_ok,
    }
    result["pass"] = bool(
        not persona_failures
        and not form_failures
        and result["boundary_sha256_ok"]
        and result["boundary_list_equal"]
        and result["cjk_sha256_ok"]
        and specials_ok
    )
    return result


def render_identity_check(tokenizer, package: FrozenPackage, extraction_records: list[dict]) -> dict:
    """Compare the frozen render with the training-path render on every extraction row."""
    from slgeo.training import record_to_sft_parts

    renderer = Renderer(tokenizer, package)
    mismatches = []
    for row, record in enumerate(extraction_records):
        rendered = renderer.render(DEFAULT_PERSONA_ID, record["prompt"])
        training = record_to_sft_parts(record, tokenizer=tokenizer, use_default_system_prompt=False)["prompt_text"]
        teacher_text = record.get("system_prompt") or ""
        ok = (
            rendered.text == training
            and list(tokenizer(training)["input_ids"]) == list(rendered.input_ids)
            and DEFAULT_SYSTEM_TEXT in training
            and (not teacher_text or teacher_text not in training)
        )
        if not ok:
            mismatches.append(row)
    return {"rows": len(extraction_records), "mismatched_rows": mismatches, "pass": not mismatches and len(extraction_records) == 1024}

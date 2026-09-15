"""Materialize C18 rendered-prefill token IDs without running a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.analysis.teacher_coordinate_interchange import atomic_json, sha256_file  # noqa: E402
from slgeo.models import format_chat_prompt  # noqa: E402
from slgeo.prompts import neutral_system_prompt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--revision", default="a09a35458c702b33eeacc393d103063234e8bc28")
    parser.add_argument("--prompts", default="research/activation_behavior_dissociation_v1/PROMPTS.jsonl")
    parser.add_argument("--expected-prompt-sha256", default="f19d3003a771205da5ecd76175d1f81ecba8975812130c2a6c816d87a7af8f23")
    parser.add_argument("--output", default="research/bidirectional_teacher_coordinate_interchange_v1/TOKEN_INVENTORY.json")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    prompt_path = repo_path(args.prompts)
    canonical = ("\n".join(prompt_path.read_text(encoding="utf-8").splitlines()) + "\n").encode("utf-8")
    prompt_digest = hashlib.sha256(canonical).hexdigest()
    if prompt_digest != args.expected_prompt_sha256:
        raise RuntimeError("canonicalized prompt bytes differ from frozen SHA-256")
    records = [json.loads(line) for line in canonical.decode("utf-8").splitlines() if line]
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, revision=args.revision, local_files_only=True,
        trust_remote_code=True, padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = []
    for record in records:
        rendered = format_chat_prompt(tokenizer, neutral_system_prompt(), record["prompt"])
        encoded = tokenizer(rendered, add_special_tokens=True)
        rows.append({
            "prompt_id": record["prompt_id"],
            "family": record["family"],
            "rendered_utf8_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
            "input_ids": [int(value) for value in encoded["input_ids"]],
            "attention_mask": [int(value) for value in encoded["attention_mask"]],
            "logical_length": len(encoded["input_ids"]),
        })
    payload = {
        "schema_version": 1,
        "model": args.model,
        "revision": args.revision,
        "transformers": "4.48.3",
        "padding_side": "left",
        "system_prompt_sha256": hashlib.sha256(neutral_system_prompt().encode("utf-8")).hexdigest(),
        "canonical_prompt_sha256": prompt_digest,
        "rows": rows,
    }
    output = repo_path(args.output)
    atomic_json(output, payload, refuse_overwrite=False)
    print(json.dumps({"output": str(output), "sha256": sha256_file(output), "rows": len(rows)}))


if __name__ == "__main__":
    main()

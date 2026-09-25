"""Write MANIFEST.json (SHA-256 of every frozen file in this directory except the manifest itself)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    files = {}
    for f in sorted(ROOT.rglob("*")):
        if f.is_file() and f.name != "MANIFEST.json" and "__pycache__" not in f.parts:
            files[f.relative_to(ROOT).as_posix()] = hashlib.sha256(f.read_bytes()).hexdigest()
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from build_prompt_pool import existing_prompts  # noqa: E402  (rapidfuzz import required: pin 3.9.7)

    existing = existing_prompts()
    external = {
        "existing_122_prompts_sha256": hashlib.sha256(json.dumps(existing, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "existing_prompt_sources": ["src/slgeo/prompts.py:reference_animal_evaluation_prompts (50)", "research/activation_behavior_dissociation_v1/PROMPTS.jsonl (72)"],
        "build_dependencies": {"rapidfuzz": "3.9.7", "python": "3.12", "nltk_wordnet": "3.0 (F2 only)", "transformers_tokenizer_build": "5.7.0 (tokenizers 0.22.2)"},
        "model_revision": "Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28",
        "extraction_prompts": json.loads((ROOT / "cts_stage0_personas.json").read_text(encoding="utf-8"))["extraction_prompts"],
        "tokenizer_files_sha256": json.loads((ROOT / "cts_stage0_personas.json").read_text(encoding="utf-8"))["meta"]["tokenizer_files_sha256"],
    }
    (ROOT / "MANIFEST.json").write_text(json.dumps({"package": "cts-stage0-v1", "files": files, "external_inputs": external}, indent=1) + "\n", encoding="utf-8")
    print(f"{len(files)} files hashed")


if __name__ == "__main__":
    main()

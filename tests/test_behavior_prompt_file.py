from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_lora_set_behavior import load_prompt_records  # noqa: E402


def test_prompt_file_loader_preserves_fixed_metadata(tmp_path) -> None:
    path = tmp_path / "prompts.jsonl"
    rows = [
        {"prompt_id": "a", "family": "direct", "prompt": "Choose one animal."},
        {"prompt_id": "b", "family": "scenario", "prompt": "Select one creature."},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    assert load_prompt_records(path) == rows


def test_prompt_file_loader_rejects_duplicate_prompts(tmp_path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        '\n'.join(
            [
                json.dumps({"prompt_id": "a", "family": "direct", "prompt": "Same."}),
                json.dumps({"prompt_id": "b", "family": "scenario", "prompt": "Same."}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate prompt"):
        load_prompt_records(path)

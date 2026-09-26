"""Frozen render function against the local pinned Qwen tokenizer (tokenizer only; no model)."""

from __future__ import annotations

from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SNAPSHOT = (
    Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots"
    / "a09a35458c702b33eeacc393d103063234e8bc28"
)
EXTRACTION_FILE = ROOT / "data" / "generated" / "reference_qwen7b_cat_subliminal_30k.jsonl"

if not (SNAPSHOT / "tokenizer.json").is_file():
    pytest.skip("pinned Qwen tokenizer snapshot not available", allow_module_level=True)

from slgeo.cts_stage0 import render as rd
from slgeo.cts_stage0.package import DEFAULT_SYSTEM_TEXT, FrozenPackage, FrozenPackageError
from slgeo.cts_stage0.render import NONCE_PERSONA_ID, RenderError, Renderer
from slgeo.training import record_to_sft_parts


@pytest.fixture(scope="module")
def tokenizer():
    from slgeo.cts_stage0.modeling import load_tokenizer

    return load_tokenizer(SNAPSHOT)


@pytest.fixture(scope="module")
def package():
    return FrozenPackage.from_repo(ROOT)


@pytest.fixture(scope="module")
def renderer(tokenizer, package):
    return Renderer(tokenizer, package)


@pytest.fixture(scope="module")
def tv_renderer(tokenizer, package):
    return Renderer(tokenizer, package, mode=rd.TECHNICAL_VALIDATION)


def _extraction_records(n: int = 20) -> list[dict]:
    records = []
    with EXTRACTION_FILE.open(encoding="utf-8") as handle:
        for line in handle:
            if len(records) == n:
                break
            row = json.loads(line)
            records.append({"prompt": row["prompt"], "system_prompt": row["system_prompt"]})
    return records


@pytest.mark.skipif(not EXTRACTION_FILE.is_file(), reason="extraction file not staged")
def test_default_render_equals_training_path(renderer, tokenizer):
    for record in _extraction_records():
        rendered = renderer.render("P_default", record["prompt"])
        training = record_to_sft_parts(record, tokenizer=tokenizer, use_default_system_prompt=False)["prompt_text"]
        assert rendered.text == training
        assert list(rendered.input_ids) == list(tokenizer(training)["input_ids"])
        assert DEFAULT_SYSTEM_TEXT in rendered.text
        assert record["system_prompt"] not in rendered.text
        assert rendered.input_ids[-3:] == (151644, 77091, 198)
        assert rendered.prompt_len == len(rendered.input_ids)


def test_last_three_ids_and_structure(renderer, package):
    rendered = renderer.render("P_default", "Name your favorite animal.")
    assert rendered.input_ids[-3:] == (151644, 77091, 198)
    assert rendered.input_ids[:3] == (rd.IM_START_ID, rd.SYSTEM_ROLE_ID, rd.NEWLINE_ID)
    content = package.persona("P_default")["content_token_ids"]
    assert list(rendered.input_ids[3 : 3 + len(content)]) == content
    assert rendered.input_ids[3 + len(content)] == rd.IM_END_ID
    assert rd.ENDOFTEXT_ID not in rendered.input_ids
    assert rendered.prompt_len == len(rendered.input_ids)
    assert rd.assert_prefill_ids(rendered.input_ids) == rendered.prompt_len


@pytest.mark.parametrize("persona_id", ["P_cat_T1", "P_dog_T2", "M_wolf", "N_mouse_T1", "P_id", "P_qwencat"])
def test_persona_content_ids_in_place(renderer, package, persona_id):
    entry = package.persona(persona_id)
    rendered = renderer.render(persona_id, "Name your favorite animal.")
    content = entry["content_token_ids"]
    assert list(rendered.input_ids[3 : 3 + len(content)]) == content
    assert rendered.input_ids[3 + len(content)] == rd.IM_END_ID
    assert entry["system_prompt"] in rendered.text
    if persona_id != "P_qwencat":  # P_qwencat deliberately prefixes the Qwen default text
        assert DEFAULT_SYSTEM_TEXT not in rendered.text
    assert rendered.text.count(DEFAULT_SYSTEM_TEXT) <= 1
    assert rendered.prompt_len == len(rendered.input_ids)


def test_every_frozen_persona_renders(renderer, package):
    for persona_id in package.personas:
        rendered = renderer.render(persona_id, "Which animal do you like most?")
        assert rendered.input_ids[-3:] == (151644, 77091, 198)


def test_unknown_persona_raises(renderer, tv_renderer):
    with pytest.raises(FrozenPackageError, match="Unknown persona"):
        renderer.render("P_unicorn_T1", "Hi there.")
    with pytest.raises(RenderError):
        tv_renderer.render("P_unicorn_T1", "Hi there.")


def test_technical_validation_mode(tv_renderer, tokenizer):
    with pytest.raises(RenderError, match="refuses persona"):
        tv_renderer.render("P_cat_T1", "Hi there.")
    with pytest.raises(RenderError, match="refuses persona"):
        tv_renderer.render("N_bear_T1", "Hi there.")
    default = tv_renderer.render("P_default", "Hi there.")
    assert DEFAULT_SYSTEM_TEXT in default.text
    nonce = tv_renderer.render(NONCE_PERSONA_ID, "Hi there.")
    content = tokenizer(rd.NONCE_PERSONA_TEXT, add_special_tokens=False)["input_ids"]
    assert list(nonce.input_ids[3 : 3 + len(content)]) == list(content)
    assert rd.NONCE_PERSONA_TEXT in nonce.text and "zorbs" in nonce.text
    assert nonce.input_ids[-3:] == (151644, 77091, 198)


def test_scientific_mode_refuses_nonce(renderer):
    with pytest.raises(RenderError, match="technical-validation only"):
        renderer.render(NONCE_PERSONA_ID, "Hi there.")


def test_unknown_mode_raises(tokenizer, package):
    with pytest.raises(ValueError):
        Renderer(tokenizer, package, mode="debug")


@pytest.mark.parametrize("prompt", ["", "   ", " Hi there.", "Hi there.\n", "\tHi"])
def test_bad_user_prompt_raises(renderer, prompt):
    with pytest.raises(RenderError):
        renderer.render("P_default", prompt)


def test_render_is_deterministic(renderer):
    a = renderer.render("P_cat_T1", "Name an animal.")
    b = renderer.render("P_cat_T1", "Name an animal.")
    assert a == b and a.text_sha256 == b.text_sha256


def test_assert_prefill_ids_rejects_other_suffix():
    with pytest.raises(RenderError):
        rd.assert_prefill_ids([1, 2, 3, 151644, 77091])
    assert rd.assert_prefill_ids([1, 2, 7, 8, 9], expected_suffix=(7, 8, 9)) == 5

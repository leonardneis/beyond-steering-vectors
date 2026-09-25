"""Red-team negative tests: rendering contract and frozen-input tampering (CTS Stage 0).

Tokenizer-dependent tests use the local pinned snapshot only (offline) and are skipped when it is absent.
Tampering is done on temporary copies of individual frozen files; the frozen package itself is never modified,
the authoring directory is never copied, and no D/C prompt text is printed.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.cts_stage0 import checks, package as pkg  # noqa: E402
from slgeo.cts_stage0.package import FrozenPackage, FrozenPackageError, sha256_path, sha256_text  # noqa: E402
from slgeo.cts_stage0.render import (  # noqa: E402
    NONCE_PERSONA_ID,
    SCIENTIFIC,
    TECHNICAL_VALIDATION,
    RenderError,
    Renderer,
    _check_content,
)

PACKAGE_DIR = ROOT / "research" / "cts_stage0_v1"
SNAPSHOT = (
    Path.home() / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots"
    / "a09a35458c702b33eeacc393d103063234e8bc28"
)
USER_PROMPT = "Name your favorite animal using only one word."


@pytest.fixture(scope="module")
def package() -> FrozenPackage:
    return FrozenPackage.from_repo(ROOT)


@pytest.fixture(scope="module")
def tokenizer():
    if not (SNAPSHOT / "tokenizer.json").is_file():
        pytest.skip("pinned Qwen2.5-7B-Instruct tokenizer snapshot not available locally")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(str(SNAPSHOT), local_files_only=True)


class FakePackage:
    """Minimal stand-in exposing ``persona`` over an explicit persona table (no file access)."""

    def __init__(self, personas: dict):
        self.personas = personas

    def persona(self, persona_id):
        try:
            return self.personas[persona_id]
        except KeyError as exc:
            raise FrozenPackageError(f"Unknown persona id: {persona_id!r}") from exc


def _personas_copy(package):
    return json.loads(json.dumps(package.personas))


# ---------------------------------------------------------------------------------------------- 3. render


@pytest.mark.parametrize("content", ["", " ", "\n", " padded", "padded\n", None, 3])
def test_check_content_refuses_empty_unstripped_and_non_strings(content):
    with pytest.raises(RenderError):
        _check_content("system prompt", content)


def test_empty_system_prompt_refused_through_renderer(package, tokenizer):
    personas = _personas_copy(package)
    personas["P_empty"] = {"system_prompt": "", "sha256": sha256_text(""), "content_token_ids": []}
    renderer = Renderer(tokenizer, FakePackage(personas))
    with pytest.raises(RenderError, match="empty system prompt"):
        renderer.render("P_empty", USER_PROMPT)


def test_non_default_persona_with_system_none_refused(package, tokenizer):
    """RT-10: only P_default may render as the default context."""
    personas = _personas_copy(package)
    personas["P_cat_T1"]["system_prompt"] = None
    with pytest.raises(RenderError, match="Only P_default"):
        Renderer(tokenizer, FakePackage(personas)).render("P_cat_T1", USER_PROMPT)


@pytest.mark.parametrize("typo", ["P_cat_t1", "P_Cat_T1", "P_cat_T1 ", "cat", "", "P_defualt", "p_default"])
def test_persona_id_typo_raises(package, tokenizer, typo):
    with pytest.raises((FrozenPackageError, RenderError)):
        Renderer(tokenizer, package).render(typo, USER_PROMPT)


@pytest.mark.parametrize("user", ["", "  ", f" {USER_PROMPT}", f"{USER_PROMPT}\n"])
def test_bad_user_prompt_refused(package, tokenizer, user):
    with pytest.raises(RenderError):
        Renderer(tokenizer, package).render("P_default", user)


def test_technical_validation_renderer_refuses_every_real_persona(package, tokenizer):
    renderer = Renderer(tokenizer, package, mode=TECHNICAL_VALIDATION)
    refused = 0
    for persona_id in sorted(package.personas):
        if persona_id == "P_default":
            continue
        with pytest.raises(RenderError, match="Technical validation refuses"):
            renderer.render(persona_id, USER_PROMPT)
        refused += 1
    assert refused == len(package.personas) - 1 == 43
    renderer.render("P_default", USER_PROMPT)
    renderer.render(NONCE_PERSONA_ID, USER_PROMPT)


def test_scientific_renderer_refuses_nonce_persona(package, tokenizer):
    with pytest.raises(RenderError, match="technical-validation only"):
        Renderer(tokenizer, package, mode=SCIENTIFIC).render(NONCE_PERSONA_ID, USER_PROMPT)
    with pytest.raises(ValueError):
        Renderer(tokenizer, package, mode="debug")


def test_prompt_len_uses_text_path_not_batchencoding(package, tokenizer):
    """RT-01: prompt_len must equal len(tokenizer(text).input_ids); tokenize=True differs under 5.x."""
    import transformers

    rendered = Renderer(tokenizer, package).render("P_default", USER_PROMPT)
    assert rendered.prompt_len == len(tokenizer(rendered.text)["input_ids"]) == len(rendered.input_ids)
    assert rendered.input_ids[-3:] == (151644, 77091, 198)
    messages = [{"role": "user", "content": USER_PROMPT}]
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if int(transformers.__version__.split(".")[0]) >= 5:
        assert len(encoded) != rendered.prompt_len  # the BatchEncoding pitfall is real in this environment
        assert list(encoded["input_ids"]) == list(rendered.input_ids)
    else:
        assert list(encoded) == list(rendered.input_ids)


def test_default_render_is_the_qwen_default_context(package, tokenizer):
    """RT-12: evaluation prompts render in the Qwen default context, never P_helpful."""
    rendered = Renderer(tokenizer, package).render("P_default", USER_PROMPT)
    helpful = Renderer(tokenizer, package).render("P_helpful", USER_PROMPT)
    assert pkg.DEFAULT_SYSTEM_TEXT in rendered.text
    assert rendered.input_ids != helpful.input_ids


# ------------------------------------------------------------------------------------ 4. frozen tampering


def test_modified_persona_string_refused_by_hash(package, tokenizer):
    personas = _personas_copy(package)
    personas["P_cat_T1"]["system_prompt"] = personas["P_cat_T1"]["system_prompt"].replace("cats", "Cats", 1)
    with pytest.raises(RenderError, match="hash mismatch"):
        Renderer(tokenizer, FakePackage(personas)).render("P_cat_T1", USER_PROMPT)


def test_template_plural_persona_refused(package, tokenizer):
    """RT-11: a persona rebuilt with the naive '{animal}s' plural ('wolfs') is refused."""
    personas = _personas_copy(package)
    rebuilt = personas["P_wolf_T1"]["system_prompt"].replace("wolves", "wolfs")
    assert rebuilt != personas["P_wolf_T1"]["system_prompt"]
    personas["P_wolf_T1"]["system_prompt"] = rebuilt
    with pytest.raises(RenderError, match="hash mismatch"):
        Renderer(tokenizer, FakePackage(personas)).render("P_wolf_T1", USER_PROMPT)
    personas["P_wolf_T1"]["sha256"] = sha256_text(rebuilt)  # even with a matching hash, the token ids differ
    with pytest.raises(RenderError, match="content tokens"):
        Renderer(tokenizer, FakePackage(personas)).render("P_wolf_T1", USER_PROMPT)


def test_modified_content_token_ids_refused(package, tokenizer):
    personas = _personas_copy(package)
    personas["P_dog_T1"]["content_token_ids"] = personas["P_dog_T1"]["content_token_ids"][:-1]
    with pytest.raises(RenderError, match="content tokens"):
        Renderer(tokenizer, FakePackage(personas)).render("P_dog_T1", USER_PROMPT)


def _tmp_package(tmp_path: Path, files: list[str]) -> Path:
    root = tmp_path / "frozen_copy"
    root.mkdir()
    for name in files:
        shutil.copy2(PACKAGE_DIR / name, root / name)
    return root


def test_manifest_hash_change_raises(tmp_path):
    root = _tmp_package(tmp_path, ["MANIFEST.json"])
    (root / "MANIFEST.json").write_bytes((root / "MANIFEST.json").read_bytes() + b" ")
    with pytest.raises(FrozenPackageError, match="frozen manifest hash"):
        FrozenPackage(root)


def test_missing_manifest_raises(tmp_path):
    with pytest.raises(FrozenPackageError, match="missing"):
        FrozenPackage(tmp_path)


def _rehashed_manifest(root: Path, monkeypatch, listed: list[str], override: dict | None = None) -> None:
    """Write a manifest listing ``listed`` with their frozen hashes and pin its own hash for the test."""
    frozen = json.loads((PACKAGE_DIR / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest = dict(frozen, files={name: frozen["files"][name] for name in listed})
    manifest["files"].update(override or {})
    (root / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(pkg, "MANIFEST_SHA256", sha256_path(root / "MANIFEST.json"))


def test_tampered_frozen_file_raises(tmp_path, monkeypatch):
    names = ["cts_stage0_personas.json", "cts_stage0_endpoint_tokens.json"]
    root = _tmp_package(tmp_path, names)
    _rehashed_manifest(root, monkeypatch, names)
    FrozenPackage(root)  # the untampered copy verifies
    data = json.loads((root / "cts_stage0_personas.json").read_text(encoding="utf-8"))
    data["personas"][1]["system_prompt"] = data["personas"][1]["system_prompt"] + " "
    (root / "cts_stage0_personas.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(FrozenPackageError, match="Frozen file hash mismatch"):
        FrozenPackage(root)


def test_extra_or_missing_frozen_file_raises(tmp_path, monkeypatch):
    names = ["cts_stage0_personas.json"]
    root = _tmp_package(tmp_path, names)
    _rehashed_manifest(root, monkeypatch, names)
    (root / "planted.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FrozenPackageError, match="extra"):
        FrozenPackage(root)
    (root / "planted.json").unlink()
    (root / "cts_stage0_personas.json").unlink()
    with pytest.raises(FrozenPackageError, match="missing"):
        FrozenPackage(root)


def _loader_package(tmp_path: Path, mutate_prompts=None, mutate_partition=None) -> FrozenPackage:
    """A FrozenPackage over tmp copies of the partition and prompt files (verification bypassed on purpose)."""
    root = tmp_path / "loader"
    root.mkdir()
    partition = json.loads((PACKAGE_DIR / "cts_stage0_partition.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (PACKAGE_DIR / "cts_stage0_prompts.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if mutate_prompts is not None:
        mutate_prompts(rows)
    if mutate_partition is not None:
        mutate_partition(partition)
    (root / "cts_stage0_partition.json").write_text(json.dumps(partition), encoding="utf-8")
    (root / "cts_stage0_prompts.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    loader = object.__new__(FrozenPackage)
    loader.root = root
    return loader


def test_loader_unmodified_copy_returns_only_s0(tmp_path, package):
    loader = _loader_package(tmp_path)
    records = loader.s0_prompts()
    s0 = set(package.partition_ids()["S0"])
    assert {r.prompt_id for r in records} == s0 and len(records) == 334


def _first_d_row(rows):
    return next(row for row in rows if row["set"] == "D")


def test_loader_refuses_d_row_labelled_s0(tmp_path):
    """RT-41: a D row whose set field says S0 is refused before any forward."""

    def flip(rows):
        _first_d_row(rows)["set"] = "S0"

    with pytest.raises(FrozenPackageError, match="not in the S0 partition list"):
        _loader_package(tmp_path, mutate_prompts=flip).s0_prompts()


def test_loader_refuses_d_id_moved_into_s0_partition(tmp_path):
    moved = {}

    def flip(rows):
        row = _first_d_row(rows)
        row["set"] = "S0"
        moved["id"] = row["prompt_id"]

    def move(partition):
        sets = partition["sets"]
        sets["D"]["prompt_ids"] = [i for i in sets["D"]["prompt_ids"] if i != moved["id"]]
        sets["S0"]["prompt_ids"] = sets["S0"]["prompt_ids"] + [moved["id"]]

    with pytest.raises(FrozenPackageError):
        _loader_package(tmp_path, mutate_prompts=flip, mutate_partition=move).s0_prompts()


def test_loader_refuses_s0_row_dropped(tmp_path):
    def drop(rows):
        index = next(i for i, row in enumerate(rows) if row["set"] == "S0")
        del rows[index]

    with pytest.raises(FrozenPackageError, match="S0 prompt ids"):
        _loader_package(tmp_path, mutate_prompts=drop).s0_prompts()


def test_loader_refuses_s0_text_edit(tmp_path):
    def edit(rows):
        row = next(row for row in rows if row["set"] == "S0")
        row["prompt"] = row["prompt"] + " Answer briefly."

    with pytest.raises(FrozenPackageError, match="Prompt hash mismatch"):
        _loader_package(tmp_path, mutate_prompts=edit).s0_prompts()


def test_loader_refuses_s0_text_and_hash_edit(tmp_path):
    def edit(rows):
        row = next(row for row in rows if row["set"] == "S0")
        row["prompt"] = row["prompt"] + " Answer briefly."
        row["sha256"] = sha256_text(row["prompt"])

    with pytest.raises(FrozenPackageError, match="hash-of-hashes"):
        _loader_package(tmp_path, mutate_prompts=edit).s0_prompts()


def test_loader_refuses_dropped_c_row(tmp_path):
    def drop(rows):
        index = next(i for i, row in enumerate(rows) if row["set"] == "C")
        del rows[index]

    with pytest.raises(FrozenPackageError, match="Non-S0 rows"):
        _loader_package(tmp_path, mutate_prompts=drop).s0_prompts()


# ----------------------------------------------------------------------- retokenization (tokenizer needed)


@pytest.fixture(scope="module")
def rule_sets(tokenizer):
    return checks.boundary_ids_by_rule(tokenizer), checks.cjk_ids_by_rule(tokenizer)


@pytest.fixture
def fast_rules(monkeypatch, rule_sets):
    boundary, cjk = rule_sets
    monkeypatch.setattr(checks, "boundary_ids_by_rule", lambda tokenizer: list(boundary))
    monkeypatch.setattr(checks, "cjk_ids_by_rule", lambda tokenizer: list(cjk))


def _endpoint_package(package, endpoint=None, personas=None):
    return SimpleNamespace(
        endpoint=endpoint if endpoint is not None else json.loads(json.dumps(package.endpoint)),
        personas=personas if personas is not None else _personas_copy(package),
    )


def test_retokenization_passes_on_frozen_inputs(package, tokenizer, fast_rules):
    result = checks.retokenization_check(tokenizer, package)
    assert result["pass"], {k: v for k, v in result.items() if k.endswith(("ok", "equal", "failures"))}


@pytest.mark.parametrize("mutation", ["drop_boundary_id", "add_im_start", "boundary_hash", "cjk_hash", "special_id"])
def test_retokenization_detects_boundary_tampering(package, tokenizer, fast_rules, mutation):
    fake = _endpoint_package(package)
    endpoint = fake.endpoint
    if mutation == "drop_boundary_id":
        endpoint["boundary_ids"] = endpoint["boundary_ids"][1:]
    elif mutation == "add_im_start":
        endpoint["boundary_ids"] = sorted(set(endpoint["boundary_ids"]) | {151644})
    elif mutation == "boundary_hash":
        endpoint["boundary_ids_sha256"] = "0" * 64
    elif mutation == "cjk_hash":
        endpoint["cjk_ids_sha256"] = "0" * 64
    else:
        endpoint["boundary_special_ids"] = {"<|im_end|>": 151644, "<|endoftext|>": 151643}
    assert checks.retokenization_check(tokenizer, fake)["pass"] is False


def test_retokenization_detects_boundary_rule_drift(package, tokenizer, monkeypatch, rule_sets):
    """RT-37: the runtime rule output (not the stored list) changes -> the frozen hash check fails."""
    boundary, cjk = rule_sets
    monkeypatch.setattr(checks, "boundary_ids_by_rule", lambda tokenizer: [i for i in boundary if i < 151643])
    monkeypatch.setattr(checks, "cjk_ids_by_rule", lambda tokenizer: list(cjk))
    result = checks.retokenization_check(tokenizer, package)
    assert result["boundary_sha256_ok"] is False and result["pass"] is False


def test_retokenization_detects_persona_and_form_tampering(package, tokenizer, fast_rules):
    personas = _personas_copy(package)
    personas["P_wolf_T1"]["system_prompt"] = personas["P_wolf_T1"]["system_prompt"].replace("wolves", "wolfs")
    fake = _endpoint_package(package, personas=personas)
    result = checks.retokenization_check(tokenizer, fake)
    assert result["persona_failures"] == ["P_wolf_T1"] and result["pass"] is False

    fake = _endpoint_package(package)
    fake.endpoint["answer_forms"]["cat"]["forms"][4]["token_ids"] = [34, 1862, 198]
    result = checks.retokenization_check(tokenizer, fake)
    assert len(result["form_failures"]) == 1 and result["pass"] is False

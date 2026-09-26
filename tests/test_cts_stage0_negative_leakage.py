"""Red-team negative tests: input guards, adapter prohibition and D/C leak scan (CTS Stage 0).

No student/adapter/C18/results path is opened: every refused path is a synthetic name under a temporary root.
D/C fingerprints are read in memory only, planted into temporary files, and never printed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.cts_stage0 import guards, integrity  # noqa: E402
from slgeo.cts_stage0.guards import GuardError, guard_input, load_frozen_teacher  # noqa: E402
from slgeo.cts_stage0.package import FrozenPackage  # noqa: E402


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    # A neutral base name: pytest otherwise embeds the test name, which could itself match a denied pattern.
    base = tmp_path_factory.mktemp("guardroot", numbered=True)
    (base / "data").mkdir()
    return base


DENIED = [
    "adapter_model.safetensors",
    "run/adapter_config.json",
    "student_lora/weights.bin",
    "runs/checkpoint-123/state.pt",
    "runs/epoch_03/weights.pt",
    "vectors/v_student.pt",
    "geometry/alignment.json",
    "runs/seed_2/metrics.json",
    "reference_reproduction_4080/log.txt",
    "results/confirmatory/table.json",
    "results/geometry/attribution/x.json",
    "lora_updates/delta.pt",
    "bidirectional/summary.json",
    "C18/decision.json",
    "c18/decision.json",
    "runs_c18_final/x.json",
    "x.c18.json",
    "research/cts_stage0_v1/authoring/stems_direct.txt",
    "Research/CTS_Stage0_V1/Authoring/stems.txt",
]


@pytest.mark.parametrize("relative", DENIED)
def test_guard_input_refuses_denied_paths(root, relative):
    path = root / "data" / relative
    with pytest.raises(GuardError, match="denied pattern"):
        guard_input(path, [root / "data"])


def test_guard_input_accepts_normal_data_path(root):
    path = root / "data" / "generated" / "reference_qwen7b_cat_subliminal_30k.jsonl"
    assert guard_input(path, [root / "data"]) == path.resolve()
    assert guard_input(root / "data", [root / "data"]) == (root / "data").resolve()


@pytest.mark.parametrize("relative", ["other/x.jsonl", "data/../other/x.jsonl", "datax/x.jsonl"])
def test_guard_input_refuses_paths_outside_allowed_roots(root, relative):
    with pytest.raises(GuardError, match="outside the allowed input roots"):
        guard_input(root / relative, [root / "data"])


def test_guard_input_refuses_with_no_roots(root):
    with pytest.raises(GuardError):
        guard_input(root / "data" / "x.jsonl", [])


def test_denied_does_not_flag_frozen_teacher_or_benign_names(root):
    assert guards.denied(root / "data" / guards.FROZEN_V_TEACHER_RELATIVE) is None
    for benign in ("seeds.json", "epochal.txt", "c180/x.json", "abc18/x.json", "checkpoint_notes.md"):
        assert guards.denied(root / "data" / benign) is None, benign


def test_guard_input_refuses_authoring_directory_itself(root):
    with pytest.raises(GuardError):
        guard_input(root / "data" / "research" / "cts_stage0_v1" / "authoring", [root / "data"])


def test_pipeline_extraction_path_is_guarded(root, monkeypatch):
    from slgeo.cts_stage0 import pipeline
    from slgeo.cts_stage0.package import load_extraction_prompts

    planted = root / "data" / "student_lora" / "reference.jsonl"
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_text('{"prompt": "1 2 3"}\n', encoding="utf-8")
    monkeypatch.setenv("SLGEO_SHARED_ROOT", str(root))
    ctx = types.SimpleNamespace(repo_root=ROOT)
    package = FrozenPackage.from_repo(ROOT)
    with pytest.raises(GuardError):
        load_extraction_prompts(package, pipeline.shared_path(ctx, "data/student_lora/reference.jsonl"))


# ----------------------------------------------------------------------------------- frozen teacher


def test_load_frozen_teacher_refuses_wrong_path(root):
    wrong = root / "data" / "vectors" / "v_teacher.pt"
    wrong.parent.mkdir(parents=True, exist_ok=True)
    wrong.write_bytes(b"not a tensor")
    with pytest.raises(GuardError, match="frozen_t_cat"):
        load_frozen_teacher(wrong)
    renamed = root / "data" / "frozen_t_cat" / "v_teacher_copy.pt"
    renamed.parent.mkdir(parents=True, exist_ok=True)
    renamed.write_bytes(b"not a tensor")
    with pytest.raises(GuardError, match="frozen_t_cat"):
        load_frozen_teacher(renamed)


def test_load_frozen_teacher_refuses_wrong_hash(root):
    torch = pytest.importorskip("torch")
    staged = root / "data" / "inputs" / "frozen_t_cat" / "v_teacher.pt"
    staged.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"raw": torch.zeros(29, 3584)}, staged)
    with pytest.raises(GuardError, match="hash mismatch"):
        load_frozen_teacher(staged)


# ------------------------------------------------------------------------------- adapter prohibition


def test_importing_every_module_does_not_import_peft():
    code = (
        "import importlib, pkgutil, sys\n"
        f"sys.path.insert(0, {str(SRC)!r})\n"
        "import slgeo.cts_stage0 as p\n"
        "names = [m.name for m in pkgutil.iter_modules(p.__path__)]\n"
        "for name in names: importlib.import_module('slgeo.cts_stage0.' + name)\n"
        "from slgeo.cts_stage0 import selftest\n"
        "selftest.tiny_model(layers=2)\n"
        "print(len(names), 'peft' in sys.modules, any(k.startswith('peft') for k in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=240, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr[-2000:]
    count, peft, peft_any = out.stdout.split()[-3:]
    assert int(count) >= 20
    assert peft == "False" and peft_any == "False"


def test_assert_no_peft_and_assert_base_model_refuse_peft_in_sys_modules(monkeypatch):
    pytest.importorskip("transformers")
    from slgeo.cts_stage0.modeling import ModelContractError, assert_base_model
    from slgeo.cts_stage0.selftest import tiny_model

    model = tiny_model(layers=2)
    monkeypatch.setitem(sys.modules, "peft", types.ModuleType("peft"))
    with pytest.raises(GuardError, match="peft"):
        guards.assert_no_peft()
    with pytest.raises(ModelContractError, match="peft"):
        assert_base_model(model)


def test_assert_base_model_refuses_lora_like_module():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from slgeo.cts_stage0.modeling import ModelContractError, assert_base_model
    from slgeo.cts_stage0.selftest import tiny_model

    model = tiny_model(layers=2)
    with pytest.raises(ModelContractError, match="geometry"):  # a clean tiny model fails only on geometry
        assert_base_model(model)
    model.model.layers[0].self_attn.q_proj.add_module("lora_A", torch.nn.Linear(64, 4, bias=False))
    with pytest.raises(ModelContractError, match="Adapter-like"):
        assert_base_model(model)
    clean = tiny_model(layers=2)
    clean.peft_config = {"default": object()}
    with pytest.raises(ModelContractError, match="Unexpected model type"):
        assert_base_model(clean)


# ------------------------------------------------------------------------------------ D/C leak scan


class _Opaque(dict):
    """D/C fingerprints with an opaque repr, so a failing test never prints a D/C prompt text."""

    def __repr__(self) -> str:
        return "<D/C fingerprints (redacted)>"


@pytest.fixture(scope="module")
def fingerprints():
    return _Opaque(FrozenPackage.from_repo(ROOT).dc_fingerprints())


@pytest.fixture(scope="module")
def s0_id():
    return sorted(FrozenPackage.from_repo(ROOT).partition_ids()["S0"])[0]


def _pick(values):
    return sorted(values)[0]


def _scan(directory, fingerprints):
    return integrity.dc_leak_scan([directory], fingerprints)


def test_dc_scan_clean_output_passes_and_empty_output_fails(tmp_path, fingerprints, s0_id):
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "criteria.json").write_text(json.dumps({"S0": [s0_id], "n": 3}), encoding="utf-8")
    np.savez(clean / "scores.npz", cids=np.asarray(["unsteered"]), x=np.zeros(3))
    assert _scan(clean, fingerprints)["pass"] is True
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _scan(empty, fingerprints)["pass"] is False  # fail closed: nothing scanned
    assert integrity.dc_leak_scan([tmp_path / "missing"], fingerprints)["pass"] is False


@pytest.mark.parametrize("kind", ["id", "sha256", "text"])
@pytest.mark.parametrize("container", ["json", "log", "npz"])
def test_dc_scan_detects_planted_fingerprint(tmp_path, fingerprints, s0_id, kind, container):
    if kind == "id":
        value = _pick(fingerprints["ids"])
    elif kind == "sha256":
        value = _pick(fingerprints["sha256"])
    else:
        value = _pick([text for text in fingerprints["texts"] if len(text) >= 12])
    out = tmp_path / "out"
    (out / "shard").mkdir(parents=True)
    (out / "shard" / "benign.json").write_text("{}", encoding="utf-8")
    if container == "json":
        (out / "shard" / "descriptive.json").write_text(json.dumps({"note": value}), encoding="utf-8")
    elif container == "log":
        (out / "job.err").write_text(f"[cts-stage0] processing {value}\n", encoding="utf-8")
    else:
        np.savez(out / "shard" / "scores.npz", prompt_ids=np.asarray([s0_id, value]))
    result = _scan(out, fingerprints)
    assert result["pass"] is False and result["files_with_hits"] == 1


def test_dc_scan_detects_id_as_npz_key(tmp_path, fingerprints):
    out = tmp_path / "out"
    out.mkdir()
    np.savez(out / "extras.npz", **{_pick(fingerprints["ids"]): np.zeros(2)})
    assert _scan(out, fingerprints)["pass"] is False


def test_dc_scan_detects_id_glued_to_underscore(tmp_path, fingerprints):
    out = tmp_path / "out"
    out.mkdir()
    np.savez(out / "extras.npz", **{f"kl_{_pick(fingerprints['ids'])}": np.zeros(2)})
    assert _scan(out, fingerprints)["pass"] is False


def test_dc_scan_detects_id_in_file_name(tmp_path, fingerprints):
    out = tmp_path / "out"
    out.mkdir()
    (out / f"{_pick(fingerprints['ids'])}.json").write_text("{}", encoding="utf-8")
    assert _scan(out, fingerprints)["pass"] is False


def test_dc_scan_detects_id_in_npy(tmp_path, fingerprints):
    out = tmp_path / "out"
    out.mkdir()
    np.save(out / "ids.npy", np.asarray([_pick(fingerprints["ids"])]))
    assert _scan(out, fingerprints)["pass"] is False


def test_dc_scan_detects_id_in_incoming_leftover(tmp_path, fingerprints):
    out = tmp_path / "out"
    out.mkdir()
    (out / "criteria.json").write_text("{}", encoding="utf-8")
    (out / "criteria.json.0123.incoming").write_text(json.dumps({"id": _pick(fingerprints["ids"])}), encoding="utf-8")
    assert _scan(out, fingerprints)["pass"] is False

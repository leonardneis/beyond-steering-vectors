from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from slgeo.analysis.c18_execution import (
    identity_block_hooks,
    replace_block_output,
    suffix_factorial_values,
)
from slgeo.analysis.c18_manifest import (
    expected_raw_ids,
    selection_inventory,
    validate_public_inputs,
    validate_manifest_contract,
)
from slgeo.analysis.c18_statistics import aggregate_y_rows, classify_outcome
from slgeo.analysis.interventions import list_lora_modules, mask_lora_modules
from slgeo.analysis.teacher_coordinate_interchange import (
    atomic_json,
    cancellation_flags,
    coordinate_clamp_hooks,
    deterministic_npz,
    four_cell_quantities,
    generate_orthogonal_families,
    iut_pass,
    normalize_rows_float64,
    replace_teacher_coordinate,
    seal_bytes,
    sha256_file,
    stratified_bootstrap_indices,
    unseal_bytes,
    validate_technical_report,
)
from slgeo.io import load_yaml
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scripts.generate_bidirectional_teacher_coordinate_interchange_dag import render as render_dag
from scripts.render_c18_technical_dag import render as render_runtime_dag
from scripts.validate_c18_execution_checkout import validate_checkout
from scripts.run_bidirectional_teacher_coordinate_interchange_manifest import command_plan


class TupleBlock(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = value

    def forward(self, hidden):
        return (hidden + self.value, f"tail-{self.value}")


class ToyModel(nn.Module):
    def __init__(self, count=28):
        super().__init__()
        self.model = SimpleNamespace(layers=nn.ModuleList([TupleBlock(i / 100) for i in range(count)]))

    def forward(self, hidden):
        tails = []
        for block in self.model.layers:
            hidden, tail = block(hidden)
            tails.append(tail)
        return hidden, tails


def non_axis_teacher():
    return torch.tensor([1.0, 2.0, -3.0, 0.5], dtype=torch.float64)


def test_exact_coordinate_replacement_non_axis_parallel():
    hidden = torch.tensor([[[0.25, -0.5, 0.75, 1.0]]], dtype=torch.float16)
    unit = non_axis_teacher() / torch.linalg.vector_norm(non_axis_teacher())
    donor = torch.tensor([[0.375]], dtype=torch.float64)
    changed, stats = replace_teacher_coordinate(hidden, donor, unit, torch.ones((1, 1)))
    actual = torch.dot(changed[0, 0].double(), unit)
    assert abs(float(actual - donor.item())) <= stats.cast_error_norm_max + 1e-12
    assert stats.bound_violation_max == 0


def test_orthogonal_component_unchanged_up_to_cast_error():
    hidden = torch.tensor([[[0.1, -0.2, 0.3, -0.4]]], dtype=torch.float16)
    unit = normalize_rows_float64(non_axis_teacher()[None])[0]
    donor = torch.tensor([[1.25]], dtype=torch.float64)
    changed, stats = replace_teacher_coordinate(hidden, donor, unit, torch.ones((1, 1)))
    delta = changed.double() - hidden.double()
    leakage = delta - torch.einsum("bth,h->bt", delta, unit)[..., None] * unit
    assert torch.linalg.vector_norm(leakage).item() <= stats.cast_error_norm_max + 1e-12


def test_padding_is_bit_identical():
    hidden = torch.randn(2, 4, 4).half()
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    donor = torch.zeros((2, 4), dtype=torch.float64)
    changed, _ = replace_teacher_coordinate(hidden, donor, non_axis_teacher(), mask)
    assert torch.equal(changed[mask == 0], hidden[mask == 0])


def test_tuple_output_hook_order_and_count():
    model = ToyModel()
    hidden = torch.zeros((1, 2, 4), dtype=torch.float16)
    teachers = normalize_rows_float64(non_axis_teacher().repeat(27, 1))
    mask = torch.ones((1, 2), dtype=torch.long)
    donors = {index: torch.zeros((1, 2), dtype=torch.float64) for index in range(27)}
    with coordinate_clamp_hooks(model, teachers, donors, mask) as audit:
        output, tails = model(hidden)
    assert output.shape == hidden.shape and len(tails) == 28
    assert audit["calls"] == list(range(27))
    assert len(audit["diagnostics"]) == 27


def test_hook_cleanup_success_and_exception():
    model = ToyModel()
    with identity_block_hooks(model) as calls:
        model(torch.zeros((1, 1, 4)))
    assert calls == list(range(27))
    assert all(not block._forward_hooks for block in model.model.layers)
    with pytest.raises(RuntimeError):
        with identity_block_hooks(model):
            raise RuntimeError("boom")
    assert all(not block._forward_hooks for block in model.model.layers)


def test_identity_hook_and_terminal_replay():
    model = ToyModel()
    hidden = torch.zeros((1, 1, 4))
    natural, _ = model(hidden)
    with identity_block_hooks(model):
        identity, _ = model(hidden)
    assert torch.equal(natural, identity)
    replacement = torch.full_like(hidden, 3.0)
    with replace_block_output(model, 27, replacement) as calls:
        terminal, _ = model(hidden)
    assert calls == [27] and torch.equal(terminal, replacement)


def test_null_operator_and_self_replay():
    hidden = torch.randn(2, 3, 4).half()
    teacher = normalize_rows_float64(non_axis_teacher()[None])[0]
    donor = torch.einsum("bth,h->bt", hidden.double(), teacher)
    replay, stats = replace_teacher_coordinate(hidden, donor, teacher, torch.ones((2, 3)))
    assert torch.equal(replay, hidden)
    assert stats.executed_delta_norm_max == 0


def test_batch_semantics_for_independent_toy_rows():
    model = ToyModel()
    batch = torch.randn(6, 4, 4)
    together, _ = model(batch)
    separate = torch.cat([model(batch[i : i + 1])[0] for i in range(6)])
    assert torch.equal(together, separate)


class FakeLora(nn.Module):
    def __init__(self):
        super().__init__()
        self.scaling = {"default": 1.0}


class FakePeft(nn.Module):
    def __init__(self):
        super().__init__()
        self.mods = nn.ModuleList([FakeLora() for _ in range(196)])


def test_mask_status_196_20_and_restore_success():
    model = FakePeft()
    names = list_lora_modules(model)
    assert len(names) == 196
    with mask_lora_modules(model, disabled_modules=names[:20]):
        assert sum(module.scaling["default"] == 0 for module in model.mods) == 20
    assert all(module.scaling["default"] == 1 for module in model.mods)


def test_mask_restore_after_exception():
    model = FakePeft()
    names = list_lora_modules(model)
    with pytest.raises(RuntimeError):
        with mask_lora_modules(model, disabled_modules=names[:20]):
            raise RuntimeError("expected")
    assert all(module.scaling["default"] == 1 for module in model.mods)


def test_four_cell_algebra():
    cells = {"Y00": np.array([1.0]), "Y01": np.array([2.0]), "Y10": np.array([4.0]), "Y11": np.array([8.0])}
    q = four_cell_quantities(cells)
    assert q["E"].item() == 7 and q["I"].item() == 3
    assert np.array_equal(q["E"], q["L"] + q["B1"])


def test_z_factorial_diagonals():
    state = {0: np.array([2.0]), 1: np.array([5.0])}
    suffix = {0: lambda x: x + 1, 1: lambda x: 2 * x}
    z = {f"Z{a}{b}": suffix[a](state[b]) for a in (0, 1) for b in (0, 1)}
    assert z["Z00"].item() == 3 and z["Z11"].item() == 10


def test_w_factorial_exact_allocations():
    w = {"W000": np.array([1.0]), "W001": np.array([2.0]), "W100": np.array([4.0]), "W101": np.array([8.0])}
    values = suffix_factorial_values(w, donor=0)
    assert values["total"].item() == 7
    assert values["upstream_under_suffix0"].item() + values["suffix_at_state1"].item() == 7


def test_bootstrap_determinism_and_family_shape():
    families = [family for family in ("a", "b", "c") for _ in range(24)]
    first = stratified_bootstrap_indices(families, draws=100, seed=20260804)
    second = stratified_bootstrap_indices(families, draws=100, seed=20260804)
    assert np.array_equal(first, second) and first.shape == (100, 72)


def test_iut_requires_both_strict_intervals():
    assert iut_pass((-0.01, 0.02), (-0.02, 0.01), 0.044)
    assert not iut_pass((-0.044, 0.01), (-0.02, 0.01), 0.044)


def test_cancellation_flags():
    eps = 0.044
    flags = cancellation_flags(
        {"B0": np.array([0.1, -0.1]), "B1": np.array([0.0, 0.0])},
        {"B0": {"S": 0.1, "N": 0.08}, "B1": {"S": 0.0, "N": 0.0}},
        {"B0": {"a": 0.1, "b": -0.1, "c": 0.0}, "B1": {"a": 0.0, "b": 0.0, "c": 0.0}},
        epsilon=eps,
    )
    assert "mean_absolute" in flags["B0"] and "opposing_families" in flags["B0"]
    assert not flags["B1"]


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"integrity_pass": False}, "technical_integrity_failure"),
        ({"reconstruction_pass": False}, "effect_reconstruction_failure"),
        ({"cancellation": True}, "cancellation_only_apparent_success"),
        ({}, "symmetric_transport"),
    ],
)
def test_outcome_matrix_ordered_disjoint(kwargs, expected):
    params = dict(
        integrity_pass=True, reconstruction_pass=True,
        intervals90={"B0": (-0.01, 0.01), "B1": (-0.01, 0.01)},
        intervals99={key: (-0.01, 0.01) for key in ("B0", "B1", "I")},
        cancellation=False, heterogeneous=False, effect_direction=-0.2, epsilon=0.044,
    )
    params.update(kwargs)
    assert classify_outcome(**params) == expected


def test_fail_closed_technical_report_rejects_outcomes():
    validate_technical_report({"status": "PASS", "batch_max_abs_error": 0.0})
    with pytest.raises(ValueError):
        validate_technical_report({"G_B0": 0.0})
    with pytest.raises(ValueError):
        validate_technical_report({"nested": {"Y01": [1.0]}})


def test_orthogonal_direction_normalization_deterministic():
    teacher = normalize_rows_float64(torch.randn(27, 19, generator=torch.Generator().manual_seed(4)))
    first = generate_orthogonal_families(teacher, family_count=5)
    second = generate_orthogonal_families(teacher, family_count=5)
    assert np.array_equal(first, second)
    assert np.max(np.abs(np.einsum("fsh,sh->fs", first, teacher.numpy()))) < 1e-12
    assert np.allclose(np.linalg.norm(first, axis=-1), 1.0)


def test_deterministic_artifact_and_hash(tmp_path):
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    arrays = {"b": np.arange(3), "a": np.eye(2)}
    deterministic_npz(first, arrays)
    deterministic_npz(second, arrays)
    assert sha256_file(first) == sha256_file(second)


def test_atomic_publication_refuses_overwrite(tmp_path):
    path = tmp_path / "record.json"
    atomic_json(path, {"a": 1})
    with pytest.raises(FileExistsError):
        atomic_json(path, {"a": 2})
    assert json.loads(path.read_text())["a"] == 1


def test_sealed_artifact_round_trip():
    cryptography = pytest.importorskip("cryptography.fernet")
    key = cryptography.Fernet.generate_key()
    sealed = seal_bytes(b"scientific payload", key)
    assert b"scientific payload" not in sealed
    assert unseal_bytes(sealed, key) == b"scientific payload"
    damaged = bytearray(sealed); damaged[-1] ^= 1
    with pytest.raises(Exception):
        unseal_bytes(bytes(damaged), key)


def test_manifest_contract_and_frozen_direction_hash():
    manifest = load_yaml(Path("configs/validation/cat_bidirectional_teacher_coordinate_interchange_v1.yaml"))
    validate_manifest_contract(manifest)
    item = manifest["frozen_inputs"]["orthogonal_directions"]
    assert sha256_file(item["path"]) == item["sha256"]
    assert len(selection_inventory(manifest["frozen_inputs"]["selection_plan"]["path"])) == 26
    checked = validate_public_inputs(manifest, Path("."), require_runtime_inputs=False)
    assert checked["teacher_tensor"] == "PASS" and checked["adapter_neutral"] == "MISSING"


def test_expected_artifact_inventory_is_id_based_and_complete():
    manifest = load_yaml(Path("configs/validation/cat_bidirectional_teacher_coordinate_interchange_v1.yaml"))
    inventory = expected_raw_ids(manifest, Path("."))
    assert len(inventory["Y"]) == 2 * 26 * 72 * 4
    assert len(inventory["Z"]) == 2 * 26 * 72 * 4
    assert len(inventory["W"]) == 2 * 26 * 72 * 8
    assert len(inventory["orthogonal"]) == 2 * 26 * 72 * 5 * 2


def test_aggregator_uses_raw_ids_not_positions():
    rows = []
    families = [family for family in ("a", "b", "c") for _ in range(24)]
    for condition in ("subliminal", "neutral"):
        for set_id in ["top", *[f"norm_{i:02d}" for i in range(25)]]:
            for prompt_index, family in enumerate(families):
                base = -0.3 if condition == "subliminal" and set_id == "top" else 0.0
                for cell, offset in {"Y00": 0.0, "Y01": 0.3, "Y10": 0.0, "Y11": 0.3}.items():
                    rows.append({"condition": condition, "set_id": set_id, "prompt_id": f"p{prompt_index:02d}", "family": family, "cell": cell, "margin": base + offset})
    rng = np.random.default_rng(9); rng.shuffle(rows)
    result = aggregate_y_rows(rows, bootstrap_draws=50)
    assert set(result["means"]) == {"E", "L", "H", "B0", "B1", "I"}


def test_htcondor_technical_dag_is_outcome_blind_and_ordered():
    manifest = load_yaml(Path("configs/validation/cat_bidirectional_teacher_coordinate_interchange_v1.yaml"))
    dag = render_dag(manifest, mode="technical", commit="a" * 40)
    assert "c18_technical_preflight" in dag and "c18_technical_00" in dag
    assert "c18_technical_audit" in dag
    assert "PARENT c18_technical_preflight CHILD c18_technical_00" in dag
    assert "PARENT c18_technical_00 CHILD c18_technical_audit" in dag
    assert "FINAL c18_notify condor/dag_notification.sub" in dag
    assert 'BsvNtfyTopic=""' in dag
    assert "subliminal" not in dag and "neutral" not in dag
    assert "@sha256:" in dag


def test_stdlib_runtime_dag_renderer_freezes_commit_and_notification():
    source = Path("condor/bidirectional_teacher_coordinate_interchange.dag").read_text()
    rendered = render_runtime_dag(
        source, commit="b" * 40, start_epoch=123,
        ntfy_topic="https://ntfy.sh/private-c18-topic/",
    )
    assert "UNFROZEN" not in rendered
    assert rendered.count("b" * 40) == 4
    assert 'BsvStartEpoch="123"' in rendered
    assert 'BsvNtfyTopic="https://ntfy.sh/private-c18-topic"' in rendered


def test_execution_checkout_validator_accepts_exact_clean_repository(tmp_path):
    from dulwich import porcelain

    porcelain.init(tmp_path)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("frozen\n", encoding="utf-8")
    porcelain.add(tmp_path, paths=[tracked])
    expected = porcelain.commit(
        tmp_path, message=b"frozen", author=b"C18 Test <c18@example.invalid>",
    ).decode("ascii")
    validate_checkout(tmp_path, expected)
    with pytest.raises(RuntimeError, match="commit differs"):
        validate_checkout(tmp_path, "0" * 40)
    tracked.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not clean"):
        validate_checkout(tmp_path, expected)


def test_manifest_plan_separates_technical_and_scientific_commands():
    manifest = load_yaml(Path("configs/validation/cat_bidirectional_teacher_coordinate_interchange_v1.yaml"))
    plan = command_plan(manifest)
    assert len(plan["technical"]) == 2 and len(plan["scientific"]) == 2
    assert "validate_bidirectional" in plan["technical"][0][0]
    assert all("run_bidirectional" in command[0] for command in plan["scientific"])

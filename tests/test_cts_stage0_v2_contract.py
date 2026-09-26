"""v2 contract: pins, reused v1 inputs, registry regeneration and code constants equal to the spec."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.cts_stage0 import conditions as cd  # noqa: E402
from slgeo.cts_stage0 import criteria as cr  # noqa: E402
from slgeo.cts_stage0 import fragility as fr  # noqa: E402
from slgeo.cts_stage0 import statistics as st  # noqa: E402
from slgeo.cts_stage0.contract import ContractError, V2Contract  # noqa: E402
from slgeo.cts_stage0.package import FrozenPackage  # noqa: E402

CONTRACT = ROOT / "research" / "cts_stage0_v2"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return yaml.safe_load((ROOT / "configs" / "validation" / "cts_stage0_v2.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def package() -> FrozenPackage:
    return FrozenPackage.from_repo(ROOT)


@pytest.fixture(scope="module")
def contract(package, manifest) -> V2Contract:
    return V2Contract.from_repo(ROOT, package, {k: manifest["contract"][k] for k in ("spec_sha256", "registry_sha256")})


@pytest.fixture(scope="module")
def spec(contract) -> dict:
    return contract.spec


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_pins_the_contract_files(manifest):
    assert manifest["contract"]["spec_sha256"] == _sha(CONTRACT / "cts_stage0_v2_decision_spec.json")
    assert manifest["contract"]["registry_sha256"] == _sha(CONTRACT / "cts_stage0_v2_condition_registry.jsonl")
    assert manifest["contract"]["status"] == "draft"


def test_pin_mismatch_is_refused(package, manifest):
    pins = {k: manifest["contract"][k] for k in ("spec_sha256", "registry_sha256")}
    with pytest.raises(ContractError):
        V2Contract.from_repo(ROOT, package, dict(pins, spec_sha256="0" * 64))
    with pytest.raises(ContractError):
        V2Contract.from_repo(ROOT, package, dict(pins, registry_sha256="0" * 64))


def test_reused_v1_inputs_match_their_hashes(spec, package):
    for name, digest in spec["frozen_v1_inputs_by_hash"]["files"].items():
        assert package.file_hashes[name] == digest
        assert _sha(package.root / name) == digest


def test_v1_package_is_byte_identical_to_its_manifest(package):
    # FrozenPackage construction already re-hashed every file; the manifest itself is pinned.
    assert _sha(package.root / "MANIFEST.json") == "6685d45685f3834b08d11641bbf058296ecfc34af48739a112f155bc27916ca0"


def test_registry_is_reproduced_by_the_generator(contract):
    contract.verify_regeneration()


def test_registry_counts(contract):
    rows = contract.rows
    assert len(rows) == 1579
    assert sum(r["gating"] for r in rows) == 1567
    groups = Counter((r["group"].split(":")[0], r["cost_class"], r["prompt_set"]) for r in rows)
    assert groups[("null", cd.L2, "S0_animal")] == 720
    assert groups[("rcov", cd.L2, "S0_animal")] == 597
    assert groups[("rcov_pc", cd.L2, "S0_animal")] == 99
    assert groups[("rcov_pc", cd.OWN, "S0_animal")] == 99
    assert groups[("persona", cd.OWN, "S0_all")] == 3
    assert sum(r["reference_rescore"] for r in rows) == 118


def test_registry_rows_roundtrip_to_conditions(contract):
    conditions = cd.registry_conditions(contract.rows)
    assert [c.cid for c in conditions] == [r["cid"] for r in contract.rows]
    for condition in conditions:
        assert condition.baseline == (cd.OWN_BASELINE if condition.cost_class == cd.OWN else cd.L2_BASELINE)


def test_tampered_registry_row_is_refused(contract):
    row = dict(contract.rows[10])
    row["kappa"] = 0.7
    with pytest.raises(cd.ConditionError):
        cd.condition_from_row(row)


def test_code_constants_equal_the_spec(spec):
    assert st.ALPHA_TS == spec["statistics"]["alpha_TS"]
    assert st.ALPHA_PC == spec["statistics"]["alpha_PC"]
    assert st.N_RCOV == spec["directions"]["random"]["R_cov"]["n"] == cd.N_RCOV
    assert cd.N_RCOV_PC == 99
    assert fr.EPSILON == spec["integrity"]["fragility"]["epsilon"]
    assert cr.SHARED_MARGIN_R == spec["decision"]["labels"]["BASE_SHARED_DIRECTION"]["r"]
    assert cr.R_THRESHOLD == spec["criteria"]["R"]["threshold"]
    assert list(cr.A4_WINDOW) == spec["criteria"]["A4"]["window"]
    assert list(cr.O_PANEL) == spec["endpoint"]["O_panel"]
    assert list(cr.O_PRIME) == spec["endpoint"]["O_for_c_cat_anim"]
    assert list(cd.TESTED_CONTRASTS) == spec["directions"]["tested_contrasts"]
    assert [row["class"] for row in spec["decision"]["order"]] == list(cr.DECISION_CLASSES)
    rank = spec["statistics"]["rank_test"]
    assert "<= 3" in rank["structured_null"] and st.rank_k_max(240, st.ALPHA_TS) == 3
    assert "<= 2" in rank["R_cov"] and st.rank_k_max(199, st.ALPHA_TS) == 2
    assert "<= 3" in rank["PC_reference"] and st.rank_k_max(99, st.ALPHA_PC) == 3


def test_personas_exist_in_the_v1_file(contract, package):
    assert set(contract.personas) <= set(package.personas)
    assert len(contract.personas) == 41
    assert not {"P_fox_T1", "P_owl_T1", "P_helpful"} & set(contract.personas)


def test_null_words_follow_the_v1_pool(contract, package):
    assert len(contract.null_words) == 16
    assert contract.null_words[0] == "bear" and contract.null_words[-1] == "mouse"
    null_rows = [r for r in contract.rows if r["group"] == "null:c_cat_dog"]
    assert len(null_rows) == 240


def test_cost_table_is_generated_output(contract):
    text = (CONTRACT / "COST_TABLE.md").read_text(encoding="utf-8")
    assert "**1579**" in text and "23.42" in text


def test_spec_is_the_machine_readable_authority(spec):
    assert spec["spec_version"] == "cts-stage0-v2"
    assert spec["execution_layout"]["canonical"] == "L2"
    assert spec["execution_layout"]["condition_batching"] == "not allowed in v2"
    json.dumps(spec)  # serializable

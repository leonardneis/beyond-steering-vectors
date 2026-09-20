"""Independently audit the outcome-blind C18-v2 technical validation bundle."""

from __future__ import annotations

import argparse
import json
import re

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.analysis.c18_v2_manifest import sha256_file, validate_manifest_contract  # noqa: E402
from slgeo.analysis.teacher_coordinate_interchange import atomic_json  # noqa: E402
from slgeo.io import load_yaml  # noqa: E402


EXPECTED_GATES = {
    "R", "H", "L_same", "identity", "null", "self_replay_p0", "self_replay_p1",
    "module_census_restore", "finiteness", "hook_order", "donor_immutability",
    "terminal_replay", "atomicity", "dag_classads",
}
EXPECTED_BLINDNESS = {
    "scientific_prompts_loaded": False, "scientific_selection_plan_loaded": False,
    "teacher_tensors_loaded": 0, "teacher_interchanges": 0, "cross_cells_computed": 0,
    "estimands_computed": 0, "raw_logits_persisted": False, "forbidden_open_attempts": 0,
}


def independently_validate_report(report: dict, manifest: dict) -> None:
    serialized = json.dumps(report, sort_keys=True)
    if any(token in serialized for token in ("Y00", "Y01", "Y10", "Y11", "G_B0", "G_B1", "candidate_logits")):
        raise RuntimeError("C18-v2 technical report contains a scientific outcome field")
    if report.get("schema_version") != 2 or report.get("experiment_id") != manifest["experiment_id"]:
        raise RuntimeError("C18-v2 technical report identity differs")
    if report.get("status") != "PASS" or report.get("mode") != "outcome_blind_synthetic_only":
        raise RuntimeError("C18-v2 technical report status/mode differs")
    if report.get("outcome_blindness") != EXPECTED_BLINDNESS:
        raise RuntimeError("C18-v2 outcome-blindness record differs")
    gates = report.get("hard_gates", {})
    if set(gates) != EXPECTED_GATES or any(value != "PASS" for value in gates.values()):
        raise RuntimeError("C18-v2 technical hard-gate inventory differs")
    if len(report.get("adapters", [])) != 2 or {item.get("condition") for item in report["adapters"]} != {"subliminal", "neutral"}:
        raise RuntimeError("C18-v2 technical adapter inventory differs")
    for item in report["adapters"]:
        if item.get("lora_module_count") != 196 or item.get("disabled_module_count") != 20:
            raise RuntimeError("C18-v2 LoRA census/mask report differs")
        if item.get("R") != item.get("H") or item.get("R") != item.get("L_same") or item.get("R") != "PASS":
            raise RuntimeError("C18-v2 R/H/L_same report differs")
        if item.get("self_replay") != {"P0": "PASS", "P1": "PASS"}:
            raise RuntimeError("C18-v2 self-replay report differs")
    scheduler = report.get("scheduler", {})
    if scheduler.get("task_id") != "c18v2_technical_00" or not str(scheduler.get("cluster_id", "")).isdigit() \
            or not str(scheduler.get("proc_id", "")).isdigit():
        raise RuntimeError("C18-v2 GPU ClassAds were not independently verified")
    commit = str(report.get("execution_git_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("C18-v2 execution commit is not frozen")
    actual = report.get("execution_identity", {})
    expected = manifest["execution"]
    for key in ("python", "torch", "cuda_runtime", "transformers", "peft", "bitsandbytes", "numpy",
                "gpu_class", "attention_backend", "container_image", "nvidia_driver"):
        if str(actual.get(key)) != str(expected.get(key)):
            raise RuntimeError(f"C18-v2 independently audited runtime mismatch: {key}")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",default="configs/validation/cat_bidirectional_teacher_coordinate_interchange_v2.yaml")
    parser.add_argument("--preflight",required=True); parser.add_argument("--validation",required=True)
    parser.add_argument("--output",required=True); args=parser.parse_args()
    manifest_path=repo_path(args.manifest); manifest=load_yaml(manifest_path); validate_manifest_contract(manifest)
    preflight_path=repo_path(args.preflight); validation_path=repo_path(args.validation)
    sidecar_path=validation_path.with_suffix(validation_path.suffix+".provenance.json")
    sums_path=validation_path.parent/"SHA256SUMS"
    for path in (preflight_path,validation_path,sidecar_path,sums_path):
        if not path.is_file() or path.stat().st_size==0: raise FileNotFoundError(path)
    preflight=json.loads(preflight_path.read_text(encoding="utf-8")); report=json.loads(validation_path.read_text(encoding="utf-8"))
    provenance=json.loads(sidecar_path.read_text(encoding="utf-8")); independently_validate_report(report, manifest)
    digest=sha256_file(manifest_path)
    if preflight.get("experiment_id")!=manifest["experiment_id"] or preflight.get("manifest_sha256")!=digest:
        raise RuntimeError("C18-v2 preflight identity differs")
    if not preflight.get("inputs") or any(value not in ("PASS","PASS_CANONICAL_LF_WORKTREE_REQUIRES_STAGING","CONTROL_SUCCESSOR")
                                               for value in preflight["inputs"].values()):
        raise RuntimeError("C18-v2 preflight input inventory did not pass")
    if preflight.get("outcome_blindness") != EXPECTED_BLINDNESS:
        raise RuntimeError("C18-v2 preflight outcome-blindness record differs")
    preflight_scheduler = preflight.get("scheduler", {})
    if preflight_scheduler.get("task_id") != "c18v2_technical_preflight" \
            or not str(preflight_scheduler.get("cluster_id", "")).isdigit() \
            or not str(preflight_scheduler.get("proc_id", "")).isdigit():
        raise RuntimeError("C18-v2 preflight ClassAds were not independently verified")
    if report["manifest_sha256"]!=digest or provenance.get("manifest_sha256")!=digest:
        raise RuntimeError("C18-v2 exact-manifest binding failed")
    if provenance.get("artifact_sha256")!=sha256_file(validation_path):
        raise RuntimeError("C18-v2 validation provenance digest failed")
    observed={}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        checksum,name=line.split("  ",1); observed[name]=checksum
    expected={validation_path.name:sha256_file(validation_path),sidecar_path.name:sha256_file(sidecar_path)}
    if observed!=expected: raise RuntimeError("C18-v2 SHA256SUMS failed")
    if report["execution_identity"]!=provenance["execution_identity"]:
        raise RuntimeError("C18-v2 execution identity differs across records")
    if provenance.get("git_commit") != report.get("execution_git_commit"):
        raise RuntimeError("C18-v2 execution commit differs across records")
    if provenance.get("git_dirty") not in ("0",0,False): raise RuntimeError("C18-v2 execution checkout was dirty")
    audit={"schema_version":2,"experiment_id":manifest["experiment_id"],"status":"PASS",
           "verdict":"READY_FOR_C18_V2_SCIENTIFIC_EXECUTION_AUTHORIZATION",
           "manifest_sha256":digest,"execution_commit":report["execution_git_commit"],
           "preflight_sha256":sha256_file(preflight_path),
           "validation_sha256":expected[validation_path.name],"provenance_sha256":expected[sidecar_path.name],
           "outcome_blind":True,"scientific_execution_authorized":False,
           "independent_checks":{"hard_gates":"PASS","runtime_identity":"PASS","artifact_hashes":"PASS",
               "dag_classads":"PASS","atomicity":"PASS","outcome_blindness":"PASS"}}
    atomic_json(repo_path(args.output),audit)


if __name__=="__main__": main()

"""CTS Stage 0 v2 command line.

Submit host (standard library only): ``submit-record``.
Cluster nodes (pinned container): ``plan``, ``run``, ``techval-cpu``, ``techval``, ``tv-project``.

Exit codes (E2): 0 success; 86 final (identity/integrity refusal, integrity-class failure, software defect;
never retried); 1 infrastructure (out-of-memory, operating-system I/O; retried at most twice per shard);
75 SIGTERM (retried); 85 GPU unavailable (mapped by the node wrapper; rematched in place).
Exit codes never depend on a Stage-0 outcome.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

from _bootstrap import bootstrap  # noqa: E402

ROOT = bootstrap()

MANIFEST = "configs/validation/cts_stage0_v2.yaml"
# Output namespaces (also in the manifest; duplicated here so the submit host needs no YAML parser).
OUTPUT_ROOT = "results/research/qwen7b_cts_stage0_v2"
TECHNICAL_VALIDATION_ROOT = "results/research/qwen7b_cts_stage0_v2_technical_validation"


def _shared_root() -> Path:
    return Path(os.environ.get("SLGEO_SHARED_ROOT", ROOT))


def _root(technical: bool) -> Path:
    """Scientific output root, or the TV-v2 attempt directory of the current run tag (one per attempt)."""
    if not technical:
        return _shared_root() / OUTPUT_ROOT
    tag = os.environ.get("SLGEO_RUN_TAG", "")
    if not tag.startswith("tv-") or not tag.replace("-", "").isalnum():
        raise RuntimeError("Technical validation requires SLGEO_RUN_TAG=tv-<utc stamp>")
    return _shared_root() / TECHNICAL_VALIDATION_ROOT / tag


def _manifest() -> dict:
    from slgeo.io import load_yaml

    manifest = load_yaml(ROOT / MANIFEST)
    if manifest["output_root"] != OUTPUT_ROOT or manifest["technical_validation_root"] != TECHNICAL_VALIDATION_ROOT:
        raise RuntimeError("Output namespaces differ between the manifest and the command line")
    return manifest


def _read_json(path: Path) -> dict:
    return json.loads(path.read_bytes())


def cmd_submit_record(args) -> int:
    """Submit host: git checks of the clean execution commit and the tracked-file listing (write-once)."""
    from slgeo.cts_stage0.atomic import atomic_write_json
    from slgeo.cts_stage0.provenance import submit_checks

    scientific = not args.technical_validation
    record = submit_checks(ROOT, require_frozen_contract=scientific)
    record["kind"] = "scientific" if scientific else "technical_validation"
    record["run_tag"] = args.run_tag
    atomic_write_json(_root(args.technical_validation) / "plan" / "run_record.json", record, write_once=True)
    print(f"{record['kind']} run record for {record['execution_commit']} ({len(record['tracked'])} tracked files)")
    return 0


def _verified_record(technical: bool) -> dict:
    from slgeo.cts_stage0.provenance import verify_tracked_blobs

    record = _read_json(_root(technical) / "plan" / "run_record.json")
    if record["execution_commit"] != os.environ.get("SLGEO_EXECUTION_GIT_COMMIT"):
        raise RuntimeError("Execution commit differs from the submitted run record")
    verify_tracked_blobs(ROOT, record["tracked"])
    return record


def _contract(manifest: dict):
    from slgeo.cts_stage0.contract import V2Contract
    from slgeo.cts_stage0.package import FrozenPackage

    package = FrozenPackage.from_repo(ROOT)
    pins = {key: manifest["contract"][key] for key in ("spec_sha256", "registry_sha256")}
    return V2Contract.from_repo(ROOT, package, pins)


def cmd_plan(args) -> int:
    """Container CPU job: build the deterministic plan from the contract and the manifest."""
    from slgeo.cts_stage0.atomic import atomic_write_json, sha256_file
    from slgeo.cts_stage0.identity import assert_identity
    from slgeo.cts_stage0.plan import build_plan, plan_sha256, projection

    manifest = _manifest()
    if manifest["contract"]["status"] != "frozen":
        raise RuntimeError("The scientific plan requires the frozen v2 contract")
    _verified_record(False)
    assert_identity(manifest["execution"], ROOT, require_gpu=False)
    contract = _contract(manifest)
    contract.verify_regeneration()
    plan = build_plan(contract, manifest)
    plan_record = {
        "plan_sha256": plan_sha256(plan),
        "run_record_sha256": sha256_file(_root(False) / "plan" / "run_record.json"),
        "projection_a100_h": projection(plan),
    }
    out = _root(False) / "plan"
    atomic_write_json(out / "plan.json", plan, write_once=True)
    atomic_write_json(out / "plan_record.json", plan_record, write_once=True)
    print(f"plan: {len(plan['shards'])} shards, {plan['n_conditions']} conditions ({plan['n_gating_conditions']} gating)")
    return 0


def _context():
    from slgeo.cts_stage0.atomic import sha256_file
    from slgeo.cts_stage0.pipeline import RunContext
    from slgeo.cts_stage0.plan import plan_sha256

    manifest = _manifest()
    root = _root(False)
    record = _read_json(root / "plan" / "run_record.json")
    plan_record = _read_json(root / "plan" / "plan_record.json")
    plan = _read_json(root / "plan" / "plan.json")
    if plan_sha256(plan) != plan_record["plan_sha256"]:
        raise RuntimeError("Plan hash differs from the plan record")
    if sha256_file(root / "plan" / "run_record.json") != plan_record["run_record_sha256"]:
        raise RuntimeError("Run record differs from the one the plan was built for")
    if record["execution_commit"] != os.environ.get("SLGEO_EXECUTION_GIT_COMMIT"):
        raise RuntimeError("Execution commit differs from the submitted run record")
    merged = dict(record, plan_sha256=plan_record["plan_sha256"])
    return RunContext(ROOT, manifest, root, plan, merged)


def cmd_run(args) -> int:
    from slgeo.cts_stage0.errors import classify
    from slgeo.cts_stage0.pipeline import attempt, run_stage

    ctx = _context()
    attempt(ctx, args.shard, "start")
    try:
        run_stage(ctx, args.shard)
    except BaseException as exc:
        event, code = classify(exc)
        attempt(ctx, args.shard, event, exit_code=code, error=f"{type(exc).__name__}: {str(exc)[:2000]}")
        print(f"{event}: {type(exc).__name__}", file=sys.stderr)
        raise SystemExit(code) from exc
    attempt(ctx, args.shard, "success")
    return 0


def _tv_common(require_gpu: bool):
    from slgeo.cts_stage0.identity import assert_identity
    from slgeo.cts_stage0.modeling import snapshot_directory, verify_snapshot

    manifest = _manifest()
    record = _verified_record(True)
    identity = assert_identity(manifest["execution"], ROOT, require_gpu=require_gpu)
    contract = _contract(manifest)
    snapshot = snapshot_directory(os.environ["HF_HOME"])
    snapshot_hashes = verify_snapshot(snapshot, manifest["model"]["snapshot_sha256"])
    return manifest, record, identity, contract, snapshot, snapshot_hashes


def _s0_length_profile(manifest, contract) -> tuple[dict, dict]:
    """The committed S0 length profile, verified against its pin, the contract, the tokenizer and the plan."""
    from slgeo.cts_stage0.package import sha256_path
    from slgeo.cts_stage0.plan import build_plan
    from slgeo.cts_stage0.s0_lengths import PROFILE_PATH, LengthProfileError, verify

    path = ROOT / PROFILE_PATH
    if sha256_path(path) != manifest["inputs"]["s0_length_profile_sha256"]:
        raise LengthProfileError("S0 length profile differs from its pinned hash")
    profile = json.loads(path.read_text(encoding="utf-8"))
    check = verify(profile, spec_sha256=contract.spec_sha256, registry_sha256=contract.registry_sha256,
                   package_manifest_sha256=manifest["frozen_package"]["manifest_sha256"], manifest=manifest,
                   plan=build_plan(contract, manifest))
    return profile, dict(check, sha256=manifest["inputs"]["s0_length_profile_sha256"])


def _planned_l2_shard_size(contract, manifest) -> int:
    from slgeo.cts_stage0.plan import build_plan

    plan = build_plan(contract, manifest)
    sizes = [len(s["payload"]["conditions"]) for s in plan["shards"]
             if s["stage"] == "score" and s["payload"]["cost_class"] == "L2_shared_prefix" and s["payload"]["prompt_set"] == "S0_animal"]
    return max(sizes)


def cmd_techval_cpu(args) -> int:
    """CPU technical validation in the execution environment (TV-v2)."""
    from slgeo.cts_stage0.atomic import atomic_write_json
    from slgeo.cts_stage0.checks import render_identity_check, retokenization_check
    from slgeo.cts_stage0.guards import guard_input
    from slgeo.cts_stage0.modeling import load_tokenizer
    from slgeo.cts_stage0.plan import build_plan, null_names, projection
    from slgeo.cts_stage0.preflight import rng_golden_check
    from slgeo.cts_stage0.selftest import tiny_model_suite
    from slgeo.cts_stage0.synthetic import artifact_drill, synthetic_decision_suite, synthetic_fragility_suite

    manifest, record, identity, contract, snapshot, snapshot_hashes = _tv_common(require_gpu=False)
    tokenizer = load_tokenizer(snapshot)
    extraction = guard_input(_shared_root() / manifest["inputs"]["extraction_file"], [_shared_root() / "data"])
    records = []
    with extraction.open(encoding="utf-8") as handle:
        for line in handle:
            if len(records) == 1024:
                break
            row = json.loads(line)
            records.append({"prompt": row["prompt"], "system_prompt": row["system_prompt"]})
    contract.verify_regeneration()
    plan = build_plan(contract, manifest)
    names = null_names(contract)
    # TV-v2 never reads S0: only the committed tokenizer-only length profile (verified against pin and plan).
    _profile, profile_check = _s0_length_profile(manifest, contract)
    result = {
        "s0_length_profile": profile_check,
        "contract_regenerated": {"pass": True, "registry_sha256": contract.registry_sha256},
        "retokenization": retokenization_check(tokenizer, contract.package),
        "render_identity": render_identity_check(tokenizer, contract.package, records),
        "tiny_model_suite": tiny_model_suite(),
        "rng_golden": rng_golden_check(),
        "synthetic_decision_suite": synthetic_decision_suite(contract.rows, names),
        "synthetic_fragility_suite": synthetic_fragility_suite(contract.rows, names),
        "artifact_drill": artifact_drill(_root(True) / "drill"),
        "plan": {"pass": True, "shards": len(plan["shards"]), "conditions": plan["n_conditions"], "gating": plan["n_gating_conditions"],
                 "placeholder_projection_a100_h": projection(plan)},
    }
    payload = {"kind": "cpu", "execution_commit": record["execution_commit"], "identity": identity,
               "snapshot_sha256": snapshot_hashes, "result": result, "pass": all(v.get("pass") for v in result.values())}
    atomic_write_json(_root(True) / "cpu" / "techval_cpu.json", payload, write_once=True)
    print(f"technical validation cpu: {'PASS' if payload['pass'] else 'FAIL'}")
    return 0 if payload["pass"] else 86  # a failed validation is final: never retried


def cmd_techval(args) -> int:
    """GPU technical validation on the real model (outcome-blind; whitelist output only)."""
    from slgeo.cts_stage0.atomic import atomic_write_json
    from slgeo.cts_stage0.guards import guard_input
    from slgeo.cts_stage0.modeling import load_model, load_tokenizer, set_deterministic
    from slgeo.cts_stage0.package import load_extraction_prompts
    from slgeo.cts_stage0.techval import run_technical_validation
    from slgeo.io import load_yaml

    manifest, record, identity, contract, snapshot, snapshot_hashes = _tv_common(require_gpu=True)
    cpu = _read_json(_root(True) / "cpu" / "techval_cpu.json")
    if not cpu.get("pass") or cpu["execution_commit"] != record["execution_commit"]:
        raise RuntimeError("CPU technical validation missing, failed or from another commit")
    profile, _check = _s0_length_profile(manifest, contract)
    determinism = set_deterministic()
    tokenizer = load_tokenizer(snapshot)
    model = load_model(snapshot, load_yaml(ROOT / manifest["model"]["model_config"]))
    extraction = guard_input(_shared_root() / manifest["inputs"]["extraction_file"], [_shared_root() / "data"])
    prompts = load_extraction_prompts(contract.package, extraction)
    result = run_technical_validation(model, tokenizer, contract.package, prompts, s0_profile=profile,
                                      l2_shard_conditions=_planned_l2_shard_size(contract, manifest))
    payload = {"kind": args.name, "execution_commit": record["execution_commit"], "identity": identity,
               "snapshot_sha256": snapshot_hashes, "determinism": determinism, "result": result}
    atomic_write_json(_root(True) / "gpu" / f"{args.name}.json", payload, write_once=True)
    verdict = bool(result["pass"])
    print(f"technical validation {args.name}: {'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 86


def cmd_techval_dry(args) -> int:
    """GPU dry shard: one planned L2 shard on TV inputs through the per-job work of a production shard. Its
    RemoteWallClockTime (read from the TV ledger by the projection node) gives the per-job fixed cost."""
    from slgeo.cts_stage0.atomic import atomic_write_json
    from slgeo.cts_stage0.guards import guard_input
    from slgeo.cts_stage0.modeling import load_model, load_tokenizer, set_deterministic
    from slgeo.cts_stage0.package import load_extraction_prompts
    from slgeo.cts_stage0.techval import run_dry_shard
    from slgeo.io import load_yaml

    manifest, record, identity, contract, snapshot, snapshot_hashes = _tv_common(require_gpu=True)
    cpu = _read_json(_root(True) / "cpu" / "techval_cpu.json")
    if not cpu.get("pass") or cpu["execution_commit"] != record["execution_commit"]:
        raise RuntimeError("CPU technical validation missing, failed or from another commit")
    profile, _check = _s0_length_profile(manifest, contract)
    determinism = set_deterministic()
    tokenizer = load_tokenizer(snapshot)
    model = load_model(snapshot, load_yaml(ROOT / manifest["model"]["model_config"]))
    extraction = guard_input(_shared_root() / manifest["inputs"]["extraction_file"], [_shared_root() / "data"])
    prompts = load_extraction_prompts(contract.package, extraction)
    result = run_dry_shard(model, tokenizer, contract.package, prompts, s0_profile=profile,
                           n_conditions=_planned_l2_shard_size(contract, manifest), publish_root=_root(True) / "dry")
    payload = {"kind": "tv_dry", "execution_commit": record["execution_commit"], "identity": identity,
               "snapshot_sha256": snapshot_hashes, "determinism": determinism, "result": result}
    atomic_write_json(_root(True) / "gpu" / "tv_dry.json", payload, write_once=True)
    verdict = bool(result["pass"])
    print(f"technical validation dry shard: {'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 86


def cmd_tv_project(args) -> int:
    """CPU node after both GPU validations: cross-host checks, measured seconds, projection P and the gate."""
    from slgeo.cts_stage0.atomic import atomic_write_json
    from slgeo.cts_stage0.plan import build_plan
    from slgeo.cts_stage0.techval import project

    manifest, record, identity, contract, _snapshot, _hashes = _tv_common(require_gpu=False)
    cpu = _read_json(_root(True) / "cpu" / "techval_cpu.json")
    gpus = [_read_json(_root(True) / "gpu" / f"{name}.json") for name in ("gpu_a", "gpu_b")]
    dry = _read_json(_root(True) / "gpu" / "tv_dry.json")
    # Written by this node's PRE script on the submit host from condor_history (RemoteWallClockTime).
    ledger = _read_json(_root(True) / "orchestration" / "budget_ledger.json")
    if any(g["execution_commit"] != record["execution_commit"] for g in gpus + [cpu, dry]):
        raise RuntimeError("Technical-validation records come from different commits")
    plan = build_plan(contract, manifest)
    budget = manifest["budget"]
    result = project(cpu, gpus, dry, ledger, plan, cap=float(budget["scientific_cap_a100_h"]),
                     planning_fraction=float(budget["planning_fraction"]))
    payload = {"kind": "projection", "execution_commit": record["execution_commit"], "identity": identity, "result": result}
    atomic_write_json(_root(True) / "projection.json", payload, write_once=True)
    verdict = bool(result["pass"])
    print(f"technical validation projection: {'PASS' if verdict else 'FAIL'}")
    return 0 if verdict else 86


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("submit-record")
    record.add_argument("--technical-validation", action="store_true")
    record.add_argument("--run-tag", required=True)
    record.set_defaults(func=cmd_submit_record)
    sub.add_parser("plan").set_defaults(func=cmd_plan)
    run = sub.add_parser("run")
    run.add_argument("--shard", required=True)
    run.set_defaults(func=cmd_run)
    techval = sub.add_parser("techval")
    techval.add_argument("--name", required=True, choices=("gpu_a", "gpu_b"))
    techval.set_defaults(func=cmd_techval)
    sub.add_parser("techval-cpu").set_defaults(func=cmd_techval_cpu)
    sub.add_parser("techval-dry").set_defaults(func=cmd_techval_dry)
    sub.add_parser("tv-project").set_defaults(func=cmd_tv_project)
    args = parser.parse_args()

    def terminate(_signum, _frame):
        raise SystemExit(75)

    signal.signal(signal.SIGTERM, terminate)
    try:
        return int(args.func(args) or 0)
    except SystemExit:
        raise
    except BaseException as exc:
        from slgeo.cts_stage0.errors import classify

        event, code = classify(exc)
        print(f"{event}: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(code) from exc


if __name__ == "__main__":
    sys.exit(main())

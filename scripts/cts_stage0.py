"""CTS Stage 0 command line.

Submit host (standard library only): ``submit-record``.
Cluster nodes (pinned container): ``plan``, ``run``, ``techval``, ``techval-cpu``.

Exit codes: 0 success; 1 software error; 75 terminated by SIGTERM (retried by the DAG); 86 execution-identity
or integrity refusal before any forward (never retried). Exit codes never depend on a Stage-0 outcome.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

from _bootstrap import bootstrap

ROOT = bootstrap()

MANIFEST = "configs/validation/cts_stage0_v1.yaml"
# Output namespaces (also in the manifest; duplicated here so the submit host needs no YAML parser).
OUTPUT_ROOT = "results/research/qwen7b_cts_stage0_v1"
TECHNICAL_VALIDATION_ROOT = "results/research/qwen7b_cts_stage0_v1_technical_validation"
IDENTITY_EXIT_CODE = 86


def _shared_root() -> Path:
    return Path(os.environ.get("SLGEO_SHARED_ROOT", ROOT))


def _root(technical: bool) -> Path:
    return _shared_root() / (TECHNICAL_VALIDATION_ROOT if technical else OUTPUT_ROOT)


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

    record = submit_checks(ROOT)
    record["kind"] = "technical_validation" if args.technical_validation else "scientific"
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


def cmd_plan(args) -> int:
    """Container CPU job: build the deterministic plan from the frozen package, choices and manifest."""
    from slgeo.cts_stage0.atomic import atomic_write_json, sha256_file
    from slgeo.cts_stage0.identity import assert_identity
    from slgeo.cts_stage0.package import FrozenPackage
    from slgeo.cts_stage0.plan import build_plan, load_choices, plan_sha256

    manifest = _manifest()
    record = _verified_record(False)
    assert_identity(manifest["execution"], ROOT, require_gpu=False)
    package = FrozenPackage.from_repo(ROOT)
    per_shard = int(manifest["plan"]["conditions_per_shard"])
    plan = build_plan(package, load_choices(ROOT, manifest), manifest, conditions_per_shard=per_shard)
    plan_record = {
        "plan_sha256": plan_sha256(plan),
        "run_record_sha256": sha256_file(_root(False) / "plan" / "run_record.json"),
        "conditions_per_shard": per_shard,
        "condition_batch_rows": int(manifest["scoring"]["condition_batch_rows"]),
        "equivalence_record_sha256": manifest["scoring"].get("equivalence_record_sha256"),
    }
    out = _root(False) / "plan"
    atomic_write_json(out / "plan.json", plan, write_once=True)
    atomic_write_json(out / "plan_record.json", plan_record, write_once=True)
    print(f"plan: {len(plan['shards'])} shards, {plan['n_conditions']} conditions ({plan['n_gating_conditions']} gating)")
    return 0 if record else 1


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
    if int(manifest["scoring"]["condition_batch_rows"]) != plan_record["condition_batch_rows"]:
        raise RuntimeError("Condition batching differs from the planned value")
    merged = dict(record)
    merged.update({key: plan_record[key] for key in ("plan_sha256", "equivalence_record_sha256")})
    return RunContext(ROOT, manifest, root, plan, merged)


def cmd_run(args) -> int:
    from slgeo.cts_stage0.guards import GuardError
    from slgeo.cts_stage0.identity import IdentityError
    from slgeo.cts_stage0.package import FrozenPackageError
    from slgeo.cts_stage0.pipeline import attempt, run_stage

    ctx = _context()
    attempt(ctx, args.shard, "start")
    try:
        run_stage(ctx, args.shard)
    except (IdentityError, FrozenPackageError, GuardError) as exc:
        attempt(ctx, args.shard, "refusal", error=f"{type(exc).__name__}: {str(exc)[:2000]}")
        print(f"refusal: {type(exc).__name__}", file=sys.stderr)
        return IDENTITY_EXIT_CODE
    except BaseException as exc:
        attempt(ctx, args.shard, "error", error=f"{type(exc).__name__}: {str(exc)[:2000]}")
        raise
    attempt(ctx, args.shard, "success")
    return 0


def cmd_techval(args) -> int:
    """GPU technical validation on the real model (outcome-blind; whitelist output only)."""
    from slgeo.cts_stage0.atomic import atomic_write_json
    from slgeo.cts_stage0.guards import guard_input
    from slgeo.cts_stage0.identity import assert_identity
    from slgeo.cts_stage0.modeling import load_model, load_tokenizer, set_deterministic, snapshot_directory, verify_snapshot
    from slgeo.cts_stage0.package import FrozenPackage, load_extraction_prompts
    from slgeo.cts_stage0.techval import run_technical_validation
    from slgeo.io import load_yaml

    manifest = _manifest()
    record = _verified_record(True)
    identity = assert_identity(manifest["execution"], ROOT, require_gpu=True)
    package = FrozenPackage.from_repo(ROOT)
    snapshot = snapshot_directory(os.environ["HF_HOME"])
    snapshot_hashes = verify_snapshot(snapshot, manifest["model"]["snapshot_sha256"])
    determinism = set_deterministic()
    tokenizer = load_tokenizer(snapshot)
    model = load_model(snapshot, load_yaml(ROOT / manifest["model"]["model_config"]))
    extraction = guard_input(_shared_root() / manifest["inputs"]["extraction_file"], [_shared_root() / "data"])
    result = run_technical_validation(model, tokenizer, package, load_extraction_prompts(package, extraction))
    payload = {"kind": args.name, "execution_commit": record["execution_commit"], "identity": identity,
               "snapshot_sha256": snapshot_hashes, "determinism": determinism, "result": result}
    atomic_write_json(_root(True) / "gpu" / f"{args.name}.json", payload, write_once=True)
    print(f"technical validation {args.name}: written")
    return 0


def cmd_techval_cpu(args) -> int:
    """CPU technical validation in the execution environment."""
    from slgeo.cts_stage0.atomic import atomic_write_json
    from slgeo.cts_stage0.checks import render_identity_check, retokenization_check
    from slgeo.cts_stage0.guards import guard_input
    from slgeo.cts_stage0.identity import assert_identity
    from slgeo.cts_stage0.modeling import load_tokenizer, snapshot_directory, verify_snapshot
    from slgeo.cts_stage0.package import FrozenPackage
    from slgeo.cts_stage0.plan import build_plan, load_choices
    from slgeo.cts_stage0.preflight import rng_golden_check
    from slgeo.cts_stage0.selftest import tiny_model_suite
    from slgeo.cts_stage0.synthetic import artifact_drill, synthetic_decision_suite

    manifest = _manifest()
    record = _verified_record(True)
    identity = assert_identity(manifest["execution"], ROOT, require_gpu=False)
    package = FrozenPackage.from_repo(ROOT)
    snapshot = snapshot_directory(os.environ["HF_HOME"])
    snapshot_hashes = verify_snapshot(snapshot, manifest["model"]["snapshot_sha256"])
    tokenizer = load_tokenizer(snapshot)
    extraction = guard_input(_shared_root() / manifest["inputs"]["extraction_file"], [_shared_root() / "data"])
    records = []
    with extraction.open(encoding="utf-8") as handle:
        for line in handle:
            if len(records) == 1024:
                break
            row = json.loads(line)
            records.append({"prompt": row["prompt"], "system_prompt": row["system_prompt"]})
    plan = build_plan(package, load_choices(ROOT, manifest), manifest, conditions_per_shard=int(manifest["plan"]["conditions_per_shard"]))
    result = {
        "retokenization": retokenization_check(tokenizer, package),
        "render_identity": render_identity_check(tokenizer, package, records),
        "tiny_model_suite": tiny_model_suite(),
        "rng_golden": rng_golden_check(),
        "synthetic_decision_suite": synthetic_decision_suite(),
        "artifact_drill": artifact_drill(_root(True) / "drill"),
        "plan_counts": {"pass": True, "conditions": plan["n_conditions"], "gating": plan["n_gating_conditions"],
                        "steered": sum(c["kind"] == "steer" for c in plan["conditions"]),
                        "steered_gating": sum(c["kind"] == "steer" and c["gating"] for c in plan["conditions"]),
                        "persona": sum(c["kind"] == "persona" for c in plan["conditions"]),
                        "shards": len(plan["shards"])},
    }
    payload = {"kind": "cpu", "execution_commit": record["execution_commit"], "identity": identity,
               "snapshot_sha256": snapshot_hashes, "result": result, "pass": all(v.get("pass") for v in result.values())}
    atomic_write_json(_root(True) / "cpu" / "techval_cpu.json", payload, write_once=True)
    print(f"technical validation cpu: {'PASS' if payload['pass'] else 'FAIL'}")
    return 0 if payload["pass"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    record = sub.add_parser("submit-record")
    record.add_argument("--technical-validation", action="store_true")
    record.set_defaults(func=cmd_submit_record)
    sub.add_parser("plan").set_defaults(func=cmd_plan)
    run = sub.add_parser("run")
    run.add_argument("--shard", required=True)
    run.set_defaults(func=cmd_run)
    techval = sub.add_parser("techval")
    techval.add_argument("--name", required=True)
    techval.set_defaults(func=cmd_techval)
    sub.add_parser("techval-cpu").set_defaults(func=cmd_techval_cpu)
    args = parser.parse_args()

    def terminate(_signum, _frame):
        raise SystemExit(75)

    signal.signal(signal.SIGTERM, terminate)
    try:
        return int(args.func(args) or 0)
    except Exception as exc:  # identity refusals map to 86; everything else propagates as 1
        if type(exc).__name__ in {"IdentityError", "FrozenPackageError", "GuardError", "ProvenanceError"}:
            print(f"refusal: {type(exc).__name__}: {exc}", file=sys.stderr)
            return IDENTITY_EXIT_CODE
        raise


if __name__ == "__main__":
    sys.exit(main())

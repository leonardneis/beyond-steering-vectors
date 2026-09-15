"""Validate the C18 manifest and emit technical/scientific command plans."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.analysis.c18_manifest import validate_manifest_contract, validate_public_inputs  # noqa: E402
from slgeo.analysis.teacher_coordinate_interchange import atomic_json  # noqa: E402
from slgeo.io import load_yaml  # noqa: E402


def command_plan(manifest: dict) -> dict:
    root = manifest["output"]["root"]
    manifest_path = "configs/validation/cat_bidirectional_teacher_coordinate_interchange_v1.yaml"
    return {
        "technical": [
            ["scripts/validate_bidirectional_teacher_coordinate_interchange.py", "--manifest", manifest_path,
             "--output", f"{root}/technical/validation.json"],
            ["scripts/audit_c18_technical_validation.py", "--manifest", manifest_path,
             "--validation", f"{root}/technical/validation.json",
             "--output", f"{root}/technical/audit.json"],
        ],
        "scientific": [
            ["scripts/run_bidirectional_teacher_coordinate_interchange.py", "--manifest", manifest_path,
             "--condition", condition, "--authorization", "condor/runtime/c18_scientific_authorization.json",
             "--technical-audit", f"{root}/technical/audit.json",
             "--output", f"{root}/sealed/raw/{condition}.npz.sealed"]
            for condition in ("subliminal", "neutral")
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/validation/cat_bidirectional_teacher_coordinate_interchange_v1.yaml")
    parser.add_argument("--mode", choices=("technical", "scientific"), default="technical")
    parser.add_argument("--emit-plan")
    parser.add_argument("--require-runtime-inputs", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    manifest = load_yaml(repo_path(args.manifest))
    validate_manifest_contract(manifest)
    inputs = validate_public_inputs(manifest, repo_path("."), require_runtime_inputs=args.require_runtime_inputs)
    plan = {"schema_version": 1, "experiment_id": manifest["experiment_id"], "inputs": inputs, "commands": command_plan(manifest)}
    if args.emit_plan:
        atomic_json(repo_path(args.emit_plan), plan, refuse_overwrite=False)
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if args.mode == "scientific" and not manifest.get("scientific_execution_authorized", False):
        raise RuntimeError("STOP: scientific C18 execution is not authorized")
    for command in plan["commands"][args.mode]:
        subprocess.run([sys.executable, *command], check=True, cwd=repo_path("."))


if __name__ == "__main__":
    main()

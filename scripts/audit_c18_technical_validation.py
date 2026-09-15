"""Independently fail closed on the outcome-blind C18 validation bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.analysis.teacher_coordinate_interchange import (  # noqa: E402
    atomic_json, sha256_file, validate_technical_report,
)
from slgeo.io import load_yaml  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/validation/cat_bidirectional_teacher_coordinate_interchange_v1.yaml")
    parser.add_argument("--validation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = repo_path(args.manifest)
    manifest = load_yaml(manifest_path)
    validation_path = repo_path(args.validation)
    sidecar_path = validation_path.with_suffix(validation_path.suffix + ".provenance.json")
    sums_path = validation_path.parent / "SHA256SUMS"
    for path in (validation_path, sidecar_path, sums_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    report = json.loads(validation_path.read_text(encoding="utf-8"))
    provenance = json.loads(sidecar_path.read_text(encoding="utf-8"))
    validate_technical_report(report)
    if report.get("status") != "PASS" or report.get("outcome_guard") != {
        "scientific_forwards": 0, "cross_cells_persisted": 0, "estimands_computed": 0
    }:
        raise RuntimeError("technical validation status/outcome guard failed")
    if report.get("manifest_sha256") != sha256_file(manifest_path):
        raise RuntimeError("technical validation used a different manifest")
    if provenance.get("artifact_sha256") != sha256_file(validation_path):
        raise RuntimeError("technical validation provenance digest failed")
    expected = {
        validation_path.name: sha256_file(validation_path),
        sidecar_path.name: sha256_file(sidecar_path),
    }
    observed = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        observed[name] = digest
    if observed != expected:
        raise RuntimeError("technical SHA256SUMS failed")
    if report["execution_identity"] != provenance["execution_identity"]:
        raise RuntimeError("technical execution identity differs across records")
    audit = {
        "schema_version": 1, "experiment_id": manifest["experiment_id"], "status": "PASS",
        "validation_sha256": expected[validation_path.name],
        "provenance_sha256": expected[sidecar_path.name],
        "manifest_sha256": sha256_file(manifest_path), "outcome_blind": True,
    }
    atomic_json(repo_path(args.output), audit)


if __name__ == "__main__":
    main()

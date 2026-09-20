"""Fail-closed independent audit of released C18 raw cells and aggregate."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

from _bootstrap import bootstrap, repo_path

bootstrap()

import numpy as np  # noqa: E402

from slgeo.analysis.c18_manifest import (  # noqa: E402
    apply_storage_overrides, expected_raw_ids, validate_manifest_contract,
    validate_public_inputs,
)
from slgeo.analysis.c18_statistics import aggregate_y_rows  # noqa: E402
from slgeo.analysis.teacher_coordinate_interchange import atomic_json, sha256_file, unseal_bytes  # noqa: E402
from slgeo.io import load_yaml  # noqa: E402
from aggregate_bidirectional_teacher_coordinate_interchange import (  # noqa: E402
    aggregate_suffix_factorial, raw_four_cell_table, relabel_cells, specificity, verify_suffix_identities,
)


def decrypt(path: Path, key: bytes) -> dict:
    with np.load(io.BytesIO(unseal_bytes(path.read_bytes(), key)), allow_pickle=False) as data:
        return json.loads(str(data["payload_json"].item()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/validation/cat_bidirectional_teacher_coordinate_interchange_v1.yaml")
    parser.add_argument("--release-token", required=True)
    parser.add_argument("--subliminal", required=True)
    parser.add_argument("--neutral", required=True)
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = apply_storage_overrides(load_yaml(repo_path(args.manifest)))
    validate_manifest_contract(manifest)
    frozen = validate_public_inputs(manifest, repo_path("."), require_runtime_inputs=True)
    release = json.loads(repo_path(args.release_token).read_text(encoding="utf-8"))
    if release.get("outcome_release_authorized") is not True:
        raise RuntimeError("audit release is not authorized")
    key = os.environ.get("SLGEO_C18_SEAL_KEY", "").encode("ascii")
    if not key:
        raise RuntimeError("runtime-only sealing key is absent")
    paths = [repo_path(args.subliminal), repo_path(args.neutral)]
    payloads = [decrypt(path, key) for path in paths]
    rows = [row for payload in payloads for row in payload["Y"]]
    expected = expected_raw_ids(manifest, repo_path("."))["Y"]
    observed = {f"{r['condition']}|{r['set_id']}|{r['prompt_id']}|{r['cell']}" for r in rows}
    if observed != expected or len(observed) != len(rows):
        raise ValueError("audit raw-cell inventory failed")
    recomputed = aggregate_y_rows(rows, bootstrap_draws=20000, bootstrap_seed=20260804)
    specificity_result = specificity(recomputed, payloads)
    if recomputed["classification"] == "insufficient_precision" and any(
        item["interval99"][0] > -manifest["design"]["epsilon"]
        and item["interval99"][1] < manifest["design"]["epsilon"]
        for item in specificity_result["controls"]
    ):
        recomputed["classification"] = "heterogeneous_result"
        recomputed["heterogeneous"] = True
    recomputed["specificity"] = specificity_result
    recomputed["full_state_factorial"] = aggregate_y_rows(
        relabel_cells([row for payload in payloads for row in payload["Z"]], "Z"),
        bootstrap_draws=20000, bootstrap_seed=20260804,
    )
    recomputed["suffix_integrity"] = verify_suffix_identities(payloads)
    recomputed["hybrid_suffix_factorial"] = aggregate_suffix_factorial(payloads)
    recomputed["raw_four_cell_table"] = raw_four_cell_table(rows)
    recomputed["numerical_diagnostics"] = [item for payload in payloads for item in payload["numerical_diagnostics"]]
    stored = json.loads(repo_path(args.aggregate).read_text(encoding="utf-8"))
    for field in (
        "means", "intervals", "cancellation_flags", "classification", "specificity",
        "full_state_factorial", "suffix_integrity", "hybrid_suffix_factorial",
        "raw_four_cell_table", "numerical_diagnostics",
    ):
        if json.dumps(stored.get(field), sort_keys=True) != json.dumps(recomputed.get(field), sort_keys=True):
            raise RuntimeError(f"aggregate audit mismatch: {field}")
    report = {
        "schema_version": 1, "status": "PASS", "experiment_id": manifest["experiment_id"],
        "raw_y_count": len(rows), "frozen_inputs": frozen,
        "checksums": {str(path): sha256_file(path) for path in [*paths, repo_path(args.aggregate)]},
    }
    atomic_json(repo_path(args.output), report)


if __name__ == "__main__":
    main()

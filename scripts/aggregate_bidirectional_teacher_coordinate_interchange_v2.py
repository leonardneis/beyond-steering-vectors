"""Aggregate complete released C18-v2 sealed raw cells only after release."""

from __future__ import annotations

import argparse
import io
import json
import os

from _bootstrap import bootstrap, repo_path

bootstrap()

import numpy as np  # noqa: E402

from slgeo.analysis.c18_v2_manifest import (  # noqa: E402
    apply_storage_overrides, expected_raw_ids, sha256_file, validate_manifest_contract,
)
from slgeo.analysis.c18_v2_authorization import (  # noqa: E402
    load_and_validate_scientific_authorization,
)
from slgeo.analysis.c18_v2_statistics import (  # noqa: E402
    aggregate_factorial, aggregate_w, aggregate_y, classify, percentile_interval,
    stratified_indices, suffix_conflict,
)
from slgeo.analysis.teacher_coordinate_interchange import atomic_json, unseal_bytes  # noqa: E402
from slgeo.io import load_yaml  # noqa: E402


def release_key(path: str, manifest_sha256: str, experiment_id: str) -> bytes:
    record = json.loads(repo_path(path).read_text(encoding="utf-8"))
    if record != {"experiment_id": experiment_id, "manifest_sha256": manifest_sha256,
                   "outcome_release_authorized": True}:
        raise RuntimeError("STOP: exact C18-v2 outcome release is not authorized")
    key = os.environ.get("SLGEO_C18_V2_SEAL_KEY", "")
    if not key:
        raise RuntimeError("STOP: C18-v2 sealing key is absent")
    return key.encode("ascii")


def load_payload(path: str, key: bytes) -> dict:
    plaintext = unseal_bytes(repo_path(path).read_bytes(), key)
    with np.load(io.BytesIO(plaintext), allow_pickle=False) as data:
        return json.loads(str(data["payload_json"].item()))


def row_ids(rows: list[dict], category: str) -> set[str]:
    if category == "orthogonal":
        return {f"{r['condition']}|{r['set_id']}|{r['prompt_id']}|O{r['dose_family']}|{r['cell']}" for r in rows}
    return {f"{r['condition']}|{r['set_id']}|{r['prompt_id']}|{r['cell']}" for r in rows}


def verify_complete(payloads: list[dict], expected: dict[str, set[str]]) -> dict[str, int]:
    counts = {}
    for category in ("Y", "Z", "W", "orthogonal"):
        rows = [row for payload in payloads for row in payload[category]]
        observed = row_ids(rows, category)
        if observed != expected[category] or len(rows) != len(observed):
            raise ValueError(f"C18-v2 {category} inventory incomplete, duplicate, or unexpected")
        counts[category] = len(rows)
    return counts


def verify_cross_factorial_identities(payloads: list[dict]) -> dict[str, object]:
    y = {(r["condition"], r["set_id"], r["prompt_id"], r["cell"]): float(r["margin"])
         for payload in payloads for r in payload["Y"]}
    z = {(r["condition"], r["set_id"], r["prompt_id"], r["cell"]): float(r["margin"])
         for payload in payloads for r in payload["Z"]}
    w = {(r["condition"], r["set_id"], r["prompt_id"], r["cell"]): float(r["margin"])
         for payload in payloads for r in payload["W"]}
    maximum = 0.0
    for condition, set_id, prompt_id, cell in y:
        a, b = int(cell[1]), int(cell[2])
        value = y[(condition, set_id, prompt_id, cell)]
        w_value = w[(condition, set_id, prompt_id, f"W{a}{b}{a}")]
        maximum = max(maximum, abs(value - w_value))
        if value != w_value:
            raise ArithmeticError("C18-v2 Y=W_ab,a raw identity failed")
        if (a, b) in ((0, 0), (1, 1)):
            if value != z[(condition, set_id, prompt_id, f"Z{a}{b}")]:
                raise ArithmeticError("C18-v2 natural Y/Z diagonal raw identity failed")
    return {"status": "PASS", "maximum_absolute_error": maximum}


def specificity(teacher: dict, payloads: list[dict]) -> dict[str, object]:
    natural = [dict(row) for payload in payloads for row in payload["Y"] if row["cell"] in ("Y00", "Y11")]
    indices = stratified_indices(teacher["families"])
    t0 = np.asarray(teacher["quantities"]["B0"]["per_prompt"], dtype=np.float64)
    t1 = np.asarray(teacher["quantities"]["B1"]["per_prompt"], dtype=np.float64)
    tdraw = np.maximum(np.abs(np.mean(t0[indices], axis=1)), np.abs(np.mean(t1[indices], axis=1)))
    controls = []
    for family in range(5):
        cross = [dict(row) for payload in payloads for row in payload["orthogonal"]
                 if int(row["dose_family"]) == family]
        result = aggregate_y(natural + cross)
        c0 = np.asarray(result["quantities"]["B0"]["per_prompt"], dtype=np.float64)
        c1 = np.asarray(result["quantities"]["B1"]["per_prompt"], dtype=np.float64)
        cdraw = np.maximum(np.abs(np.mean(c0[indices], axis=1)), np.abs(np.mean(c1[indices], axis=1)))
        delta = cdraw - tdraw
        controls.append({
            "dose_family": family,
            "D": float(max(abs(result["quantities"]["B0"]["G"]), abs(result["quantities"]["B1"]["G"]))
                       - max(abs(teacher["quantities"]["B0"]["G"]), abs(teacher["quantities"]["B1"]["G"]))),
            "interval99": percentile_interval(delta, 0.99),
            "G_B0": result["quantities"]["B0"]["G"],
            "G_B1": result["quantities"]["B1"]["G"],
        })
    return {"status": "single_secondary_specificity_family", "controls": controls,
            "teacher_specificity_modifier_pass": all(item["interval99"][0] > 0 for item in controls)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/validation/cat_bidirectional_teacher_coordinate_interchange_v2.yaml")
    parser.add_argument("--release-token", required=True)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--technical-directory", required=True)
    parser.add_argument("--execution-git-commit", required=True)
    parser.add_argument("--subliminal", required=True)
    parser.add_argument("--neutral", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = repo_path(args.manifest)
    manifest = apply_storage_overrides(load_yaml(manifest_path))
    validate_manifest_contract(manifest)
    digest = sha256_file(manifest_path)
    load_and_validate_scientific_authorization(
        args.authorization, manifest, manifest_path, root=repo_path("."),
        execution_commit=args.execution_git_commit,
        technical_directory=args.technical_directory,
    )
    key = release_key(args.release_token, digest, manifest["experiment_id"])
    payloads = [load_payload(args.subliminal, key), load_payload(args.neutral, key)]
    for payload in payloads:
        if payload.get("experiment_id") != manifest["experiment_id"] or payload["provenance"]["manifest_sha256"] != digest:
            raise RuntimeError("released C18-v2 shard identity differs")
    counts = verify_complete(payloads, expected_raw_ids(manifest, repo_path(".")))
    identity = verify_cross_factorial_identities(payloads)
    y_result = aggregate_y([row for payload in payloads for row in payload["Y"]])
    z_result = aggregate_factorial([row for payload in payloads for row in payload["Z"]], "Z")
    w_result = aggregate_w([row for payload in payloads for row in payload["W"]])
    conflict, reasons = suffix_conflict(z_result, w_result)
    result = {
        "schema_version": 2, "experiment_id": manifest["experiment_id"],
        "manifest_sha256": digest, "inventory_counts": counts,
        "factorial_identities": identity, "Y": y_result, "Z": z_result, "W": w_result,
        "suffix_conflict": conflict, "suffix_conflict_reasons": reasons,
        "specificity": specificity(y_result, payloads),
        "classification": classify(integrity=True,
                                   reconstruction=bool(y_result["natural_effect_reconstruction"]),
                                   y=y_result, suffix_conflict_value=conflict),
        "numerical_diagnostics": [item for payload in payloads for item in payload["numerical_diagnostics"]],
    }
    atomic_json(repo_path(args.output), result)


if __name__ == "__main__":
    main()

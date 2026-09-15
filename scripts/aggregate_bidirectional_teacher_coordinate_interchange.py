"""Independently aggregate released C18 cells from authenticated sealed shards."""

from __future__ import annotations

import argparse
import io
import json
import os

from _bootstrap import bootstrap, repo_path

bootstrap()

import numpy as np  # noqa: E402

from slgeo.analysis.c18_manifest import expected_raw_ids  # noqa: E402
from slgeo.analysis.c18_statistics import aggregate_y_rows  # noqa: E402
from slgeo.analysis.teacher_coordinate_interchange import (  # noqa: E402
    atomic_json, percentile_interval, stratified_bootstrap_indices, unseal_bytes,
)
from slgeo.io import load_yaml  # noqa: E402


def released_key(token_path: str) -> bytes:
    record = json.loads(repo_path(token_path).read_text(encoding="utf-8"))
    if record.get("outcome_release_authorized") is not True:
        raise RuntimeError("outcome release is not authorized")
    key = os.environ.get("SLGEO_C18_SEAL_KEY", "")
    if not key:
        raise RuntimeError("runtime-only sealing key is absent")
    return key.encode("ascii")


def load_payload(path: str, key: bytes) -> dict:
    plaintext = unseal_bytes(repo_path(path).read_bytes(), key)
    with np.load(io.BytesIO(plaintext), allow_pickle=False) as data:
        return json.loads(str(data["payload_json"].item()))


def relabel_cells(rows: list[dict], prefix: str) -> list[dict]:
    output = []
    for row in rows:
        item = dict(row)
        item["cell"] = "Y" + str(item["cell"])[len(prefix):]
        output.append(item)
    return output


def specificity(teacher: dict, payloads: list[dict]) -> dict:
    families = []
    prompt_order = []
    for row in payloads[0]["Y"]:
        if row["cell"] == "Y00" and row["set_id"] == "top" and row["condition"] == "subliminal":
            prompt_order.append(row["prompt_id"]); families.append(row["family"])
    indices = stratified_bootstrap_indices(families, draws=20000, seed=20260804)
    teacher_b0 = np.asarray(teacher["per_prompt"]["B0"])
    teacher_b1 = np.asarray(teacher["per_prompt"]["B1"])
    controls = []
    natural = [dict(row) for payload in payloads for row in payload["Y"] if row["cell"] in ("Y00", "Y11")]
    for family in range(5):
        cross = [dict(row) for payload in payloads for row in payload["orthogonal"] if row["dose_family"] == family]
        result = aggregate_y_rows(natural + cross, bootstrap_draws=20000, bootstrap_seed=20260804)
        b0, b1 = np.asarray(result["per_prompt"]["B0"]), np.asarray(result["per_prompt"]["B1"])
        draw = np.maximum(np.abs(np.mean(b0[indices], axis=1)), np.abs(np.mean(b1[indices], axis=1))) - np.maximum(
            np.abs(np.mean(teacher_b0[indices], axis=1)), np.abs(np.mean(teacher_b1[indices], axis=1))
        )
        controls.append({
            "family": family, "D": float(max(abs(np.mean(b0)), abs(np.mean(b1))) - max(abs(np.mean(teacher_b0)), abs(np.mean(teacher_b1)))),
            "interval99": percentile_interval(draw, 0.99), "G_B0": result["means"]["B0"], "G_B1": result["means"]["B1"],
        })
    return {"controls": controls, "gate_pass": all(item["interval99"][0] > 0 for item in controls)}


def verify_suffix_identities(payloads: list[dict], atol: float = 1e-12) -> dict:
    y = {(r["condition"], r["set_id"], r["prompt_id"], r["cell"]): float(r["margin"]) for p in payloads for r in p["Y"]}
    w = {(r["condition"], r["set_id"], r["prompt_id"], r["cell"]): float(r["margin"]) for p in payloads for r in p["W"]}
    maximum = 0.0
    for (condition, set_id, prompt, cell), value in y.items():
        a, b = int(cell[1]), int(cell[2])
        maximum = max(maximum, abs(value - w[(condition, set_id, prompt, f"W{a}{b}{a}")]))
    if maximum > atol:
        raise ArithmeticError(f"Y/W suffix identity failed: {maximum}")
    return {"Y_equals_W_aba": "PASS", "max_abs_error": maximum}


def aggregate_scalar(values: dict[tuple[str, str, str], float], families: dict[str, str]) -> dict:
    prompts = list(dict.fromkeys(prompt for _condition, _set_id, prompt in values))
    norms = list(dict.fromkeys(set_id for _condition, set_id, _prompt in values if set_id != "top"))
    per_prompt = []
    for prompt in prompts:
        components = {}
        for condition in ("subliminal", "neutral"):
            components[condition] = values[(condition, "top", prompt)] - np.mean([
                values[(condition, set_id, prompt)] for set_id in norms
            ])
        per_prompt.append(components["subliminal"] - components["neutral"])
    per_prompt = np.asarray(per_prompt, dtype=np.float64)
    indices = stratified_bootstrap_indices([families[prompt] for prompt in prompts], draws=20000, seed=20260804)
    draws = np.mean(per_prompt[indices], axis=1)
    return {
        "G": float(np.mean(per_prompt)), "interval90": percentile_interval(draws, 0.90),
        "interval95": percentile_interval(draws, 0.95), "interval99": percentile_interval(draws, 0.99),
        "per_prompt": per_prompt.tolist(),
    }


def aggregate_suffix_factorial(payloads: list[dict]) -> dict:
    rows = [row for payload in payloads for row in payload["W"]]
    lookup = {(r["condition"], r["set_id"], r["prompt_id"], r["cell"]): float(r["margin"]) for r in rows}
    locations = list(dict.fromkeys((r["condition"], r["set_id"], r["prompt_id"]) for r in rows))
    families = {r["prompt_id"]: r["family"] for r in rows}
    output = {}
    for b in (0, 1):
        metrics = {name: {} for name in ("total", "upstream_suffix0", "suffix_at_state1", "upstream_suffix1", "suffix_at_state0", "interaction")}
        for location in locations:
            c, s, p = location
            get = lambda a, r: lookup[(c, s, p, f"W{a}{b}{r}")]
            metrics["total"][location] = get(1, 1) - get(0, 0)
            metrics["upstream_suffix0"][location] = get(1, 0) - get(0, 0)
            metrics["suffix_at_state1"][location] = get(1, 1) - get(1, 0)
            metrics["upstream_suffix1"][location] = get(1, 1) - get(0, 1)
            metrics["suffix_at_state0"][location] = get(0, 1) - get(0, 0)
            metrics["interaction"][location] = (get(1, 1) - get(1, 0)) - (get(0, 1) - get(0, 0))
        output[f"donor_{b}"] = {name: aggregate_scalar(values, families) for name, values in metrics.items()}
    return output


def raw_four_cell_table(rows: list[dict]) -> list[dict]:
    lookup = {(r["condition"], r["set_id"], r["prompt_id"], r["cell"]): r for r in rows}
    locations = sorted({key[:3] for key in lookup})
    output = []
    for condition, set_id, prompt_id in locations:
        records = [lookup[(condition, set_id, prompt_id, f"Y{a}{b}")] for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))]
        y00, y01, y10, y11 = [float(record["margin"]) for record in records]
        values = {"E": y11-y00, "L": y01-y00, "H": y11-y10, "B0": y10-y00, "B1": y11-y01, "I": y11-y10-y01+y00}
        algebra_error = max(abs(values["E"]-values["L"]-values["B1"]), abs(values["E"]-values["H"]-values["B0"]), abs(values["I"]-values["H"]+values["L"]), abs(values["I"]-values["B1"]+values["B0"]))
        output.append({
            "condition": condition, "set_id": set_id, "prompt_id": prompt_id,
            "family": records[0]["family"], "Y00": y00, "Y01": y01, "Y10": y10, "Y11": y11,
            **values, "algebra_max_abs_error": algebra_error,
            "candidate_diagnostics": {record["cell"]: {
                "logits": record["candidate_logits"], "probabilities": record["candidate_probabilities"],
                "entropy": record["candidate_entropy"],
            } for record in records},
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/validation/cat_bidirectional_teacher_coordinate_interchange_v1.yaml")
    parser.add_argument("--release-token", required=True)
    parser.add_argument("--subliminal", required=True)
    parser.add_argument("--neutral", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = load_yaml(repo_path(args.manifest))
    key = released_key(args.release_token)
    payloads = [load_payload(args.subliminal, key), load_payload(args.neutral, key)]
    rows = [row for payload in payloads for row in payload["Y"]]
    expected = expected_raw_ids(manifest, repo_path("."))["Y"]
    observed = {
        f"{row['condition']}|{row['set_id']}|{row['prompt_id']}|{row['cell']}" for row in rows
    }
    if observed != expected or len(rows) != len(observed):
        raise ValueError("released Y inventory is incomplete, duplicated, or unexpected")
    result = aggregate_y_rows(rows, bootstrap_draws=20000, bootstrap_seed=20260804)
    z_rows = relabel_cells([row for payload in payloads for row in payload["Z"]], "Z")
    z_result = aggregate_y_rows(z_rows, bootstrap_draws=20000, bootstrap_seed=20260804)
    specificity_result = specificity(result, payloads)
    if result["classification"] == "insufficient_precision" and any(
        item["interval99"][0] > -manifest["design"]["epsilon"]
        and item["interval99"][1] < manifest["design"]["epsilon"]
        for item in specificity_result["controls"]
    ):
        result["classification"] = "heterogeneous_result"
        result["heterogeneous"] = True
    result.update({
        "schema_version": 1, "experiment_id": manifest["experiment_id"], "raw_y_count": len(rows),
        "full_state_factorial": z_result, "specificity": specificity_result,
        "suffix_integrity": verify_suffix_identities(payloads),
        "hybrid_suffix_factorial": aggregate_suffix_factorial(payloads),
        "raw_four_cell_table": raw_four_cell_table(rows),
        "numerical_diagnostics": [item for payload in payloads for item in payload["numerical_diagnostics"]],
    })
    atomic_json(repo_path(args.output), result)


if __name__ == "__main__":
    main()

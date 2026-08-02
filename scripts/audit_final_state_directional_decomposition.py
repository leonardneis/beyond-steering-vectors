"""Fail-closed scientific audit for Final-State Directional Decomposition v1."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from _bootstrap import bootstrap, repo_path

bootstrap()

from aggregate_final_state_directional_decomposition import load_artifact  # noqa: E402
from notify import notify  # noqa: E402
from run_confirmatory_manifest import sha256, tree_digest, write_json_atomic  # noqa: E402
from run_final_state_directional_decomposition_manifest import validate_inputs  # noqa: E402
from slgeo.io import load_yaml  # noqa: E402
from slgeo.analysis.directional_decomposition import decision_gate  # noqa: E402


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _validate_provenance(
    path: Path, artifact: Path, *, manifest_sha: str, experiment_id: str,
    expected_commit: str, stage: str, command_index: int,
) -> dict:
    record = _read_json(path)
    expected = {
        "manifest_sha256": manifest_sha,
        "experiment_id": experiment_id,
        "git_commit": expected_commit,
        "stage": stage,
        "command_index": command_index,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"Provenance mismatch for {artifact}: {key}={record.get(key)!r}, expected {value!r}")
    if record.get("git_dirty") is not False:
        raise ValueError(f"Execution worktree was dirty for {artifact}")
    def research_suffix(value: str) -> str:
        normalized = value.replace("\\", "/")
        marker = "results/research/"
        if marker not in normalized:
            raise ValueError(f"Provenance path is outside the research namespace: {value}")
        return marker + normalized.split(marker, maxsplit=1)[1]

    if research_suffix(str(record.get("output"))) != research_suffix(str(artifact)):
        raise ValueError(f"Provenance output path mismatch for {artifact}")
    if record.get("output_sha256") != tree_digest(artifact):
        raise ValueError(f"Provenance checksum mismatch for {artifact}")
    return record


def audit(manifest_path: Path, expected_commit: str) -> dict:
    manifest = load_yaml(manifest_path)
    validate_inputs(manifest, require_adapters=False)
    root = repo_path(manifest["output_root"])
    artifacts = {
        "subliminal": root / "states/subliminal.npz",
        "neutral": root / "states/neutral.npz",
        "decomposition": root / "aggregate/decomposition.json",
        "prompt_estimands": root / "aggregate/prompt_estimands.csv",
    }
    provenance = {key: path.with_suffix(path.suffix + ".provenance.json") for key, path in artifacts.items()}
    markers = {
        "subliminal": root / "orchestration/state_runs_00.complete.json",
        "neutral": root / "orchestration/state_runs_01.complete.json",
        "decomposition": root / "orchestration/aggregate_00.complete.json",
    }
    required = [*artifacts.values(), *provenance.values(), *markers.values()]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise FileNotFoundError(f"Missing or empty required artifacts: {missing}")
    forbidden = list(root.rglob("*.failed.json")) + list(root.rglob("*.running.json")) + list(root.rglob("*.tmp")) + list(root.rglob("*.incoming"))
    if forbidden:
        raise RuntimeError(f"Incomplete/temporary markers exist: {[str(path) for path in forbidden]}")
    actual = {
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).parts[0] not in {"runtime", "audit"}
    }
    if actual != set(required):
        extra = sorted(str(path) for path in actual - set(required))
        absent = sorted(str(path) for path in set(required) - actual)
        raise RuntimeError(f"Artifact inventory differs; extra={extra}, missing={absent}")

    manifest_sha = sha256(manifest_path)
    records = {}
    for key, stage, index in (
        ("subliminal", "state_runs", 0), ("neutral", "state_runs", 1),
        ("decomposition", "aggregate", 0), ("prompt_estimands", "aggregate", 0),
    ):
        records[key] = _validate_provenance(
            provenance[key], artifacts[key], manifest_sha=manifest_sha,
            experiment_id=manifest["experiment_id"], expected_commit=expected_commit,
            stage=stage, command_index=index,
        )
    for key, marker_path in markers.items():
        marker = _read_json(marker_path)
        if marker.get("git_commit") != expected_commit or marker.get("git_dirty") is not False:
            raise ValueError(f"Invalid completion marker execution state: {marker_path}")
        artifact_key = key
        if marker.get("output_sha256") != tree_digest(artifacts[artifact_key]):
            raise ValueError(f"Completion marker checksum mismatch: {marker_path}")

    state_artifacts = {key: load_artifact(artifacts[key]) for key in ("subliminal", "neutral")}
    study = manifest["study"]
    for condition, payload in state_artifacts.items():
        metadata = payload["metadata"]
        expected_metadata = {
            "condition": condition,
            "prompt_file_sha256": manifest["prompt_file_sha256"],
            "selection_plan_sha256": manifest["selection_plan_sha256"],
            "teacher_vector_sha256": manifest["teacher_vector_sha256"],
            "adapter_sha256": manifest["adapter_sha256"][condition],
            "target_animal": study["target_animal"],
            "competitor_animal": study["competitor_animal"],
        }
        for key, value in expected_metadata.items():
            if metadata.get(key) != value:
                raise ValueError(f"{condition} metadata mismatch: {key}")
        if float(metadata["state_head_max_abs_error"]) > float(study["state_head_atol"]):
            raise ValueError(f"{condition} state-to-head reconstruction tolerance failed")
        if payload["full_state"].shape[0] != int(study["prompt_count"]):
            raise ValueError(f"{condition} prompt dimension differs")
        if payload["ablated_state"].shape[:2] != (int(study["control_draws"]) + 1, int(study["prompt_count"])):
            raise ValueError(f"{condition} intervention dimensions differ")
        numeric_arrays = (
            value for key, value in payload.items()
            if key != "metadata" and np.issubdtype(value.dtype, np.number)
        )
        if not all(np.isfinite(value).all() for value in numeric_arrays):
            raise ValueError(f"{condition} contains non-finite arrays")

    result = _read_json(artifacts["decomposition"])
    expected_input_hashes = {
        "subliminal_sha256": sha256(artifacts["subliminal"]),
        "neutral_sha256": sha256(artifacts["neutral"]),
        "parent_aggregate_sha256": manifest["parent"]["aggregate_sha256"],
    }
    for key, value in expected_input_hashes.items():
        if result.get("inputs", {}).get(key) != value:
            raise ValueError(f"Aggregate input checksum mismatch: {key}")
    contract = result.get("contract", {})
    contract_expected = {
        "parent_margin": float(study["parent_margin_contrast"]),
        "equivalence_margin": float(study["equivalence_margin"]),
        "parent_reconstruction_atol": float(study["parent_reconstruction_atol"]),
        "decomposition_atol": float(study["decomposition_atol"]),
        "bootstrap_samples": int(study["bootstrap_samples"]),
        "bootstrap_seed": int(study["bootstrap_seed"]),
        "prompt_count": int(study["prompt_count"]),
        "control_draws": int(study["control_draws"]),
    }
    if contract != contract_expected:
        raise ValueError("Aggregate contract does not exactly match the manifest")
    numerical = result.get("numerical_audit", {})
    if float(numerical.get("max_abs_error", float("inf"))) > float(study["decomposition_atol"]):
        raise ValueError("Aggregate decomposition identity exceeds tolerance")
    if set(result.get("estimands", {})) != {"full", "teacher", "residual"}:
        raise ValueError("Aggregate estimand inventory differs")
    expected_gate = decision_gate(
        result["estimands"], parent_margin=contract["parent_margin"],
        equivalence_margin=contract["equivalence_margin"],
        parent_reconstruction_atol=contract["parent_reconstruction_atol"],
    )
    if result.get("decision_gate") != expected_gate:
        raise ValueError("Aggregate decision gate does not match the frozen hierarchy")
    lines = artifacts["prompt_estimands"].read_text(encoding="utf-8").splitlines()
    if len(lines) != int(study["prompt_count"]) + 1:
        raise ValueError("Prompt-estimand CSV row count differs")

    checksums = {str(path.relative_to(root)).replace("\\", "/"): sha256(path) for path in sorted(required)}
    return {
        "schema_version": 1,
        "audit": "final_state_directional_causal_decomposition_v1",
        "status": "PASS",
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "experiment_id": manifest["experiment_id"],
        "execution_git_commit": expected_commit,
        "required_artifact_count": len(required),
        "checksums": checksums,
        "integrity": {
            "completion_markers": "PASS",
            "provenance": "PASS",
            "frozen_inputs": "PASS",
            "state_head_reconstruction": "PASS",
            "directional_identities": "PASS",
            "finite_arrays": "PASS",
            "inventory": "PASS",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/validation/cat_final_state_directional_causal_decomposition_v1.yaml")
    parser.add_argument("--expected-git-commit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checksums", required=True)
    args = parser.parse_args()
    output, checksums_path = repo_path(args.output), repo_path(args.checksums)
    manifest_path = repo_path(args.manifest)
    manifest = load_yaml(manifest_path)
    root = repo_path(manifest["output_root"])
    runtime_metadata = root / "runtime/dag.json"
    dag = _read_json(runtime_metadata) if runtime_metadata.is_file() else {}
    dag_id = str(dag.get("dag_id") or os.getenv("HTCONDOR_DAG_ID") or "unknown")
    audit_started = time.time()

    def send(status: str) -> None:
        try:
            notify(
                study="Final-State Directional Causal Decomposition v1",
                event="AUDIT", status=status, dag_id=dag_id,
                git_commit=args.expected_git_commit,
                duration_seconds=time.time() - audit_started,
                result_path=str(root) if status == "SUCCESS" else None,
                metadata_output=root / "runtime/audit.json",
            )
        except Exception as exc:
            print(f"ntfy audit notification failed without changing audit status: {exc}")

    try:
        if output.exists() or checksums_path.exists():
            raise FileExistsError("Refusing to overwrite an existing scientific audit")
        result = audit(manifest_path, args.expected_git_commit)
        write_json_atomic(output, result)
        checksums_path.parent.mkdir(parents=True, exist_ok=True)
        checksums_path.write_text(
            "".join(f"{digest}  {path}\n" for path, digest in result["checksums"].items()),
            encoding="utf-8", newline="\n",
        )
    except BaseException:
        send("FAILED")
        raise
    send("SUCCESS")
    print(f"PASS: audited {result['required_artifact_count']} required artifacts")


if __name__ == "__main__":
    main()

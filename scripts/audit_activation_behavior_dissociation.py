"""Fail-closed integrity audit for Activation--Behavior Dissociation v1."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from _bootstrap import bootstrap, repo_path

bootstrap()

from run_confirmatory_manifest import sha256  # noqa: E402
from slgeo.io import load_yaml  # noqa: E402


STUDY_ID = "qwen7b_cat_activation_behavior_dissociation_v1"
METRICS = ("target_logprob", "target_probability", "target_vs_lion_margin", "target_choice")
COMPONENTS = ("paired", "subliminal", "neutral")


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _assert_prompt_records(records: list[dict], expected: list[dict], label: str) -> None:
    if records != expected:
        raise ValueError(f"Prompt records differ in {label}")


def _assert_prompt_array(summary: dict, expected_count: int, label: str) -> None:
    values = summary.get("per_prompt")
    if not isinstance(values, list) or len(values) != expected_count:
        raise ValueError(f"{label} has {len(values) if isinstance(values, list) else 'no'} prompt values")


def _audit_provenance(
    artifact: Path,
    *,
    manifest_sha: str,
    expected_commit: str,
    expected_stage: str,
    marker: Path,
) -> None:
    sidecar = artifact.with_suffix(artifact.suffix + ".provenance.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing provenance sidecar: {sidecar}")
    provenance = _read_json(sidecar)
    if provenance.get("experiment_id") != STUDY_ID:
        raise ValueError(f"Wrong experiment ID in {sidecar}")
    if provenance.get("manifest_sha256") != manifest_sha:
        raise ValueError(f"Manifest checksum mismatch in {sidecar}")
    if provenance.get("git_commit") != expected_commit:
        raise ValueError(f"Execution source was not the expected commit in {sidecar}")
    if provenance.get("git_dirty") not in (False, None):
        raise ValueError(f"Execution provenance records a dirty tree in {sidecar}")
    if provenance.get("stage") != expected_stage:
        raise ValueError(f"Wrong stage in {sidecar}")
    if provenance.get("output_sha256") != sha256(artifact):
        raise ValueError(f"Output checksum mismatch in {sidecar}")
    complete = _read_json(marker)
    if complete.get("output") != provenance.get("output"):
        raise ValueError(f"Completion marker and provenance output paths differ: {marker}")
    study_index = artifact.parts.index(STUDY_ID)
    expected_suffix = "/".join(artifact.parts[study_index:])
    if not str(complete.get("output", "")).replace("\\", "/").endswith(expected_suffix):
        raise ValueError(f"Completion marker points to a different output: {marker}")
    if complete.get("output_sha256") != provenance.get("output_sha256"):
        raise ValueError(f"Marker and sidecar checksums differ: {marker}")
    if complete.get("git_commit") != expected_commit or complete.get("manifest_sha256") != manifest_sha:
        raise ValueError(f"Completion marker source identity mismatch: {marker}")


def _audit_raw(
    path: Path,
    *,
    expected_prompts: list[dict],
    prompt_sha: str,
    selection_sha: str,
    expected_k: int,
    expected_draws: int,
) -> None:
    payload = _read_json(path)
    if payload.get("schema_version") != 2 or payload.get("analysis") != "lora_set_behavior":
        raise ValueError(f"Unexpected raw behavior schema: {path}")
    if payload.get("prompt_file_sha256") != prompt_sha:
        raise ValueError(f"Prompt checksum mismatch: {path}")
    if payload.get("selection_plan_sha256") != selection_sha:
        raise ValueError(f"Selection-plan checksum mismatch: {path}")
    if payload.get("num_samples") != len(expected_prompts):
        raise ValueError(f"Wrong prompt count: {path}")
    for name in ("base_evaluation", "full_evaluation"):
        evaluation = payload[name]
        _assert_prompt_records(evaluation["prompt_records"], expected_prompts, f"{path}:{name}")
        if len(evaluation["target_choice_per_prompt"]) != len(expected_prompts):
            raise ValueError(f"Wrong choice-row count: {path}:{name}")
        if len(evaluation["token_metrics"]["rows"]) != len(expected_prompts):
            raise ValueError(f"Wrong token-row count: {path}:{name}")
    rows = payload.get("interventions", [])
    expected_rows = 2 * (1 + 2 * expected_draws)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} interventions, got {len(rows)}: {path}")
    counts = Counter((row["mode"], row["set_name"]) for row in rows)
    for mode in ("necessity", "sufficiency"):
        expected = {"top_k": 1, "random_control": expected_draws, "norm_matched_control": expected_draws}
        for set_name, count in expected.items():
            if counts[mode, set_name] != count:
                raise ValueError(f"Wrong {mode}/{set_name} count in {path}")
    for row in rows:
        if row["k"] != expected_k or len(row["modules"]) != expected_k:
            raise ValueError(f"Wrong intervention size in {path}")
        _assert_prompt_records(row["prompt_records"], expected_prompts, f"{path}:intervention")
        if len(row["target_choice_per_prompt"]) != len(expected_prompts):
            raise ValueError(f"Wrong intervention choice count in {path}")
        if len(row["token_metrics"]["rows"]) != len(expected_prompts):
            raise ValueError(f"Wrong intervention token count in {path}")


def _audit_paired(
    path: Path,
    *,
    expected_prompts: list[dict],
    prompt_sha: str,
    selection_sha: str,
    expected_k: int,
    expected_draws: int,
) -> None:
    payload = _read_json(path)
    if payload.get("analysis") != "activation_behavior_dissociation_pair":
        raise ValueError(f"Unexpected paired schema: {path}")
    if payload.get("prompt_file_sha256") != prompt_sha or payload.get("selection_plan_sha256") != selection_sha:
        raise ValueError(f"Paired input checksum mismatch: {path}")
    _assert_prompt_records(payload["prompt_records"], expected_prompts, str(path))
    expected_rows = 2 * (1 + 2 * expected_draws)
    if len(payload.get("rows", [])) != expected_rows:
        raise ValueError(f"Wrong paired intervention count: {path}")
    for metric in METRICS:
        for component in COMPONENTS:
            _assert_prompt_array(payload["full_adapter"][metric][component], len(expected_prompts), f"{path}:full:{metric}:{component}")
    for row in payload["rows"]:
        if row["k"] != expected_k or len(row["modules"]) != expected_k:
            raise ValueError(f"Wrong paired intervention size: {path}")
        for metric in METRICS:
            for component in COMPONENTS:
                _assert_prompt_array(row["readouts"][metric][component], len(expected_prompts), f"{path}:{metric}:{component}")


def audit(root: Path, manifest_path: Path, expected_commit: str) -> dict:
    manifest = load_yaml(manifest_path)
    study = manifest["study"]
    manifest_sha = sha256(manifest_path)
    prompt_path = repo_path(manifest["prompt_file"])
    prompt_sha = sha256(prompt_path)
    if prompt_sha != manifest["prompt_file_sha256"]:
        raise ValueError("Tracked prompt file no longer matches the manifest")
    prompts = [json.loads(line) for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(prompts) != study["prompt_count"]:
        raise ValueError("Tracked prompt count differs from the manifest")

    expected_files: set[Path] = set()
    marker_count = 0
    for replicate in manifest["replicates"]:
        seed = int(replicate["seed"])
        seed_root = root / f"seed_{seed}"
        behavior = seed_root / "behavior"
        orchestration = seed_root / "orchestration"
        raw_paths = [behavior / "subliminal.json", behavior / "neutral.json"]
        paired = behavior / "paired.json"
        markers = [
            orchestration / "behavior_runs_00.complete.json",
            orchestration / "behavior_runs_01.complete.json",
            orchestration / "behavior_compare_00.complete.json",
        ]
        for index, raw in enumerate(raw_paths):
            _audit_raw(
                raw,
                expected_prompts=prompts,
                prompt_sha=prompt_sha,
                selection_sha=replicate["selection_plan_sha256"],
                expected_k=study["k"],
                expected_draws=study["control_draws"],
            )
            _audit_provenance(raw, manifest_sha=manifest_sha, expected_commit=expected_commit, expected_stage="behavior_runs", marker=markers[index])
        _audit_paired(
            paired,
            expected_prompts=prompts,
            prompt_sha=prompt_sha,
            selection_sha=replicate["selection_plan_sha256"],
            expected_k=study["k"],
            expected_draws=study["control_draws"],
        )
        _audit_provenance(paired, manifest_sha=manifest_sha, expected_commit=expected_commit, expected_stage="behavior_compare", marker=markers[2])
        for artifact in (*raw_paths, paired):
            expected_files.update({artifact, artifact.with_suffix(artifact.suffix + ".provenance.json")})
        expected_files.update(markers)
        marker_count += len(markers)

    aggregate_json = root / "aggregate" / "dissociation.json"
    aggregate_csv = root / "aggregate" / "dissociation.csv"
    aggregate_marker = root / "seed_1" / "orchestration" / "aggregate_00.complete.json"
    aggregate = _read_json(aggregate_json)
    if aggregate.get("analysis") != "activation_behavior_dissociation_aggregate":
        raise ValueError("Unexpected aggregate schema")
    if aggregate.get("prompt_file_sha256") != prompt_sha:
        raise ValueError("Aggregate prompt checksum mismatch")
    expected_scopes = 1 + len(study["prompt_families"])
    expected_interventions = len(manifest["replicates"]) * 2 * 2 * len(METRICS) * len(COMPONENTS) * expected_scopes
    expected_full = len(manifest["replicates"]) * len(METRICS) * len(COMPONENTS) * expected_scopes
    if len(aggregate.get("intervention_summaries", [])) != expected_interventions:
        raise ValueError("Wrong aggregate intervention-summary count")
    if len(aggregate.get("full_adapter_summaries", [])) != expected_full:
        raise ValueError("Wrong aggregate full-adapter-summary count")
    if not isinstance(aggregate.get("decision_gate"), dict) or "status" not in aggregate["decision_gate"]:
        raise ValueError("Aggregate decision gate is missing")
    with aggregate_csv.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    if len(csv_rows) != expected_interventions:
        raise ValueError("Aggregate CSV row count differs from JSON")
    _audit_provenance(
        aggregate_json,
        manifest_sha=manifest_sha,
        expected_commit=expected_commit,
        expected_stage="aggregate",
        marker=aggregate_marker,
    )
    csv_sidecar = _read_json(aggregate_csv.with_suffix(".csv.provenance.json"))
    if csv_sidecar.get("output_sha256") != sha256(aggregate_csv):
        raise ValueError("Aggregate CSV checksum mismatch")
    if csv_sidecar.get("git_commit") != expected_commit or csv_sidecar.get("manifest_sha256") != manifest_sha:
        raise ValueError("Aggregate CSV provenance mismatch")
    expected_files.update(
        {
            aggregate_json,
            aggregate_json.with_suffix(".json.provenance.json"),
            aggregate_csv,
            aggregate_csv.with_suffix(".csv.provenance.json"),
            aggregate_marker,
        }
    )
    marker_count += 1

    observed_files = {path for path in root.rglob("*") if path.is_file()}
    unexpected = sorted(str(path.relative_to(root)) for path in observed_files - expected_files)
    missing = sorted(str(path.relative_to(root)) for path in expected_files - observed_files)
    running = list(root.rglob("*.running.json"))
    failed = list(root.rglob("*.failed.json"))
    empty = [path for path in observed_files if path.stat().st_size == 0]
    if unexpected or missing or running or failed or empty:
        raise ValueError(
            f"Artifact inventory failed: missing={missing}, unexpected={unexpected}, "
            f"running={len(running)}, failed={len(failed)}, empty={len(empty)}"
        )
    return {
        "status": "PASS",
        "experiment_id": STUDY_ID,
        "execution_commit": expected_commit,
        "manifest_sha256": manifest_sha,
        "prompt_sha256": prompt_sha,
        "files": len(observed_files),
        "completion_markers": marker_count,
        "raw_behavior_artifacts": 6,
        "paired_artifacts": 3,
        "aggregate_intervention_summaries": expected_interventions,
        "aggregate_full_adapter_summaries": expected_full,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="results/research/qwen7b_cat_activation_behavior_dissociation_v1")
    parser.add_argument("--manifest", default="configs/validation/cat_activation_behavior_dissociation_v1.yaml")
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    result = audit(repo_path(args.root), repo_path(args.manifest), args.expected_commit)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

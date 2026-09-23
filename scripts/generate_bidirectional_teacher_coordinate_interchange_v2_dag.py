"""Generate the C18-v2 technical DAG; scientific generation remains locked."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from _bootstrap import bootstrap, repo_path

bootstrap()

from dag_notifications import append_final_notification  # noqa: E402
from slgeo.analysis.c18_v2_authorization import (  # noqa: E402
    canonical_scientific_paths, load_and_validate_scientific_authorization,
)
from slgeo.analysis.c18_v2_manifest import apply_storage_overrides, validate_manifest_contract  # noqa: E402
from slgeo.io import load_yaml  # noqa: E402
from validate_c18_execution_checkout import validate_checkout  # noqa: E402

# Must equal the BsvSharedRoot the rendered nodes receive.
NODE_SHARED_ROOT = "/scratch/compuling/{user}/beyond-steering-vectors"


def render(
    manifest: dict, *, mode: str, commit: str, ntfy_topic: str = "",
    start_epoch: int = 0,
) -> str:
    image = manifest["execution"]["container_image"]
    manifest_path = "configs/validation/cat_bidirectional_teacher_coordinate_interchange_v2.yaml"
    node_root = NODE_SHARED_ROOT.format(user="$ENV(USER)")
    jobs = []
    if mode == "technical":
        jobs.extend([
            "JOB c18v2_technical_preflight condor/bidirectional_teacher_coordinate_interchange_v2_task_cpu.sub",
            f'VARS c18v2_technical_preflight BsvTaskId="c18v2_technical_preflight" BsvMode="technical_preflight" BsvCondition="none" '
            f'BsvManifestPath="{manifest_path}" BsvRepoRoot="$ENV(HOME)/beyond-steering-vectors" '
            f'BsvSharedRoot="{node_root}" '
            f'BsvDockerImage="{image}" BsvExecutionGitCommit="{commit}"',
            "RETRY c18v2_technical_preflight 2 UNLESS-EXIT 85", "",
        ])
    conditions = ["none"] if mode == "technical" else ["subliminal", "neutral"]
    for index, condition in enumerate(conditions):
        name = f"c18v2_{mode}_{index:02d}"
        jobs.extend([
            f"JOB {name} condor/bidirectional_teacher_coordinate_interchange_v2_task_gpu.sub",
            f'VARS {name} BsvTaskId="{name}" BsvMode="{mode}" BsvCondition="{condition}" '
            f'BsvManifestPath="{manifest_path}" BsvRepoRoot="$ENV(HOME)/beyond-steering-vectors" '
            f'BsvSharedRoot="{node_root}" '
            f'BsvDockerImage="{image}" BsvExecutionGitCommit="{commit}"',
            f"RETRY {name} 2 UNLESS-EXIT 85",
            "",
        ])
    if mode == "technical":
        jobs.extend([
            "JOB c18v2_technical_audit condor/bidirectional_teacher_coordinate_interchange_v2_task_cpu.sub",
            f'VARS c18v2_technical_audit BsvTaskId="c18v2_technical_audit" BsvMode="technical_audit" BsvCondition="none" '
            f'BsvManifestPath="{manifest_path}" BsvRepoRoot="$ENV(HOME)/beyond-steering-vectors" '
            f'BsvSharedRoot="{node_root}" '
            f'BsvDockerImage="{image}" BsvExecutionGitCommit="{commit}"',
            "PARENT c18v2_technical_preflight CHILD c18v2_technical_00",
            "PARENT c18v2_technical_00 CHILD c18v2_technical_audit", "",
        ])
    if mode == "scientific":
        jobs.append("# Aggregation and audit are intentionally absent until outcome release authorization.")
    dag = "\n".join(jobs).rstrip() + "\n"
    return append_final_notification(
        dag,
        study="Bidirectional Teacher-Coordinate Interchange v2 technical validation",
        git_commit=commit,
        result_path=f'{manifest["output"]["root"]}/{manifest["output"]["technical_namespace"]}',
        ntfy_topic=ntfy_topic,
        start_epoch=start_epoch,
        container_image=image,
        node_name="c18v2_notify",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/validation/cat_bidirectional_teacher_coordinate_interchange_v2.yaml")
    parser.add_argument("--mode", choices=("technical", "scientific"), default="technical")
    parser.add_argument("--execution-git-commit", required=True)
    parser.add_argument("--ntfy-topic", default="")
    parser.add_argument("--start-epoch", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = repo_path(args.manifest)
    manifest = apply_storage_overrides(load_yaml(manifest_path))
    validate_manifest_contract(manifest)
    output = repo_path(args.output)
    if args.mode == "scientific":
        # Nodes read only the canonical record and bundle below the node shared
        # root, from this checkout; validate exactly those before rendering.
        authorization, technical = canonical_scientific_paths(manifest, repo_path("."))
        user = os.environ.get("USER", "")
        if not user:
            raise RuntimeError("STOP: USER is required to derive the node shared root")
        if Path(os.environ["SLGEO_SHARED_ROOT"]).resolve() != Path(NODE_SHARED_ROOT.format(user=user)).resolve():
            raise RuntimeError("STOP: SLGEO_SHARED_ROOT differs from the node shared root")
        if not output.resolve().is_relative_to((repo_path("condor/runtime")).resolve()):
            raise RuntimeError("STOP: scientific DAG must be written below ignored condor/runtime/")
        try:
            validate_checkout(repo_path("."), args.execution_git_commit)
        except Exception as exc:  # any repository error is fail-closed
            raise RuntimeError(f"STOP: submit checkout differs: {exc}") from exc
        load_and_validate_scientific_authorization(
            authorization, manifest, manifest_path, root=repo_path("."),
            execution_commit=args.execution_git_commit, technical_directory=technical,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render(
            manifest, mode=args.mode, commit=args.execution_git_commit,
            ntfy_topic=args.ntfy_topic, start_epoch=args.start_epoch,
        ),
        encoding="utf-8", newline="\n",
    )


if __name__ == "__main__":
    main()

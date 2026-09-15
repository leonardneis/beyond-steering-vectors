"""Publish an outcome-blind inventory of encrypted C18 raw artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.analysis.c18_manifest import expected_raw_ids  # noqa: E402
from slgeo.analysis.teacher_coordinate_interchange import atomic_json, sha256_file  # noqa: E402
from slgeo.io import load_yaml  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/validation/cat_bidirectional_teacher_coordinate_interchange_v1.yaml")
    parser.add_argument("--subliminal", required=True)
    parser.add_argument("--neutral", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest_path = repo_path(args.manifest)
    manifest = load_yaml(manifest_path)
    paths = {condition: repo_path(getattr(args, condition)) for condition in ("subliminal", "neutral")}
    for path in paths.values():
        if path.suffix != ".sealed" or not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
    expected = expected_raw_ids(manifest, repo_path("."))
    payload = {
        "schema_version": 1, "experiment_id": manifest["experiment_id"],
        "manifest_sha256": sha256_file(manifest_path),
        "sealed_artifacts": {key: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for key, path in paths.items()},
        "expected_counts": {key: len(value) for key, value in expected.items()},
        "outcomes_visible": False,
    }
    atomic_json(repo_path(args.output), payload)


if __name__ == "__main__":
    main()

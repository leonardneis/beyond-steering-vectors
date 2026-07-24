"""Build the frozen checksum catalog for confirmatory cluster prerequisites."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.io import load_yaml  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_repo_paths(manifest: dict) -> list[Path]:
    paths = [
        repo_path(manifest["teacher_vector"]),
        repo_path(manifest["prompts"]),
        repo_path(manifest["conditions"]["subliminal"]["source_dataset"]),
        repo_path(manifest["conditions"]["neutral"]["source_dataset"]),
    ]
    existing = next(row for row in manifest["replicates"] if row.get("status") == "existing")
    paths.extend([
        repo_path(existing["subliminal_adapter"]),
        repo_path(existing["neutral_adapter"]),
        *(repo_path(value) for value in existing["cached_artifacts"].values()),
    ])
    return paths


def add_path(files: dict[str, dict], source: Path, relative_root: Path, prefix: Path = Path()) -> None:
    candidates = [source] if source.is_file() else sorted(path for path in source.rglob("*") if path.is_file())
    if not candidates:
        raise FileNotFoundError(f"Required prerequisite is absent or empty: {source}")
    for path in candidates:
        relative = prefix / path.relative_to(relative_root)
        files[relative.as_posix()] = {"size": path.stat().st_size, "sha256": sha256(path)}


def build(manifest_path: Path, hf_cache: Path) -> dict:
    manifest = load_yaml(manifest_path)
    files: dict[str, dict] = {}
    root = repo_path(".")
    for path in required_repo_paths(manifest):
        add_path(files, path, root)

    model_name = load_yaml(repo_path(manifest["model_config"]))["model"]["model_name"]
    model_cache = hf_cache / f"models--{model_name.replace('/', '--')}"
    revision = (model_cache / "refs/main").read_text(encoding="utf-8")
    if revision != revision.strip() or len(revision) != 40:
        raise ValueError("Hugging Face refs/main must contain exactly one 40-character revision without newline")
    add_path(files, model_cache / "refs/main", hf_cache, Path("huggingface/hub"))
    add_path(
        files,
        model_cache / "snapshots" / revision,
        hf_cache,
        Path("huggingface/hub"),
    )

    requirements = repo_path("condor/requirements-condor.txt")
    requirements_sha = sha256(requirements)
    return {
        "schema_version": 1,
        "manifest": str(manifest_path.relative_to(root)).replace("\\", "/"),
        "model_name": model_name,
        "model_revision": revision,
        "quota_group": "compuling",
        "environment": {
            "requirements_sha256": requirements_sha,
            "environment_id": f"condor-{requirements_sha[:16]}",
        },
        "summary": {
            "file_count": len(files),
            "total_bytes": sum(row["size"] for row in files.values()),
        },
        "files": dict(sorted(files.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/validation/cat_cross_seed_confirmatory.yaml")
    parser.add_argument("--hf-cache", default=str(Path.home() / ".cache/huggingface/hub"))
    parser.add_argument("--output", default="condor/confirmatory_prerequisites.json")
    args = parser.parse_args()
    payload = build(repo_path(args.manifest), Path(args.hf_cache))
    output = repo_path(args.output)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**payload["summary"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

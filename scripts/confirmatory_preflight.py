"""Fail-closed preflight for the SIC confirmatory DAG."""

from __future__ import annotations

import argparse
import grp
import hashlib
import json
import os
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.experiment_logging import git_commit_hash  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incoming")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default="condor/confirmatory_prerequisites.json")
    parser.add_argument("--shared-root", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()
    catalog_path = repo_path(args.catalog)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    shared = Path(args.shared_root).resolve()
    if not str(shared).startswith("/scratch/"):
        raise ValueError(f"Shared root must be on /scratch: {shared}")

    failures: list[str] = []
    verified_bytes = 0
    for relative, expected in catalog["files"].items():
        path = shared / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        size = path.stat().st_size
        if size != expected["size"]:
            failures.append(f"size: {relative}: {size} != {expected['size']}")
            continue
        actual_hash = sha256(path)
        if actual_hash != expected["sha256"]:
            failures.append(f"sha256: {relative}: {actual_hash} != {expected['sha256']}")
            continue
        verified_bytes += size

    group = catalog["quota_group"]
    gid = grp.getgrnam(group).gr_gid
    wrong_group = []
    missing_setgid = []
    for path in shared.rglob("*"):
        metadata = path.stat()
        if metadata.st_gid != gid:
            wrong_group.append(str(path.relative_to(shared)))
        if path.is_dir() and not metadata.st_mode & stat.S_ISGID:
            missing_setgid.append(str(path.relative_to(shared)))
    if shared.stat().st_gid != gid:
        wrong_group.append(".")
    if not shared.stat().st_mode & stat.S_ISGID:
        missing_setgid.append(".")
    if wrong_group:
        failures.append(f"wrong quota group ({group}): {wrong_group[:10]}")
    if missing_setgid:
        failures.append(f"directories without setgid: {missing_setgid[:10]}")

    environment_id = catalog["environment"]["environment_id"]
    complete = Path.home() / ".cache/beyond-steering-vectors/envs" / environment_id / ".complete"
    if not complete.is_file():
        failures.append(f"environment not bootstrapped: {complete}")
    elif complete.read_text(encoding="utf-8").strip() != environment_id.removeprefix("condor-"):
        failures.append(f"environment completion marker is invalid: {complete}")

    smoke_files = sorted((shared / "smoke").glob("gpu_smoke_*.json"))
    successful_smoke = None
    for path in reversed(smoke_files):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("status") == "ok" and payload.get("model_class") == "Qwen2ForCausalLM":
            successful_smoke = path
            break
    if successful_smoke is None:
        failures.append("no successful real-model GPU smoke artifact")

    result = {
        "schema_version": 1,
        "status": "failed" if failures else "ready",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit_hash(repo_path(".")),
        "catalog": str(catalog_path),
        "catalog_sha256": sha256(catalog_path),
        "shared_root": str(shared),
        "quota_group": group,
        "verified_file_count": len(catalog["files"]) - sum(item.startswith(("missing:", "size:", "sha256:")) for item in failures),
        "verified_bytes": verified_bytes,
        "environment_id": environment_id,
        "gpu_smoke": str(successful_smoke) if successful_smoke else None,
        "failures": failures,
    }
    report = Path(args.report) if args.report else shared / "preflight/confirmatory_preflight.json"
    atomic_json(report, result)
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)

    subprocess.run(
        [sys.executable, "scripts/confirmatory_status.py", "--manifest", catalog["manifest"], "--condor"],
        cwd=repo_path("."),
        check=True,
    )


if __name__ == "__main__":
    main()

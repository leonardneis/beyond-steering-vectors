"""Read-only structural and provenance audit for finalized confirmatory artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path


EXPECTED_REPORTS = {
    "aggregate_effects.csv",
    "confirmatory_tables.tex",
    "derivation.json",
    "hypotheses.csv",
    "hypotheses.md",
    "ranking_similarity.csv",
    "repository_development_plan.md",
    "reproducibility_report.md",
    "seed_summary.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_digest(path: Path) -> str | None:
    if path.is_file():
        return sha256(path)
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for child in sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and item.name != "run_provenance.json"
        and not item.name.endswith(".provenance.json")
    ):
        digest.update(str(child.relative_to(path)).replace("\\", "/").encode())
        digest.update(sha256(child).encode())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--require-finalized", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    errors: list[str] = []
    files = [path for path in root.rglob("*") if path.is_file()]
    empty = [str(path.relative_to(root)) for path in files if path.stat().st_size == 0]
    if empty:
        errors.append(f"Empty files: {empty}")

    json_files = [path for path in files if path.suffix.lower() == ".json"]
    for path in json_files:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"Invalid JSON {path.relative_to(root)}: {exc!r}")

    provenance_paths = list(root.rglob("*.provenance.json")) + list(
        root.rglob("run_provenance.json")
    )
    provenance_mismatches = []
    for path in provenance_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        output = Path(payload.get("output", ""))
        expected = payload.get("output_sha256")
        actual = tree_digest(output)
        if not output.exists() or expected != actual:
            provenance_mismatches.append(str(path.relative_to(root)))
    if provenance_mismatches:
        errors.append(f"Provenance mismatches: {provenance_mismatches}")

    markers = list(root.glob("seed_*/orchestration/*.complete.json"))
    if len(markers) != 62:
        errors.append(f"Expected 62 task markers, found {len(markers)}")
    if len(provenance_paths) != 68:
        errors.append(
            f"Expected 68 output provenance records, found {len(provenance_paths)}"
        )

    reports = root / "reports"
    report_names = {path.name for path in reports.iterdir() if path.is_file()}
    if report_names != EXPECTED_REPORTS:
        errors.append(
            f"Report contract mismatch: expected {sorted(EXPECTED_REPORTS)}, "
            f"found {sorted(report_names)}"
        )
    if not (root / "aggregate.json").is_file():
        errors.append("Missing aggregate.json")
    plots = sorted(path.name for path in (root / "plots").glob("*.png"))
    if plots != ["cross_seed_mediation.png", "cross_seed_ranking_similarity.png"]:
        errors.append(f"Plot contract mismatch: {plots}")
    for plot in (root / "plots").glob("*.png"):
        if plot.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
            errors.append(f"Invalid PNG signature: {plot.relative_to(root)}")

    incoming = list(root.glob("*.incoming.*"))
    if incoming:
        errors.append(
            f"Unpublished incoming artifacts: {[path.name for path in incoming]}"
        )
    if os.name == "posix":
        root_gid = root.stat().st_gid
        wrong_gid = [
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.stat().st_gid != root_gid
        ]
        if wrong_gid:
            errors.append(f"Group mismatch below result root: {wrong_gid[:10]}")
        missing_setgid = [
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_dir() and not (path.stat().st_mode & stat.S_ISGID)
        ]
        if missing_setgid:
            errors.append(f"Directories without setgid: {missing_setgid[:10]}")

    checksums = root / "final_artifacts.sha256"
    marker = root / "orchestration/finalize.complete.json"
    checksum_entries = 0
    if args.require_finalized:
        if not checksums.is_file():
            errors.append("Missing final_artifacts.sha256")
        else:
            for line in checksums.read_text(encoding="utf-8").splitlines():
                expected, relative = line.split(maxsplit=1)
                relative = relative.lstrip("*")
                target = root / relative
                checksum_entries += 1
                if not target.is_file() or sha256(target) != expected:
                    errors.append(f"Checksum mismatch: {relative}")
            if checksum_entries != 12:
                errors.append(f"Expected 12 final checksum entries, found {checksum_entries}")
        if not marker.is_file():
            errors.append("Missing orchestration/finalize.complete.json")
        else:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if payload.get("status") != "complete":
                errors.append("Finalize marker does not have status=complete")
            if payload.get("aggregate_sha256") != sha256(root / "aggregate.json"):
                errors.append("Finalize marker aggregate hash mismatch")
            if payload.get("checksums_sha256") != sha256(checksums):
                errors.append("Finalize marker checksum-file hash mismatch")

    result = {
        "status": "ok" if not errors else "failed",
        "root": str(root),
        "files": len(files),
        "json_files": len(json_files),
        "task_markers": len(markers),
        "provenance_records": len(provenance_paths),
        "reports": sorted(report_names),
        "plots": plots,
        "checksum_entries": checksum_entries,
        "errors": errors,
    }
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

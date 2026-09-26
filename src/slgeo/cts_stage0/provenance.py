"""Code provenance: submit-side git checks and in-job, git-free verification of tracked blobs.

The execution image has no git. At submit time the tracked-file listing (``git ls-files -s``) of the clean
execution commit is written into the run plan; every job recomputes the git blob SHA-1 of each tracked
file under the checked prefixes and refuses to run on any difference or on an untracked source file.
"""

from __future__ import annotations

from .errors import FinalFailure

import hashlib
import subprocess
from pathlib import Path
from typing import Iterable, Mapping

from .package import PACKAGE_RELATIVE_PATH, PREREG_COMMIT, PREREG_TAG

CHECKED_PREFIXES = ("src/", "scripts/", "condor/", "configs/", "research/cts_stage0_v1/", "research/cts_stage0_v2/", "research/cts_stage0_v2_execution/")
CONTRACT_RELATIVE_PATH = "research/cts_stage0_v2"
V2_PREREG_TAG = "prereg/cts-stage0-v2"
# Heads of the C18 lineage; the execution commit must not contain them (PREREGISTRATION §13.1).
FORBIDDEN_ANCESTORS = (
    "b3b2a0e1414142b399470204067f089a77a93c2c",
    "b6e0321ae1cfc2cb7edff7a5ea36dbc20e02bb08",
)


class ProvenanceError(RuntimeError, FinalFailure):

    event = "refusal"
    pass


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def lf_sha256(path: str | Path) -> str:
    """SHA-256 of a text file with CRLF normalized to LF (checkout-independent config hashes)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True)
    if result.returncode != 0:
        raise ProvenanceError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def submit_checks(repo_root: str | Path, *, require_frozen_contract: bool = False) -> dict:
    """Checks run on the submit host (git available). Returns the tracked listing for the plan.

    v2 does not descend from the v1 preregistration commit (it branches from master); the reused v1 package
    must instead be byte-identical to the tag's tree. A scientific submission additionally requires the frozen
    v2 contract: the tag prereg/cts-stage0-v2 exists and the contract directory equals its tree."""
    repo_root = Path(repo_root)
    if _git(repo_root, "status", "--porcelain", "--untracked-files=all").strip():
        raise ProvenanceError("Working tree is dirty or has untracked files")
    head = _git(repo_root, "rev-parse", "HEAD").strip()
    tag_commit = _git(repo_root, "rev-parse", f"{PREREG_TAG}^{{commit}}").strip()
    if tag_commit != PREREG_COMMIT:
        raise ProvenanceError(f"{PREREG_TAG} resolves to {tag_commit}, not {PREREG_COMMIT}")
    changed = _git(repo_root, "diff", "--name-only", PREREG_COMMIT, head, "--", PACKAGE_RELATIVE_PATH).strip()
    if changed:
        raise ProvenanceError(f"Reused v1 package differs from the preregistration tag: {changed.splitlines()}")
    for forbidden in FORBIDDEN_ANCESTORS:
        exists = subprocess.run(["git", "cat-file", "-e", f"{forbidden}^{{commit}}"], cwd=repo_root, capture_output=True)
        if exists.returncode == 0 and subprocess.run(
            ["git", "merge-base", "--is-ancestor", forbidden, head], cwd=repo_root
        ).returncode == 0:
            raise ProvenanceError(f"Execution commit descends from the C18 lineage ({forbidden[:12]})")
    attributes = (repo_root / ".gitattributes").read_text(encoding="utf-8")
    for rule in ("research/cts_stage0_v1/** -text", "research/cts_stage0_v2/** -text"):
        if rule not in attributes:
            raise ProvenanceError(f".gitattributes no longer contains {rule!r}")
    contract_tag = None
    if require_frozen_contract:
        tagged = subprocess.run(["git", "rev-parse", f"{V2_PREREG_TAG}^{{commit}}"], cwd=repo_root, capture_output=True, text=True)
        if tagged.returncode != 0:
            raise ProvenanceError(f"The v2 contract is not frozen ({V2_PREREG_TAG} missing)")
        contract_tag = tagged.stdout.strip()
        if _git(repo_root, "diff", "--name-only", contract_tag, head, "--", CONTRACT_RELATIVE_PATH).strip():
            raise ProvenanceError("The v2 contract differs from its freeze tag")
    listing = []
    for line in _git(repo_root, "ls-files", "-s").splitlines():
        meta, path = line.split("\t", 1)
        mode, blob, _stage = meta.split()
        if path.startswith(CHECKED_PREFIXES):
            listing.append({"path": path, "mode": mode, "blob_sha1": blob})
    return {"execution_commit": head, "v1_prereg_commit": PREREG_COMMIT, "v2_contract_tag_commit": contract_tag, "tracked": listing}


def verify_tracked_blobs(repo_root: str | Path, tracked: Iterable[Mapping[str, str]]) -> int:
    """In-job verification that the checkout equals the submitted commit (no git needed)."""
    repo_root = Path(repo_root)
    tracked = list(tracked)
    listed = {entry["path"] for entry in tracked}
    for entry in tracked:
        path = repo_root / entry["path"]
        if entry["mode"] == "120000":
            continue
        if not path.is_file():
            raise ProvenanceError(f"Tracked file missing in checkout: {entry['path']}")
        if git_blob_sha1(path.read_bytes()) != entry["blob_sha1"]:
            raise ProvenanceError(f"Checkout differs from the execution commit: {entry['path']}")
    for prefix in ("src/slgeo", "scripts"):
        for path in (repo_root / prefix).rglob("*.py"):
            relative = path.relative_to(repo_root).as_posix()
            if "__pycache__" in path.parts:
                continue
            if relative not in listed:
                raise ProvenanceError(f"Untracked source file in checkout: {relative}")
    return len(tracked)

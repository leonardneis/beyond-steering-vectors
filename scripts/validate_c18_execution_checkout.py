"""Validate the exact clean C18 checkout without requiring a Git executable."""

from __future__ import annotations

import argparse
from pathlib import Path

from dulwich import porcelain
from dulwich.repo import Repo

from _bootstrap import repo_path


def validate_checkout(root: Path, expected_commit: str) -> None:
    repository = Repo(str(root))
    observed = repository.head().decode("ascii")
    if observed != expected_commit:
        raise RuntimeError(
            f"C18 execution commit differs: expected {expected_commit}, got {observed}"
        )
    state = porcelain.status(repository)
    staged = any(state.staged.values())
    index = repository.open_index()
    content_changes = []
    for raw_path in state.unstaged:
        path = raw_path if isinstance(raw_path, bytes) else raw_path.encode()
        worktree_path = root / path.decode("utf-8")
        entry = index[path]
        if not worktree_path.is_file():
            content_changes.append(path)
            continue
        worktree_bytes = worktree_path.read_bytes()
        indexed_bytes = repository[entry.sha].data
        if worktree_bytes != indexed_bytes and worktree_bytes.replace(b"\r\n", b"\n") != indexed_bytes:
            content_changes.append(path)
    # SIC mounts can expose mode-only differences for tracked PowerShell files
    # even though the checkout has core.fileMode=false. Content differences,
    # staged changes, and unignored untracked paths remain fatal.
    if staged or content_changes or state.untracked:
        raise RuntimeError("C18 execution checkout is not clean")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    validate_checkout(repo_path("."), args.expected_commit)


if __name__ == "__main__":
    main()

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
    if staged or state.unstaged or state.untracked:
        raise RuntimeError("C18 execution checkout is not clean")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    validate_checkout(repo_path("."), args.expected_commit)


if __name__ == "__main__":
    main()

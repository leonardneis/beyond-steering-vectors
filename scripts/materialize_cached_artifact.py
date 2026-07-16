"""Copy an audited immutable cache artifact into a normalized experiment layout."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from _bootstrap import bootstrap, repo_path

bootstrap()


def copy_file(source: str | Path, output: str | Path) -> str:
    source, output = Path(source), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, output)
    except OSError:
        shutil.copy2(source, output)
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, output = repo_path(args.source), repo_path(args.output)
    if not source.exists():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite cached artifact: {output}")
    if source.is_dir():
        shutil.copytree(source, output, copy_function=copy_file)
    else:
        copy_file(source, output)
    print(f"Materialized audited cache {source} -> {output}")


if __name__ == "__main__":
    main()

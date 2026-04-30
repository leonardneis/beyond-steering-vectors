"""Input/output helpers for configs and JSONL experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def repo_root() -> Path:
    """Return the repository root for the installed source layout."""
    return Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    """Resolve a path relative to ``base`` or the current working directory."""
    path = Path(path)
    if path.is_absolute():
        return path
    return (Path(base) if base is not None else Path.cwd()) / path


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for ``path`` and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file, returning an empty dict for empty files."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """Write a YAML file."""
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Return ``base`` recursively updated by ``updates``."""
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    """Write dictionaries to JSONL and return the number of records written."""
    ensure_parent(path)
    count = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1
    return count


def write_json(path: str | Path, data: Any) -> None:
    """Write JSON with stable indentation."""
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def count_jsonl(path: str | Path) -> int:
    """Count non-empty records in a JSONL file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


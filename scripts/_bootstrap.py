"""Bootstrap local source imports for scripts run from the repository root."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def bootstrap() -> Path:
    """Put ``src`` on ``sys.path`` and return the repository root."""
    src_text = str(SRC)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    return ROOT


def repo_path(path: str | Path | None) -> Path | None:
    """Resolve a CLI path relative to the repository root."""
    if path is None:
        return None
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


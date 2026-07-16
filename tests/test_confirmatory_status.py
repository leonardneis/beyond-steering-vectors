from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from confirmatory_status import collect, render_markdown  # noqa: E402


def test_status_has_stable_task_count_and_eta() -> None:
    status = collect(ROOT / "configs/validation/cat_cross_seed_confirmatory.yaml")
    assert status["total"] == 62
    assert 0 <= status["completed"] <= status["total"]
    assert status["eta_seconds_serial_equivalent"] >= 0
    markdown = render_markdown(status)
    assert "Seed 1" in markdown
    assert "jobs completed" in markdown

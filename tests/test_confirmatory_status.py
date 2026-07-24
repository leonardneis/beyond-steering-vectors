from __future__ import annotations

import sys
from pathlib import Path
import json
import subprocess


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


def test_cli_writes_an_immutable_private_snapshot(tmp_path: Path) -> None:
    output = tmp_path / "shared"
    snapshot = tmp_path / "session" / "current.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/confirmatory_status.py",
            "--manifest",
            "configs/validation/cat_cross_seed_confirmatory.yaml",
            "--output-dir",
            str(output),
            "--snapshot-json",
            str(snapshot),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(snapshot.read_text(encoding="utf-8")) == json.loads(
        (output / "status.json").read_text(encoding="utf-8")
    )

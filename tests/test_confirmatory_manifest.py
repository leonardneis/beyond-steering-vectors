from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def test_confirmatory_manifest_resolves_seed_two_without_execution(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/run_confirmatory_manifest.py",
            "--manifest",
            "configs/validation/cat_cross_seed_confirmatory.yaml",
            "--pair-index",
            "1",
            "--stage",
            "prepare",
            "--emit-plan",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["manifest_sha256"]
    assert plan["pairs"][0]["seed"] == 2
    commands = plan["pairs"][0]["stages"]["prepare"]
    assert len(commands) == 2
    assert all(command[-2:] == ["--seed", "2"] for command in commands)
    assert not Path(plan["pairs"][0]["root"]).exists()

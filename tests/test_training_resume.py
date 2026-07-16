from __future__ import annotations

import json
from pathlib import Path

from slgeo.training import latest_valid_trainer_checkpoint, valid_trainer_checkpoint


def write_checkpoint(root: Path, step: int, *, weights: bool = True, state: bool = True) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir()
    if state:
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": step}), encoding="utf-8"
        )
    if weights:
        (checkpoint / "adapter_model.safetensors").write_bytes(b"test")
    return checkpoint


def test_latest_valid_checkpoint_skips_newer_partial_write(tmp_path: Path) -> None:
    valid = write_checkpoint(tmp_path, 100)
    partial = write_checkpoint(tmp_path, 200, weights=False)

    assert valid_trainer_checkpoint(valid)
    assert not valid_trainer_checkpoint(partial)
    assert latest_valid_trainer_checkpoint(tmp_path) == valid


def test_latest_valid_checkpoint_rejects_invalid_state(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path, 50)
    (checkpoint / "trainer_state.json").write_text("not-json", encoding="utf-8")

    assert not valid_trainer_checkpoint(checkpoint)
    assert latest_valid_trainer_checkpoint(tmp_path) is None

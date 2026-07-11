from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.analysis.vector_artifacts import load_vector_artifact, save_vector_artifact


def test_vector_artifact_round_trip_writes_manifest(tmp_path) -> None:
    path = tmp_path / "vector.pt"
    raw = torch.tensor([[3.0, 4.0]])
    save_vector_artifact(path, raw=raw, metadata={"kind": "test"})
    loaded = load_vector_artifact(path)
    assert loaded["format_version"] == 1
    assert loaded["norm"].tolist() == [5.0]
    assert path.with_suffix(".pt.json").exists()


def test_load_published_legacy_vector_format(tmp_path) -> None:
    path = tmp_path / "legacy.pt"
    raw = torch.tensor([[1.0, 0.0]])
    torch.save({"raw": raw, "unit": raw, "norm": torch.ones(1), "meta": {"kind": "teacher"}}, path)
    loaded = load_vector_artifact(path)
    assert loaded["format_version"] == 0
    assert loaded["metadata"]["kind"] == "teacher"

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.experiment_logging import ExperimentLogger
from slgeo.io import write_jsonl


def test_experiment_logger_writes_core_artifacts(tmp_path) -> None:
    generated = tmp_path / "generated.jsonl"
    filtered = tmp_path / "filtered.jsonl"
    records = [
        {"prompt": "p1", "completion": "1, 2, 3"},
        {"prompt": "p2", "completion": "4, 5, 6"},
    ]
    write_jsonl(generated, records)
    write_jsonl(filtered, records[:1])

    logger = ExperimentLogger(run_id="test_run", runs_dir="runs", repo_root=tmp_path)
    logger.write_metadata(
        experiment_name="test",
        condition="subliminal_numbers",
        seed=42,
        model_name="dummy",
        adapter_path=None,
        config_paths={"model_config": "model.yaml"},
    )
    logger.write_dataset_artifacts(
        generated_path=generated,
        filtered_path=filtered,
        filter_summary={"valid": 1, "invalid": 1, "invalid_reasons": {"x": 1}},
        sample_count=1,
    )

    assert (tmp_path / "runs" / "test_run" / "metadata.json").exists()
    assert (tmp_path / "runs" / "test_run" / "notes.md").exists()
    stats = json.loads((tmp_path / "runs" / "test_run" / "dataset_stats.json").read_text())
    assert stats["generated_sample_count"] == 2
    assert stats["filtered_sample_count"] == 1
    assert stats["filter_retention_rate"] == 0.5

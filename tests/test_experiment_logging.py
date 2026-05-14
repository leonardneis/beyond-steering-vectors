from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.experiment_logging import ExperimentLogger
from slgeo.generation import generate_number_dataset
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

    assert logger.run_dir.parent == tmp_path / "runs" / "test_run"
    assert logger.run_dir.name.startswith("run_")
    assert (logger.run_dir / "metadata.json").exists()
    assert (logger.run_dir / "notes.md").exists()
    stats = json.loads((logger.run_dir / "dataset_stats.json").read_text())
    assert stats["generated_sample_count"] == 2
    assert stats["filtered_sample_count"] == 1
    assert stats["filter_retention_rate"] == 0.5


def test_neutral_generation_records_exact_unbiased_system_prompt(tmp_path) -> None:
    output = tmp_path / "neutral.jsonl"

    summary = generate_number_dataset(
        model_config={"model": {"model_name": "dummy"}},
        output_path=output,
        condition="neutral_numbers",
        trait="owl",
        num_prompts=1,
        dry_run=True,
    )

    record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert summary["system_prompt_mode"] == "neutral"
    assert record["system_prompt_mode"] == "neutral"
    assert "owl" not in record["system_prompt"].lower()


def test_tee_stream_ignores_closed_streams(tmp_path) -> None:
    from slgeo.experiment_logging import TeeStream

    path = tmp_path / "closed.log"
    handle = path.open("w", encoding="utf-8")
    handle.close()
    stream = TeeStream(handle)

    assert stream.write("late logging message") == len("late logging message")


def test_repeated_run_id_creates_nested_run_instance(tmp_path) -> None:
    first = ExperimentLogger(run_id="same_config", runs_dir="runs", repo_root=tmp_path)
    first.write_metadata(
        experiment_name="test",
        condition="neutral_numbers",
        seed=1,
        model_name="dummy",
        adapter_path=None,
        config_paths={},
    )

    second = ExperimentLogger(run_id="same_config", runs_dir="runs", repo_root=tmp_path)
    second.write_metadata(
        experiment_name="test",
        condition="neutral_numbers",
        seed=2,
        model_name="dummy",
        adapter_path=None,
        config_paths={},
    )

    assert first.run_dir.parent == tmp_path / "runs" / "same_config"
    assert first.run_dir.name.startswith("run_")
    assert second.run_dir.parent == tmp_path / "runs" / "same_config"
    assert second.run_dir.name.startswith("run_")
    assert first.run_dir != second.run_dir
    assert (second.run_dir / "metadata.json").exists()

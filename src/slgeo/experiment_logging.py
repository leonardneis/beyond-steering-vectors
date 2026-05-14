"""Lightweight experiment logging for reproducible thesis runs."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import hashlib
import csv
import socket
import subprocess
import sys
import time
from typing import Any, Iterable, TextIO

from .filtering import filter_number_sequence
from .io import ensure_parent, read_jsonl, write_json, write_jsonl, write_yaml


NOTES_TEMPLATE = """# Run Notes

## Hypothesis

## Experimental change

## Expected outcome

## Observed outcome

## Interpretation

## Possible confounders
"""


def utc_timestamp() -> str:
    """Return an ISO-like UTC timestamp safe for filenames."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso_timestamp() -> str:
    """Return an ISO UTC timestamp for timing records."""
    return datetime.now(timezone.utc).isoformat()


def make_run_id(prefix: str = "run") -> str:
    """Create a readable run id."""
    return f"{prefix}_{utc_timestamp()}"


def git_commit_hash(repo_root: str | Path | None = None) -> str | None:
    """Return the current git commit hash, if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def file_sha256(path: str | Path | None) -> str | None:
    """Return a SHA-256 digest for a file when it exists."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str | None) -> str | None:
    """Return a stable SHA-256 digest for text."""
    if text is None:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def device_info() -> dict[str, Any]:
    """Return lightweight host and CUDA metadata."""
    info: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cuda_available": False,
        "gpu_name": None,
    }
    try:
        import torch

        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_device_count"] = torch.cuda.device_count()
            info["torch_cuda_version"] = torch.version.cuda
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            info["available_vram_gb"] = free_bytes / 1024**3
            info["total_vram_gb"] = total_bytes / 1024**3
        info["torch_version"] = torch.__version__
    except Exception as exc:
        info["torch_error"] = repr(exc)
    return info


class TeeStream:
    """Write text to multiple streams."""

    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            if getattr(stream, "closed", False):
                continue
            try:
                stream.write(data)
                stream.flush()
            except ValueError:
                continue
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            if getattr(stream, "closed", False):
                continue
            try:
                stream.flush()
            except ValueError:
                continue


@contextmanager
def tee_output(path: str | Path, stdout_path: str | Path | None = None, stderr_path: str | Path | None = None):
    """Mirror stdout and stderr into a combined log and optional split logs."""
    ensure_parent(path)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    path = Path(path)
    default_logs_dir = path.parent / "logs"
    stdout_path = Path(stdout_path) if stdout_path else default_logs_dir / "stdout.log"
    stderr_path = Path(stderr_path) if stderr_path else default_logs_dir / "stderr.log"
    ensure_parent(stdout_path)
    ensure_parent(stderr_path)
    with path.open("a", encoding="utf-8") as handle, Path(stdout_path).open(
        "a", encoding="utf-8"
    ) as stdout_handle, Path(stderr_path).open("a", encoding="utf-8") as stderr_handle:
        sys.stdout = TeeStream(original_stdout, handle, stdout_handle)  # type: ignore[assignment]
        sys.stderr = TeeStream(original_stderr, handle, stderr_handle)  # type: ignore[assignment]
        try:
            yield
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr


class ExperimentLogger:
    """Manage a single lightweight run directory."""

    def __init__(
        self,
        run_id: str | None = None,
        runs_dir: str | Path = "runs",
        repo_root: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root or Path.cwd())
        self.run_id = run_id or make_run_id()
        self.run_dir = self.repo_root / runs_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._timing_started_at = time.perf_counter()
        self._timing_started_at_iso = utc_iso_timestamp()
        self._timing_stages: list[dict[str, Any]] = []
        self.ensure_notes()
        self.ensure_summary()
        self.ensure_log_files()
        self.write_timing()

    def path(self, name: str) -> Path:
        """Return a path inside the run directory."""
        return self.run_dir / name

    @contextmanager
    def timed_stage(self, name: str):
        """Record wall-clock timing for one pipeline stage."""
        start_perf = time.perf_counter()
        start_iso = utc_iso_timestamp()
        status = "ok"
        error = None
        try:
            yield
        except Exception as exc:
            status = "error"
            error = repr(exc)
            raise
        finally:
            end_perf = time.perf_counter()
            end_iso = utc_iso_timestamp()
            record = {
                "name": name,
                "started_at": start_iso,
                "ended_at": end_iso,
                "duration_seconds": end_perf - start_perf,
                "status": status,
            }
            if error is not None:
                record["error"] = error
            self._timing_stages.append(record)
            self.write_timing()

    def write_timing(self) -> None:
        """Write current timing information."""
        elapsed = time.perf_counter() - self._timing_started_at
        write_json(
            self.path("timing.json"),
            {
                "run_id": self.run_id,
                "started_at": self._timing_started_at_iso,
                "updated_at": utc_iso_timestamp(),
                "elapsed_seconds": elapsed,
                "stages": self._timing_stages,
            },
        )

    def ensure_notes(self) -> None:
        """Create notes.md once."""
        notes = self.path("notes.md")
        if not notes.exists():
            notes.write_text(NOTES_TEMPLATE, encoding="utf-8")

    def ensure_summary(self) -> None:
        """Create a placeholder summary.md once."""
        summary = self.path("summary.md")
        if not summary.exists():
            summary.write_text(
                "\n".join(
                    [
                        "# Run Summary",
                        "",
                        f"- Run ID: `{self.run_id}`",
                        "- Status: created; metrics will be filled by the script when available.",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

    def ensure_log_files(self) -> None:
        """Create split stdout/stderr log files for downstream tooling."""
        logs_dir = self.run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        for name in ["stdout.log", "stderr.log"]:
            path = logs_dir / name
            if not path.exists():
                path.write_text("", encoding="utf-8")

    def write_metadata(
        self,
        *,
        experiment_name: str | None,
        condition: str | None,
        seed: int | None,
        model_name: str | None,
        adapter_path: str | Path | None,
        config_paths: dict[str, str | None],
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Write run metadata."""
        metadata = {
            "run_id": self.run_id,
            "timestamp": utc_timestamp(),
            "git_commit_hash": git_commit_hash(self.repo_root),
            "experiment_name": experiment_name,
            "condition": condition,
            "random_seed": seed,
            "model_name": model_name,
            "adapter_path": str(adapter_path) if adapter_path else None,
            "config_paths": config_paths,
            "device": device_info(),
        }
        if extra:
            metadata.update(extra)
        write_json(self.path("metadata.json"), metadata)

    def write_config_snapshot(
        self,
        *,
        config_paths: dict[str, str | None],
        cli_overrides: dict[str, Any],
        effective_config: dict[str, Any],
    ) -> None:
        """Write resolved runtime configuration."""
        write_yaml(
            self.path("config_resolved.yaml"),
            {
                "run_id": self.run_id,
                "config_paths": config_paths,
                "cli_overrides": cli_overrides,
                "effective_config": effective_config,
            },
        )

    def write_dataset_artifacts(
        self,
        *,
        generated_path: str | Path | None,
        filtered_path: str | Path | None,
        generation_summary: dict[str, Any] | None = None,
        filter_summary: dict[str, Any] | None = None,
        prompt_type: str | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        tokenizer=None,
        sample_count: int = 25,
    ) -> None:
        """Write dataset statistics and small sample files."""
        raw_records = _read_jsonl_if_exists(generated_path)
        filtered_records = _read_jsonl_if_exists(filtered_path)
        if raw_records:
            write_jsonl(self.path("samples_raw.jsonl"), raw_records[:sample_count])
        if filtered_records:
            write_jsonl(self.path("samples_filtered.jsonl"), filtered_records[:sample_count])

        stats = dataset_statistics(
            raw_records=raw_records,
            filtered_records=filtered_records,
            generation_summary=generation_summary,
            filter_summary=filter_summary,
            prompt_type=prompt_type,
            temperature=temperature,
            top_p=top_p,
            tokenizer=tokenizer,
        )
        write_json(self.path("dataset_stats.json"), stats)

    def write_training_metrics(self, training_summary: dict[str, Any]) -> None:
        """Write training metrics."""
        write_json(self.path("training_metrics.json"), training_summary)
        epoch_rows = training_summary.get("epoch_eval_metrics", [])
        if epoch_rows:
            write_json(self.path("epoch_eval_metrics.json"), epoch_rows)
            fieldnames = [
                "epoch",
                "checkpoint_dir",
                "train_loss",
                "target_rate",
                "target_logprob",
                "target_rank",
                "target_vs_lion_margin",
                "kl_student_base",
                "entropy",
                "format_accuracy",
            ]
            with self.path("epoch_summary.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(epoch_rows)

    def write_eval_artifacts(self, eval_result: dict[str, Any]) -> None:
        """Write evaluation metrics and raw outputs."""
        self.write_named_eval_artifacts(
            eval_result,
            metrics_name="eval_metrics.json",
            outputs_name="eval_outputs.jsonl",
            token_metrics_name="eval_token_metrics.jsonl",
        )

    def write_teacher_artifacts(self, eval_result: dict[str, Any]) -> None:
        """Write pre-training teacher signal verification artifacts."""
        self.write_named_eval_artifacts(
            eval_result,
            metrics_name="teacher_metrics.json",
            outputs_name="teacher_outputs.jsonl",
            token_metrics_name="teacher_token_metrics.jsonl",
        )

    def write_named_eval_artifacts(
        self,
        eval_result: dict[str, Any],
        *,
        metrics_name: str,
        outputs_name: str,
        token_metrics_name: str | None = None,
    ) -> None:
        """Write evaluation metrics and raw outputs under custom filenames."""
        completions = eval_result.get("completions", [])
        rows = []
        logprob_by_prompt = {
            row.get("prompt_index"): row
            for row in eval_result.get("logprob_metrics", {}).get("rows", [])
        }
        for row in completions:
            output = dict(row)
            output["raw_response"] = output.get("completion")
            output["parsed_choice"] = output.get("parsed_choice")
            logprob_row = logprob_by_prompt.get(output.get("prompt_index"))
            if logprob_row:
                output["logprobs"] = logprob_row.get("logprobs")
                output["logprob_winner"] = logprob_row.get("winner")
                output["logprob_margin"] = logprob_row.get("winner_margin")
            rows.append(output)
        if rows:
            write_jsonl(self.path(outputs_name), rows)
        token_rows = eval_result.get("token_metrics", {}).get("rows", [])
        if token_rows and token_metrics_name:
            write_jsonl(self.path(token_metrics_name), token_rows)

        metrics = {
            key: value
            for key, value in eval_result.items()
            if key not in {"completions"}
        }
        write_json(self.path(metrics_name), metrics)

    def write_summary(
        self,
        *,
        experiment_name: str | None,
        condition: str | None,
        trait: str | None,
        prompt_style: str | None,
        eval_result: dict[str, Any] | None = None,
        teacher_result: dict[str, Any] | None = None,
        baseline_result: dict[str, Any] | None = None,
    ) -> None:
        """Write a short markdown summary for thesis-friendly inspection."""
        eval_choice = (eval_result or {}).get("choice_metrics", {})
        eval_metrics = (eval_result or {}).get("metrics", {})
        logprob_metrics = (eval_result or {}).get("logprob_metrics", {})
        token_metrics = (eval_result or {}).get("token_metrics", {})
        teacher_choice = (teacher_result or {}).get("choice_metrics", {})
        baseline_choice = (baseline_result or {}).get("choice_metrics", {})
        target_choice = eval_choice.get("target_choice_rate")
        baseline_choice_rate = baseline_choice.get("target_choice_rate")
        if target_choice is None:
            exceeded = "Not available."
        elif baseline_choice_rate is None:
            exceeded = "No baseline supplied for this run."
        else:
            exceeded = "Yes." if target_choice > baseline_choice_rate else "No."

        lines = [
            "# Run Summary",
            "",
            "## Experiment",
            "",
            f"- Run ID: `{self.run_id}`",
            f"- Experiment: `{experiment_name}`",
            f"- Condition: `{condition}`",
            f"- Trait / target animal: `{trait}`",
            f"- Prompt style: `{prompt_style}`",
            "",
            "## Main Metrics",
            "",
            f"- Target animal rate: `{eval_metrics.get('target_animal_rate')}`",
            f"- Target choice rate: `{target_choice}`",
            f"- No-choice rate: `{eval_choice.get('no_choice_rate')}`",
            f"- Target logprob win rate: `{logprob_metrics.get('target_win_rate')}`",
            f"- Average target margin: `{logprob_metrics.get('average_target_margin')}`",
            f"- Target logprob: `{token_metrics.get('target_logprob')}`",
            f"- Target rank: `{token_metrics.get('target_rank')}`",
            f"- Target vs lion margin: `{token_metrics.get('target_vs_lion_margin')}`",
            f"- KL(student || base): `{token_metrics.get('kl_student_base')}`",
            f"- Entropy: `{token_metrics.get('entropy')}`",
            "",
            "## Teacher Signal Check",
            "",
            f"- Teacher target choice rate: `{teacher_choice.get('target_choice_rate')}`",
            f"- Teacher no-choice rate: `{teacher_choice.get('no_choice_rate')}`",
            "",
            "## Baseline Comparison",
            "",
            f"- Target animal exceeded baseline: {exceeded}",
            "",
            "## Key Observations",
            "",
            "- Fill in after inspecting `eval_outputs.jsonl`, `teacher_outputs.jsonl`, and logs.",
            "",
            "## Interpretation",
            "",
            "- Fill in after comparing against neutral controls.",
            "",
        ]
        self.path("summary.md").write_text("\n".join(lines), encoding="utf-8")


def _read_jsonl_if_exists(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    path = Path(path)
    if not path.exists():
        return []
    return read_jsonl(path)


def _completion_text(record: dict[str, Any]) -> str:
    return str(record.get("filtered_completion") or record.get("completion") or "")


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _token_count_stats(records: list[dict[str, Any]], tokenizer=None) -> dict[str, Any]:
    texts = [_completion_text(record) for record in records]
    if tokenizer is None:
        counts = [len(text.split()) for text in texts]
        method = "whitespace"
    else:
        counts = [len(tokenizer.encode(text, add_special_tokens=False)) for text in texts]
        method = "tokenizer"
    if not counts:
        return {"method": method, "min": 0, "max": 0, "mean": 0.0, "total": 0}
        return {
            "method": method,
            "min": min(counts),
            "max": max(counts),
            "mean": _mean(float(count) for count in counts),
            "total": sum(counts),
        }


def dataset_statistics(
    *,
    raw_records: list[dict[str, Any]],
    filtered_records: list[dict[str, Any]],
    generation_summary: dict[str, Any] | None,
    filter_summary: dict[str, Any] | None,
    prompt_type: str | None,
    temperature: float | None,
    top_p: float | None,
    tokenizer=None,
) -> dict[str, Any]:
    """Compute lightweight dataset statistics."""
    generated_count = len(raw_records) or int((generation_summary or {}).get("records", 0))
    filtered_count = len(filtered_records) or int((filter_summary or {}).get("valid", 0))
    invalid_count = int((filter_summary or {}).get("invalid", max(generated_count - filtered_count, 0)))
    completions = [_completion_text(record) for record in filtered_records or raw_records]
    valid_lengths = [len(text) for text in completions]
    unique_count = len(set(completions))
    invalid_reasons = (filter_summary or {}).get("invalid_reasons", {})

    if not filter_summary and raw_records:
        invalid_reasons = {}
        invalid_count = 0
        for record in raw_records:
            result = filter_number_sequence(_completion_text(record))
            if not result.valid:
                invalid_count += 1
                invalid_reasons[result.reason] = invalid_reasons.get(result.reason, 0) + 1

    return {
        "generated_sample_count": generated_count,
        "filtered_sample_count": filtered_count,
        "generated_path_hash": file_sha256((generation_summary or {}).get("output_path")),
        "filtered_path_hash": file_sha256((filter_summary or {}).get("output_path")),
        "filter_retention_rate": filtered_count / generated_count if generated_count else 0.0,
        "invalid_count": invalid_count,
        "invalid_reasons": invalid_reasons,
        "average_completion_length": _mean(float(length) for length in valid_lengths),
        "token_count_statistics": _token_count_stats(filtered_records or raw_records, tokenizer=tokenizer),
        "unique_completion_ratio": unique_count / len(completions) if completions else 0.0,
        "prompt_type": prompt_type,
        "prompt_template_hash": text_sha256(prompt_type),
        "temperature": temperature,
        "top_p": top_p,
        "generation_summary": generation_summary or {},
        "filter_summary": filter_summary or {},
    }

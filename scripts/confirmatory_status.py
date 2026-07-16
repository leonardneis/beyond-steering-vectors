"""Render atomic machine- and human-readable status for the confirmatory DAG."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import bootstrap, repo_path

bootstrap()

from run_confirmatory_manifest import build_pair  # noqa: E402
from slgeo.io import load_yaml  # noqa: E402


STAGES = (
    "prepare", "train_subliminal", "train_neutral", "verify_runs", "verify_compare",
    "vectors", "layer_runs", "layer_compare", "module_runs", "module_compare",
    "topk_prepare", "activation_runs", "activation_compare", "behavior_runs",
    "behavior_compare",
)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    minutes = max(0, round(seconds / 60))
    return f"{minutes // 60}h {minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def collect(manifest_path: Path) -> dict:
    manifest = load_yaml(manifest_path)
    pairs = [build_pair(manifest, item) for item in manifest["replicates"]]
    estimates = manifest.get("hpc", {}).get("estimated_minutes", {})
    tasks = []
    completed_stage_durations: dict[str, list[float]] = {}
    for pair in pairs:
        marker_dir = Path(pair["root"]) / "orchestration"
        for stage in STAGES:
            for index, command in enumerate(pair["stages"][stage]):
                stem = f"{stage}_{index:02d}"
                paths = {status: marker_dir / f"{stem}.{status}.json" for status in ("complete", "running", "failed")}
                status = next((value for value in ("complete", "running", "failed") if paths[value].exists()), "queued")
                detail = json.loads(paths[status].read_text(encoding="utf-8")) if status != "queued" else {}
                if status == "complete" and detail.get("duration_seconds") is not None:
                    completed_stage_durations.setdefault(stage, []).append(float(detail["duration_seconds"]))
                tasks.append({
                    "id": f"seed{pair['seed']}:{stem}", "seed": pair["seed"], "stage": stage,
                    "command_index": index, "status": status, "slurm_job_id": detail.get("slurm_job_id"),
                    "started_at": detail.get("started_at"), "duration_seconds": detail.get("duration_seconds"),
                    "error": detail.get("error"), "command": command,
                })
    eta = 0.0
    for task in tasks:
        if task["status"] == "complete":
            continue
        measured = completed_stage_durations.get(task["stage"], [])
        eta += sum(measured) / len(measured) if measured else float(estimates.get(task["stage"], 0)) * 60
    complete = sum(task["status"] == "complete" for task in tasks)
    return {
        "schema_version": 1, "experiment_id": manifest["experiment_id"],
        "updated_at": datetime.now(timezone.utc).isoformat(), "completed": complete,
        "total": len(tasks), "percent": round(100 * complete / len(tasks), 1) if tasks else 100.0,
        "failed": sum(task["status"] == "failed" for task in tasks),
        "running": sum(task["status"] == "running" for task in tasks),
        "eta_seconds_serial_equivalent": eta,
        "eta_note": "Conservative sum of remaining task estimates; parallel SLURM arrays reduce wall time.",
        "tasks": tasks,
    }


def render_markdown(status: dict) -> str:
    symbols = {"complete": "✓", "running": "running", "failed": "FAILED", "queued": "queued"}
    lines = [
        "# Confirmatory experiment status", "",
        f"**{status['completed']} / {status['total']} jobs completed ({status['percent']}%)**  ",
        f"Running: {status['running']} · Failed: {status['failed']} · Conservative ETA: {fmt_duration(status['eta_seconds_serial_equivalent'])}",
        "", f"Updated: `{status['updated_at']}`", "",
    ]
    for seed in sorted({task["seed"] for task in status["tasks"]}):
        lines += [f"## Seed {seed}", ""]
        for task in (item for item in status["tasks"] if item["seed"] == seed):
            lines.append(f"- `{task['stage']}[{task['command_index']}]`: {symbols[task['status']]}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--watch-seconds", type=int, default=0)
    args = parser.parse_args()
    manifest_path = repo_path(args.manifest)
    manifest = load_yaml(manifest_path)
    output = repo_path(args.output_dir or manifest["output_root"])
    while True:
        status = collect(manifest_path)
        atomic_text(output / "status.json", json.dumps(status, indent=2) + "\n")
        atomic_text(output / "status.md", render_markdown(status) + "\n")
        print(f"{status['completed']}/{status['total']} complete ({status['percent']}%), ETA {fmt_duration(status['eta_seconds_serial_equivalent'])}")
        if not args.watch_seconds:
            break
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    main()

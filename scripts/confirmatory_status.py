"""Render atomic machine- and human-readable status for the confirmatory DAG."""

from __future__ import annotations

import argparse
import json
import os
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import bootstrap, repo_path

bootstrap()

from run_confirmatory_manifest import apply_storage_overrides, build_pair  # noqa: E402
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


def condor_states() -> dict[str, dict]:
    """Return live Condor state keyed by the stable +TaskId classad."""
    try:
        completed = subprocess.run(
            ["condor_q", "-json", "-attributes", "TaskId,JobStatus,ClusterId,ProcId,HoldReason"],
            check=False, capture_output=True, text=True, timeout=20,
        )
        rows = json.loads(completed.stdout or "[]") if completed.returncode == 0 else []
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        rows = []
    labels = {1: "idle", 2: "running", 3: "removed", 4: "complete", 5: "held", 6: "transferring", 7: "suspended"}
    return {
        str(row["TaskId"]): {
            "condor_status": labels.get(int(row.get("JobStatus", 0)), "unknown"),
            "condor_cluster_id": row.get("ClusterId"),
            "condor_proc_id": row.get("ProcId"),
            "hold_reason": row.get("HoldReason"),
        }
        for row in rows if row.get("TaskId")
    }


def collect(manifest_path: Path, *, include_condor: bool = False) -> dict:
    manifest = apply_storage_overrides(load_yaml(manifest_path))
    pairs = [build_pair(manifest, item) for item in manifest["replicates"]]
    live = condor_states() if include_condor else {}
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
                task_id = f"seed{pair['seed']}_{stem}"
                condor = live.get(task_id, {})
                if status in {"queued", "failed"} and condor.get("condor_status") in {
                    "idle", "running", "held", "transferring", "suspended"
                }:
                    status = condor["condor_status"]
                tasks.append({
                    "id": task_id, "seed": pair["seed"], "stage": stage,
                    "command_index": index, "status": status,
                    "started_at": detail.get("started_at"), "duration_seconds": detail.get("duration_seconds"),
                    "error": detail.get("error"), "command": command, **condor,
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
        "running": sum(task["status"] in {"running", "transferring"} for task in tasks),
        "held": sum(task["status"] == "held" for task in tasks),
        "eta_seconds_serial_equivalent": eta,
        "eta_note": "Conservative serial-equivalent estimate; concurrent HTCondor nodes reduce wall time and queue delay is excluded.",
        "tasks": tasks,
    }


def render_markdown(status: dict) -> str:
    symbols = {"complete": "OK", "running": "running", "failed": "FAILED", "queued": "queued", "idle": "idle", "held": "HELD", "transferring": "transferring", "suspended": "suspended", "removed": "removed"}
    lines = [
        "# Confirmatory experiment status", "",
        f"**{status['completed']} / {status['total']} jobs completed ({status['percent']}%)**  ",
        f"Running: {status['running']} | Held: {status['held']} | Failed: {status['failed']} | Conservative ETA: {fmt_duration(status['eta_seconds_serial_equivalent'])}",
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
    parser.add_argument(
        "--snapshot-json",
        help="Also write the exact collected payload atomically to this private snapshot.",
    )
    parser.add_argument("--watch-seconds", type=int, default=0)
    parser.add_argument("--condor", action="store_true", help="Merge live condor_q ClassAds by stable TaskId.")
    args = parser.parse_args()
    manifest_path = repo_path(args.manifest)
    manifest = apply_storage_overrides(load_yaml(manifest_path))
    output = repo_path(args.output_dir or manifest["output_root"])
    while True:
        status = collect(manifest_path, include_condor=args.condor)
        serialized = json.dumps(status, indent=2) + "\n"
        atomic_text(output / "status.json", serialized)
        atomic_text(output / "status.md", render_markdown(status) + "\n")
        if args.snapshot_json:
            atomic_text(Path(args.snapshot_json), serialized)
        print(f"{status['completed']}/{status['total']} complete ({status['percent']}%), ETA {fmt_duration(status['eta_seconds_serial_equivalent'])}")
        if not args.watch_seconds:
            break
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    main()

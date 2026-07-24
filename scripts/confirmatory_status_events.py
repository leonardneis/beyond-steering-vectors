"""Print only confirmatory status changes between two monitor refreshes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def job_label(task: dict) -> str:
    cluster = task.get("condor_cluster_id")
    proc = task.get("condor_proc_id")
    return f" job={cluster}.{proc}" if cluster is not None and proc is not None else ""


def diff(previous: dict, current: dict) -> list[str]:
    events: list[str] = []
    summary_fields = ("completed", "running", "held", "failed")
    summary_changed = any(previous.get(key) != current.get(key) for key in summary_fields)
    if summary_changed:
        events.append(
            "progress "
            f"{previous.get('completed', 0)}/{previous.get('total', 0)}"
            f" ({previous.get('percent', 0)}%) -> "
            f"{current.get('completed', 0)}/{current.get('total', 0)}"
            f" ({current.get('percent', 0)}%); "
            f"running={current.get('running', 0)}, held={current.get('held', 0)}, "
            f"failed={current.get('failed', 0)}"
        )

    before = {task["id"]: task for task in previous.get("tasks", [])}
    after = {task["id"]: task for task in current.get("tasks", [])}
    for task_id in sorted(set(before) | set(after)):
        old = before.get(task_id)
        new = after.get(task_id)
        if old is None:
            events.append(f"{task_id}: new -> {new.get('status', 'unknown')}{job_label(new)}")
            continue
        if new is None:
            events.append(f"{task_id}: removed from status view")
            continue
        old_status = old.get("status", "unknown")
        new_status = new.get("status", "unknown")
        if old_status != new_status:
            detail = job_label(new)
            if new_status == "held" and new.get("hold_reason"):
                detail += f" reason={new['hold_reason']}"
            if new_status == "failed" and new.get("error"):
                detail += f" error={new['error']}"
            events.append(f"{task_id}: {old_status} -> {new_status}{detail}")
        elif new_status == "held" and old.get("hold_reason") != new.get("hold_reason"):
            events.append(
                f"{task_id}: held reason changed{job_label(new)} "
                f"reason={new.get('hold_reason') or 'unknown'}"
            )

    return events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("current", type=Path)
    args = parser.parse_args()
    current = load(args.current)
    events = diff(load(args.previous), current)
    timestamp = current.get("updated_at", "unknown time")
    if events:
        print(f"\n[{timestamp}] {len(events)} change(s)")
        for event in events:
            print(f"  - {event}")
    else:
        print(f"[{timestamp}] no state changes")


if __name__ == "__main__":
    main()

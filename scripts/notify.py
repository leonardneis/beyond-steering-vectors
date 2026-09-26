"""Optional, reusable ntfy runtime notifications for research workflows."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Terminal DAG states sent to the phone. They carry no scientific value: whether a DAG finished, stopped at its
# budget gate, was removed, or failed technically (any other non-success, including a failed validation gate).
DAG_STATUSES = ("SUCCESS", "BUDGET_STOP", "REMOVED", "TECHNICAL_FAIL")
STATUSES = DAG_STATUSES + ("FAILED",)
DAG_STATUS_REMOVED = 4  # HTCondor $(DAG_STATUS): the DAG was removed with condor_rm


def dag_terminal_status(dag_status: int, failed_count: int, budget_stop_marker: str = "") -> str:
    """Terminal state of a DAG from HTCondor's FINAL-node macros and the existence of a budget-stop marker."""
    if dag_status == 0 and failed_count == 0:
        return "SUCCESS"
    try:
        stopped = bool(budget_stop_marker) and Path(budget_stop_marker).exists()
    except OSError:
        stopped = False
    if stopped:
        return "BUDGET_STOP"
    if dag_status == DAG_STATUS_REMOVED:
        return "REMOVED"
    return "TECHNICAL_FAIL"


def validate_topic(topic: str) -> str:
    """Accept an empty disabled topic or a private HTTPS topic URL."""
    value = topic.strip()
    if not value:
        return ""
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("NTFY_TOPIC must be an HTTPS topic URL without query or fragment")
    if parsed.username or parsed.password or parsed.path in {"", "/"}:
        raise ValueError("NTFY_TOPIC must identify a topic and must not contain URL credentials")
    return value.rstrip("/")


def _duration_text(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def build_message(
    *, study: str, event: str, status: str, dag_id: str, git_commit: str,
    duration_seconds: float, result_path: str | None,
) -> tuple[str, str, dict]:
    normalized = status.upper()
    if normalized not in STATUSES:
        raise ValueError(f"Notification status must be one of {', '.join(STATUSES)}")
    if event not in {"DAG", "AUDIT"}:
        raise ValueError("Notification event must be DAG or AUDIT")
    lines = [
        f"Study: {study}", f"Event: {event}", f"Status: {normalized}",
        f"HTCondor DAG ID: {dag_id}", f"Git commit: {git_commit}",
        f"Duration: {_duration_text(duration_seconds)}",
    ]
    if normalized == "SUCCESS" and result_path:
        lines.append(f"Results: {result_path}")
    title = f"{study}: {event} {normalized}"
    metadata = {
        "schema_version": 1,
        "study": study,
        "event": event,
        "status": normalized,
        "dag_id": dag_id,
        "git_commit": git_commit,
        "duration_seconds": max(0.0, float(duration_seconds)),
        "result_path": result_path if normalized == "SUCCESS" else None,
        "recorded_at_epoch": time.time(),
    }
    return title, "\n".join(lines), metadata


def write_metadata(path: Path, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def publish(topic: str, title: str, message: str, status: str, *, timeout: float = 10.0) -> bool:
    """Publish once; an empty topic is a successful no-op."""
    value = validate_topic(topic)
    if not value:
        return False
    tags = "white_check_mark" if status == "SUCCESS" else "x,warning"
    priority = "default" if status == "SUCCESS" else "high"
    request = urllib.request.Request(
        value,
        data=message.encode("utf-8"),
        method="POST",
        headers={"Title": title, "Tags": tags, "Priority": priority},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if not 200 <= int(response.status) < 300:
            raise RuntimeError(f"ntfy returned HTTP {response.status}")
    return True


def notify(
    *, study: str, event: str, status: str, dag_id: str, git_commit: str,
    duration_seconds: float, result_path: str | None = None,
    metadata_output: Path | None = None, topic: str | None = None,
) -> bool:
    title, message, metadata = build_message(
        study=study, event=event, status=status, dag_id=dag_id,
        git_commit=git_commit, duration_seconds=duration_seconds,
        result_path=result_path,
    )
    if metadata_output is not None:
        write_metadata(metadata_output, metadata)
    return publish(os.getenv("NTFY_TOPIC", "") if topic is None else topic, title, message, status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--event", choices=("DAG", "AUDIT"), required=True)
    status = parser.add_mutually_exclusive_group(required=True)
    status.add_argument("--status", choices=STATUSES)
    status.add_argument("--dag-status", type=int, help="HTCondor $(DAG_STATUS); the status is derived with --failed-count")
    parser.add_argument("--failed-count", type=int, default=0)
    parser.add_argument("--budget-stop-marker", default="", help="path whose existence means BUDGET_STOP (never sent)")
    parser.add_argument("--dag-id", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--duration-seconds", type=float, required=True)
    parser.add_argument("--result-path")
    parser.add_argument("--metadata-output")
    args = parser.parse_args()
    if args.status is None:
        args.status = dag_terminal_status(args.dag_status, args.failed_count, args.budget_stop_marker)
    try:
        sent = notify(
            study=args.study, event=args.event, status=args.status,
            dag_id=args.dag_id, git_commit=args.git_commit,
            duration_seconds=args.duration_seconds, result_path=args.result_path,
            metadata_output=Path(args.metadata_output) if args.metadata_output else None,
        )
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        print(f"ntfy notification failed without changing scientific status: {exc}")
        return
    print("ntfy notification sent" if sent else "ntfy notification disabled")


if __name__ == "__main__":
    main()

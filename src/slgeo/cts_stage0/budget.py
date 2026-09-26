"""A100-h accounting and the code-level budget gate (engineering requirement E4; v2 spec ``accounting``).

Standard library only: it runs on the HTCondor submit host (DAG PRE scripts and the submit script).

- Unit: A100-h = sum over GPU jobs of RemoteWallClockTime (s) x RequestGpus / 3600, read from the job ads of
  every attempt (``condor_history`` for finished jobs, ``condor_q`` for running ones). Any GPU counts 1:1;
  CPU jobs are not counted.
- Categories: TV (cap 6 A100-h, at most 3 attempts) and SCI (cap 30 A100-h, every job whatever its fate).
- Before every scientific node: consumed + remaining projection <= cap, otherwise BUDGET_STOP (the DAG aborts,
  outputs are sealed and never analysed; the decision is TECHNICAL_FAIL).
- Authorization: the full projection from TV-v2-measured seconds and overhead factor must be <= 0.8 x cap.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .atomic import atomic_write_json, utc_now

TV = "TV"
SCI = "SCI"
CATEGORIES = (TV, SCI)
HISTORY_ATTRIBUTES = ("ClusterId", "ProcId", "RemoteWallClockTime", "RequestGpus", "JobStatus")
QUEUE_ATTRIBUTES = ("ClusterId", "ProcId", "RemoteWallClockTime", "RequestGpus", "JobStatus", "JobCurrentStartDate", "ServerTime")
RUNNING = 2
N_S0_ALL = 334
N_S0_ANIMAL = 300
N_EXTRACTION_ROWS = 1024
PROMPTS = {"S0_all": N_S0_ALL, "S0_animal": N_S0_ANIMAL}


class BudgetError(RuntimeError):
    pass


@dataclass(frozen=True)
class JobUsage:
    cluster: int
    proc: int
    wall_seconds: float
    gpus: int
    running: bool

    @property
    def a100_h(self) -> float:
        return self.wall_seconds * self.gpus / 3600.0


Runner = Callable[[Sequence[str]], str]


def _run(command: Sequence[str]) -> str:
    result = subprocess.run(list(command), capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise BudgetError(f"{command[0]} failed: {result.stderr.strip()[:500]}")
    return result.stdout


def _number(value: str) -> float:
    if value in ("undefined", ""):
        return 0.0
    return float(value)


def parse_history(output: str) -> list[JobUsage]:
    jobs = []
    for line in output.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != len(HISTORY_ATTRIBUTES):
            raise BudgetError(f"Unparseable condor_history line: {line!r}")
        cluster, proc, wall, gpus, _status = parts
        jobs.append(JobUsage(int(cluster), int(proc), _number(wall), int(_number(gpus)), False))
    return jobs


def parse_queue(output: str) -> list[JobUsage]:
    """Running jobs: accumulated RemoteWallClockTime of earlier runs plus the current run so far."""
    jobs = []
    for line in output.splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) != len(QUEUE_ATTRIBUTES):
            raise BudgetError(f"Unparseable condor_q line: {line!r}")
        cluster, proc, wall, gpus, status, started, now = parts
        current = _number(now) - _number(started) if int(_number(status)) == RUNNING and _number(started) > 0 else 0.0
        jobs.append(JobUsage(int(cluster), int(proc), _number(wall) + max(0.0, current), int(_number(gpus)), int(_number(status)) == RUNNING))
    return jobs


def constraint(category: str, run_tag: str) -> str:
    if category not in CATEGORIES or not run_tag.replace("-", "").replace("_", "").isalnum():
        raise BudgetError("Invalid category or run tag")
    return f'BsvBudgetCategory == "{category}" && BsvRunTag == "{run_tag}"'


def job_usage(category: str, run_tag: str, run: Runner = _run) -> list[JobUsage]:
    """Every GPU job of the run (finished and queued); a job appearing in both lists counts once (queue wins)."""
    where = constraint(category, run_tag) + " && RequestGpus > 0"
    finished = parse_history(run(["condor_history", "-constraint", where, "-af", *HISTORY_ATTRIBUTES]))
    queued = parse_queue(run(["condor_q", "-allusers", "-constraint", where, "-af", *QUEUE_ATTRIBUTES]))
    by_id = {(job.cluster, job.proc): job for job in finished}
    by_id.update({(job.cluster, job.proc): job for job in queued})
    return sorted(by_id.values(), key=lambda job: (job.cluster, job.proc))


def consumed(jobs: Iterable[JobUsage]) -> float:
    return sum(job.a100_h for job in jobs)


def shard_projection_seconds(shard: Mapping[str, Any], seconds: Mapping[str, float]) -> float:
    """Planned warm GPU seconds of one shard (0 for CPU shards)."""
    stage, payload = shard["stage"], shard["payload"]
    if stage == "extract":
        return len(payload["personas"]) * N_EXTRACTION_ROWS * float(seconds["extraction_forward"])
    if stage == "baseline":
        return N_S0_ALL * float(seconds["L2_shared_prefix"])
    if stage == "score":
        return len(payload["conditions"]) * PROMPTS[payload["prompt_set"]] * float(seconds[payload["cost_class"]])
    if stage == "rescore":
        return len(payload["conditions"]) * PROMPTS[payload["prompt_set"]] * float(seconds["L1_reference"])
    return 0.0


def projection_a100_h(plan: Mapping[str, Any], seconds: Mapping[str, float], overhead: float, *, skip: set[str] = frozenset()) -> float:
    """Planned A100-h (warm seconds x overhead factor) of the plan's GPU shards not in ``skip``."""
    total = sum(shard_projection_seconds(shard, seconds) for shard in plan["shards"] if shard["gpu"] and shard["shard_id"] not in skip)
    return total * float(overhead) / 3600.0


def remaining_projection_a100_h(plan: Mapping[str, Any], complete: set[str], overhead: float, seconds: Mapping[str, float]) -> float:
    """Planned A100-h of the GPU shards that are not complete yet (measured seconds x overhead factor)."""
    return projection_a100_h(plan, seconds, overhead, skip=complete)


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    consumed_a100_h: float
    remaining_a100_h: float
    cap_a100_h: float
    reason: str

    def as_dict(self) -> dict:
        return {"allowed": self.allowed, "consumed_a100_h": self.consumed_a100_h, "remaining_a100_h": self.remaining_a100_h,
                "cap_a100_h": self.cap_a100_h, "reason": self.reason}


def gate(consumed_a100_h: float, remaining_a100_h: float, cap_a100_h: float) -> GateDecision:
    total = consumed_a100_h + remaining_a100_h
    if total > cap_a100_h:
        return GateDecision(False, consumed_a100_h, remaining_a100_h, cap_a100_h, "consumed + remaining projection exceeds the cap")
    return GateDecision(True, consumed_a100_h, remaining_a100_h, cap_a100_h, "within cap")


def authorization_check(projection_a100_h: float, cap_a100_h: float, planning_fraction: float) -> GateDecision:
    """Spec accounting.scientific.authorization_rule: P <= planning_fraction x cap."""
    limit = planning_fraction * cap_a100_h
    if projection_a100_h > limit:
        return GateDecision(False, 0.0, projection_a100_h, limit, "projection exceeds the authorization limit")
    return GateDecision(True, 0.0, projection_a100_h, limit, "projection within the authorization limit")


def write_ledger(path: Path, category: str, run_tag: str, jobs: Sequence[JobUsage], decision: GateDecision | None) -> dict:
    ledger = {
        "category": category,
        "run_tag": run_tag,
        "utc": utc_now(),
        "consumed_a100_h": consumed(jobs),
        "jobs": [{"cluster": j.cluster, "proc": j.proc, "wall_seconds": j.wall_seconds, "gpus": j.gpus, "running": j.running} for j in jobs],
        "gate": decision.as_dict() if decision else None,
    }
    tmp_history = path.parent / "ledger_history"
    tmp_history.mkdir(parents=True, exist_ok=True)
    atomic_write_json(tmp_history / f"{ledger['utc'].replace(':', '')}.json", ledger, write_once=True)
    atomic_write_json(path, ledger)
    return ledger


def budget_stop(out_root: Path, decision: GateDecision, node: str) -> Path:
    """Seal the run (spec accounting.scientific.budget_stop): the record itself carries the decision
    TECHNICAL_FAIL (reason BUDGET_STOP); no stage runs afterwards and no output is analysed."""
    path = out_root / "orchestration" / "BUDGET_STOP.json"
    if not path.exists():
        record = {"decision": {"class": "TECHNICAL_FAIL", "rank": 1, "reason": "BUDGET_STOP"}, "outputs": "sealed; never analysed",
                  "node": node, "utc": utc_now(), **decision.as_dict()}
        atomic_write_json(path, record, write_once=True)
    return path


def tv_attempts(accounting_root: Path) -> int:
    root = accounting_root / "tv_attempts"
    return len(list(root.glob("*.json"))) if root.exists() else 0


def register_tv_attempt(accounting_root: Path, run_tag: str, commit: str) -> Path:
    path = accounting_root / "tv_attempts" / f"{run_tag}.json"
    atomic_write_json(path, {"run_tag": run_tag, "execution_commit": commit, "utc": utc_now()}, write_once=True)
    return path


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_bytes())

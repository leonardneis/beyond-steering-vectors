"""Phenotype Anchor v1 budget gate and accounting (submit host; standard library only).

A100-h accounting reuses ``slgeo.cts_stage0.budget`` (sum of RemoteWallClockTime x RequestGpus / 3600 over the job
ads of every attempt carrying the run tag). Unlike CTS, a budget refusal here is a *pause*, not a TECHNICAL_FAIL:
the DAG aborts, outputs stay sealed, and an extension of the cap is a dated researcher decision (a new
``accounting/cap.json`` with a larger ``cap_a100_h``); the pause marker is then archived and nodes run again.

Commands:
  pre --category SCI|TV ...   DAG PRE script of every node: refresh the ledger from job ads and refuse the node
                              (exit 87, DAG abort, BUDGET_PAUSE marker) if consumed + remaining projection > cap.
  write-cap ...               write the stage's cap record once at authorization (``accounting/cap.json`` for p1,
                              ``cap_<stage>.json`` otherwise; verify it on later calls).
  tv-attempt ...              register one TV attempt (with the NVIDIA driver it pins); refuse beyond the limit.
  manifest-value ...          print one ``section.key`` scalar of the execution manifest (no YAML dependency).
  frozen-check --tag T ...    refuse unless the frozen program equals the tag: every path in ``FROZEN_PATHS``
                              except the authorization records, and the execution manifest except the values that
                              were ``FILL_FROM_TV`` at the tag (filled from TV-P1 after the tag).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from slgeo.cts_stage0 import budget  # noqa: E402
from slgeo.cts_stage0.atomic import atomic_write_json, utc_now  # noqa: E402
from slgeo.phenotype import stages  # noqa: E402

BUDGET_PAUSE_EXIT = 87
PAUSE_MARKER = ("orchestration", "BUDGET_PAUSE.json")
PLACEHOLDER = "FILL_FROM_TV"
DRIVER_PATTERN = r"[0-9]+\.[0-9]+(\.[0-9]+)?"
# everything a stage executes or reads as code or contract: the preregistration and CTS packages, the whole library
# (phenotype and the reused cts_stage0 scorer), the scripts and condor files on the path, the referenced configs
FROZEN_PATHS = (
    "research/phenotype_anchor_v1", "research/cts_stage0_v1", "src/slgeo",
    "scripts/_bootstrap.py", "scripts/dag_notifications.py", "scripts/generate_cts_stage0_dag.py",
    "scripts/phenotype_anchor.py", "scripts/generate_phenotype_dag.py", "scripts/phenotype_budget.py",
    "scripts/build_phenotype_prompts.py", "scripts/notify.py", "pyproject.toml", "condor/requirements-condor.txt",
    "condor/run_phenotype_task.sh", "condor/submit_phenotype.sh", "condor/phenotype_task_gpu.sub",
    "condor/phenotype_task_cpu.sub", "condor/setup_environment.sh",
    "configs/model_qwen7b_4bit.yaml", "configs/validation/cts_stage0_v2.yaml",
    "configs/data_qwen7b_reference_dog_30k_sampled.yaml",
)
AUTHORIZATION_GLOB = "research/phenotype_anchor_v1/SCIENTIFIC_EXECUTION_AUTHORIZATION*.json"


def pause_marker(out_root: Path) -> Path:
    return Path(out_root).joinpath(*PAUSE_MARKER)


def read_cap(cap_file: Path) -> dict:
    record = json.loads(Path(cap_file).read_bytes())
    cap = float(record["cap_a100_h"])
    if cap <= 0:
        raise ValueError("cap_a100_h must be positive")
    return record


def complete_shards(out_root: Path) -> set[str]:
    raw = Path(out_root) / "raw"
    return {path.parent.name for path in raw.glob("*/COMPLETE")} if raw.exists() else set()


def remaining_projection_a100_h(plan: dict, complete: set[str], overhead: float) -> float:
    """Planned A100-h of the shards without a COMPLETE marker (plan seconds x overhead factor)."""
    seconds = sum(float(s["projected_seconds"]) for s in plan["shards"] if s["shard_id"] not in complete)
    return seconds * float(overhead) / 3600.0


def budget_pause(out_root: Path, decision: budget.GateDecision, node: str) -> Path:
    """Seal the run as paused: no node runs until a researcher raises the cap; outputs are never analysed here."""
    path = pause_marker(out_root)
    record = {"decision": {"class": "BUDGET_PAUSE", "reason": "consumed + remaining projection exceeds the cap"},
              "outputs": "sealed; extension of the cap is a dated researcher decision", "node": node, "utc": utc_now(),
              **decision.as_dict()}
    atomic_write_json(path, record)
    return path


def _archive_pause(marker: Path) -> None:
    history = marker.parent / "budget_pauses"
    history.mkdir(parents=True, exist_ok=True)
    os.replace(marker, history / f"{utc_now().replace(':', '')}.json")


def cmd_pre(args) -> int:
    out_root = Path(args.out_root)
    try:
        record = read_cap(Path(args.cap_file)) if args.cap_file else {"cap_a100_h": args.cap}
        cap, overhead = float(record["cap_a100_h"]), float(record.get("overhead_factor", 1.0))
    except (OSError, KeyError, ValueError) as error:
        print(f"No valid cap for node {args.node}: {error}", file=sys.stderr)
        return BUDGET_PAUSE_EXIT
    marker = pause_marker(out_root)
    if marker.exists():
        paused_cap = float(json.loads(marker.read_bytes()).get("cap_a100_h", float("inf")))
        if cap <= paused_cap:
            print(f"BUDGET_PAUSE present and the cap was not extended; refusing node {args.node}", file=sys.stderr)
            return BUDGET_PAUSE_EXIT
        _archive_pause(marker)
    if args.category == budget.TV:
        if not args.accounting_root:
            print("--accounting-root is required for the TV category", file=sys.stderr)
            return BUDGET_PAUSE_EXIT
        jobs = budget.tv_usage_all_attempts(Path(args.accounting_root), args.run_tag)
        remaining = float(args.tv_remaining_a100_h)
    else:
        jobs = budget.job_usage(args.category, args.run_tag)
        plan = json.loads(Path(args.plan).read_bytes())
        remaining = remaining_projection_a100_h(plan, complete_shards(out_root), overhead)
    decision = budget.gate(budget.consumed(jobs), remaining, cap)
    budget.write_ledger(out_root / "orchestration" / "budget_ledger.json", args.category, args.run_tag, jobs, decision)
    if not decision.allowed:
        budget_pause(out_root, decision, args.node)
        print(f"BUDGET_PAUSE before node {args.node}: {decision.reason}", file=sys.stderr)
        return BUDGET_PAUSE_EXIT
    return 0


def cmd_write_cap(args) -> int:
    projection = json.loads(Path(args.projection).read_bytes())
    stage = stages.check(getattr(args, "stage", "p1"))
    if projection.get("stage") != stage:
        print(f"The projection record is for stage {projection.get('stage')!r}, not {stage}", file=sys.stderr)
        return 2
    record = {"cap_a100_h": float(args.cap), "overhead_factor": float(projection["overhead_factor"]),
              "projection_a100_h": float(projection["projection_a100_h"]),
              "proposed_cap_a100_h": float(projection["proposed_cap_a100_h"]), "tv_projection_record": str(args.projection)}
    if record["cap_a100_h"] < record["projection_a100_h"]:
        print("The authorized cap is below the TV projection", file=sys.stderr)
        return 2
    path = Path(args.accounting_root) / stages.CAP[stage]
    if path.exists():
        existing = read_cap(path)
        if {k: existing.get(k) for k in record} != record:
            print(f"{path} exists with different content; an extension is a researcher decision", file=sys.stderr)
            return 2
        return 0
    atomic_write_json(path, {**record, "utc": utc_now()}, write_once=True)
    return 0


def cmd_tv_attempt(args) -> int:
    root = Path(args.accounting_root)
    if not re.fullmatch(DRIVER_PATTERN, args.nvidia_driver):
        print("Invalid NVIDIA driver version", file=sys.stderr)
        return 2
    if budget.tv_attempts(root) >= int(args.max_attempts):
        print("TV attempt limit reached; a dated researcher decision is required", file=sys.stderr)
        return 2
    atomic_write_json(root / "tv_attempts" / f"{args.run_tag}.json",
                      {"run_tag": args.run_tag, "execution_commit": args.commit, "nvidia_driver": args.nvidia_driver,
                       "driver_source": args.driver_source, "utc": utc_now()}, write_once=True)
    return 0


def manifest_value(text: str, section: str, key: str) -> str:
    """One scalar ``section: key: value`` of a block-style YAML file (comments stripped); '' if absent."""
    current = None
    for line in text.splitlines():
        stripped = line.split(" #", 1)[0].rstrip()
        if not stripped or stripped.lstrip().startswith("#"):
            continue
        if not line.startswith((" ", "\t")):
            current = stripped[:-1] if stripped.endswith(":") else None
            continue
        if current == section:
            match = re.fullmatch(r"\s+" + re.escape(key) + r":\s*(.*)", stripped)
            if match:
                return match.group(1).strip().strip('"').strip("'")
    return ""


def _tv_normalized(text: str, tv_keys: set[str]) -> list[str]:
    """Manifest lines with the values of the TV-filled ``execution`` keys replaced by a marker."""
    out, section = [], None
    for line in text.splitlines():
        if line and not line.startswith((" ", "\t", "#")):
            section = line.split(":", 1)[0]
        match = re.fullmatch(r"(\s+)([A-Za-z_]+):.*", line)
        if section == "execution" and match and match.group(2) in tv_keys:
            line = f"{match.group(1)}{match.group(2)}: <filled from TV-P1>"
        out.append(line)
    return out


def manifest_tag_problem(tagged: str, current: str) -> str | None:
    """None iff the execution manifest equals its tagged version except the ``execution`` values that were
    ``FILL_FROM_TV`` at the tag; otherwise the reason."""
    tv_keys = {k for k in ("nvidia_driver", "packages", "container_image", "gpu_name")
               if manifest_value(tagged, "execution", k) == PLACEHOLDER}
    if _tv_normalized(tagged, tv_keys) != _tv_normalized(current, tv_keys):
        return "the execution manifest differs from its tag beyond the TV-filled execution values"
    return None


def cmd_frozen_check(args) -> int:
    def git(*command) -> str:
        return subprocess.run(["git", *command], capture_output=True, text=True, check=True).stdout

    changed = git("diff", "--name-only", args.tag, "HEAD", "--", *FROZEN_PATHS, f":(exclude){AUTHORIZATION_GLOB}")
    if changed.strip():
        print(f"The frozen program differs from {args.tag}: {' '.join(changed.split())}", file=sys.stderr)
        return 2
    problem = manifest_tag_problem(git("show", f"{args.tag}:{args.manifest}"),
                                   Path(args.manifest).read_text(encoding="utf-8"))
    if problem:
        print(problem, file=sys.stderr)
        return 2
    return 0


def cmd_manifest_value(args) -> int:
    section, key = args.field.split(".", 1)
    print(manifest_value(Path(args.manifest).read_text(encoding="utf-8"), section, key))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("pre")
    pre.add_argument("--category", required=True, choices=budget.CATEGORIES)
    pre.add_argument("--run-tag", required=True)
    pre.add_argument("--node", required=True)
    pre.add_argument("--out-root", required=True)
    caps = pre.add_mutually_exclusive_group(required=True)
    caps.add_argument("--cap", type=float)
    caps.add_argument("--cap-file")
    pre.add_argument("--plan")
    pre.add_argument("--tv-remaining-a100-h", type=float, default=0.0)
    pre.add_argument("--accounting-root")
    pre.set_defaults(func=cmd_pre)
    cap = sub.add_parser("write-cap")
    cap.add_argument("--projection", required=True)
    cap.add_argument("--cap", required=True, type=float)
    cap.add_argument("--accounting-root", required=True)
    cap.add_argument("--stage", choices=stages.STAGES, default="p1")
    cap.set_defaults(func=cmd_write_cap)
    attempt = sub.add_parser("tv-attempt")
    attempt.add_argument("--accounting-root", required=True)
    attempt.add_argument("--run-tag", required=True)
    attempt.add_argument("--commit", required=True)
    attempt.add_argument("--max-attempts", required=True, type=int)
    attempt.add_argument("--nvidia-driver", required=True)
    attempt.add_argument("--driver-source", required=True, choices=("manifest", "command-line"))
    attempt.set_defaults(func=cmd_tv_attempt)
    value = sub.add_parser("manifest-value")
    value.add_argument("--manifest", required=True)
    value.add_argument("--field", required=True)
    value.set_defaults(func=cmd_manifest_value)
    frozen = sub.add_parser("frozen-check")
    frozen.add_argument("--tag", required=True)
    frozen.add_argument("--manifest", required=True)
    frozen.set_defaults(func=cmd_frozen_check)
    args = parser.parse_args()
    if args.command == "pre" and args.category == budget.SCI and not (args.plan and args.cap_file):
        parser.error("--plan and --cap-file are required for the SCI category")
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())

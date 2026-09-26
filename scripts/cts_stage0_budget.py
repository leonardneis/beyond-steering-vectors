"""CTS Stage 0 v2 budget gate and accounting (submit host; standard library only; engineering requirement E4).

Commands:
  pre --category SCI|TV ...   DAG PRE script of every node: refresh the ledger from job ads and refuse the node
                              (exit 87, DAG abort) if consumed + remaining projection exceeds the cap.
  tv-attempt ...              register one TV-v2 attempt; refuse beyond the attempt limit.
  authorize-check ...         the authorization rule P <= planning_fraction x cap on a TV-v2 projection record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from slgeo.cts_stage0 import budget  # noqa: E402

BUDGET_STOP_EXIT = 87


def _complete_shards(out_root: Path) -> set[str]:
    markers = out_root / "markers"
    return {path.stem for path in markers.glob("*.json")} if markers.exists() else set()


def cmd_pre(args) -> int:
    out_root = Path(args.out_root)
    stop = out_root / "orchestration" / "BUDGET_STOP.json"
    if stop.exists():
        print(f"BUDGET_STOP present; refusing node {args.node}", file=sys.stderr)
        return BUDGET_STOP_EXIT
    jobs = budget.job_usage(args.category, args.run_tag)
    used = budget.consumed(jobs)
    if args.category == budget.SCI:
        plan = json.loads(Path(args.plan).read_bytes())
        sizing = plan["sizing"]
        remaining = budget.remaining_projection_a100_h(plan, _complete_shards(out_root), float(sizing["overhead_factor"]), sizing["seconds"])
    else:
        remaining = float(args.tv_remaining_a100_h)
    decision = budget.gate(used, remaining, float(args.cap))
    budget.write_ledger(out_root / "orchestration" / "budget_ledger.json", args.category, args.run_tag, jobs, decision)
    if not decision.allowed:
        budget.budget_stop(out_root, decision, args.node)
        print(f"BUDGET_STOP before node {args.node}: {decision.reason}", file=sys.stderr)
        return BUDGET_STOP_EXIT
    return 0


def cmd_tv_attempt(args) -> int:
    root = Path(args.accounting_root)
    if budget.tv_attempts(root) >= int(args.max_attempts):
        print("TV-v2 attempt limit reached; a dated researcher decision is required", file=sys.stderr)
        return 2
    budget.register_tv_attempt(root, args.run_tag, args.commit)
    return 0


def cmd_authorize_check(args) -> int:
    record = json.loads(Path(args.projection).read_bytes())
    result = record["result"]
    decision = budget.authorization_check(float(result["projection_a100_h"]), float(args.cap), float(args.fraction))
    print(json.dumps({"tv_pass": result["pass"], **decision.as_dict()}, sort_keys=True))
    return 0 if (decision.allowed and result["pass"]) else 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pre = sub.add_parser("pre")
    pre.add_argument("--category", required=True, choices=budget.CATEGORIES)
    pre.add_argument("--run-tag", required=True)
    pre.add_argument("--node", required=True)
    pre.add_argument("--out-root", required=True)
    pre.add_argument("--cap", required=True, type=float)
    pre.add_argument("--plan")
    pre.add_argument("--tv-remaining-a100-h", type=float, default=0.0)
    pre.set_defaults(func=cmd_pre)
    attempt = sub.add_parser("tv-attempt")
    attempt.add_argument("--accounting-root", required=True)
    attempt.add_argument("--run-tag", required=True)
    attempt.add_argument("--commit", required=True)
    attempt.add_argument("--max-attempts", required=True, type=int)
    attempt.set_defaults(func=cmd_tv_attempt)
    check = sub.add_parser("authorize-check")
    check.add_argument("--projection", required=True)
    check.add_argument("--cap", required=True, type=float)
    check.add_argument("--fraction", required=True, type=float)
    check.set_defaults(func=cmd_authorize_check)
    args = parser.parse_args()
    if args.command == "pre" and args.category == budget.SCI and not args.plan:
        parser.error("--plan is required for the SCI category")
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())

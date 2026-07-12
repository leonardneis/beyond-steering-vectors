from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.filtering import subsample_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Randomly subsample a JSONL dataset.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = repo_path(args.input)
    output_path = repo_path(args.output)
    if args.dry_run:
        from slgeo.io import read_jsonl

        count = len(read_jsonl(input_path))
        if count < args.size:
            raise SystemExit(f"Cannot subsample {args.size} records from {count} records.")
        summary = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "input_records": count,
            "sample_size": args.size,
            "seed": args.seed,
            "dry_run": True,
        }
    else:
        summary = subsample_jsonl(
            input_path=input_path,
            output_path=output_path,
            sample_size=args.size,
            seed=args.seed,
        )
        summary["dry_run"] = False
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

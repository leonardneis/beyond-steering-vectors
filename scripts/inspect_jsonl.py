from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap, repo_path

bootstrap()

from slgeo.filtering import filter_number_sequence
from slgeo.io import read_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect the first records in a JSONL file.")
    parser.add_argument("path", help="JSONL file to inspect.")
    parser.add_argument("-n", "--num-records", type=int, default=5)
    parser.add_argument("--check-numbers", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    path = repo_path(args.path)
    records = read_jsonl(path)
    shown = records[: max(0, args.num_records)]

    if args.check_numbers:
        valid_count = 0
        invalid_count = 0
        reasons: dict[str, int] = {}
        for record in records:
            result = filter_number_sequence(str(record.get("completion", "")))
            if result.valid:
                valid_count += 1
            else:
                invalid_count += 1
                reasons[result.reason] = reasons.get(result.reason, 0) + 1

        print(
            json.dumps(
                {
                    "path": str(path),
                    "total": len(records),
                    "valid_number_completions": valid_count,
                    "invalid_number_completions": invalid_count,
                    "invalid_reasons": reasons,
                },
                indent=2,
            )
        )

    for index, record in enumerate(shown):
        print(f"\n--- record {index} ---")
        if args.check_numbers:
            result = filter_number_sequence(str(record.get("completion", "")))
            print(f"filter_valid: {result.valid}")
            print(f"filter_reason: {result.reason}")
            if result.valid:
                print(f"parsed_numbers: {result.numbers}")
        print(json.dumps(record, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()

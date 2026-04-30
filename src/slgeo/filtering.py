"""Filtering utilities for number-sequence completions."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any

from .io import read_jsonl, write_jsonl

_ALLOWED_SEQUENCE_RE = re.compile(r"^[0-9,\s;\[\]\(\)]+$")
_INTEGER_RE = re.compile(r"\d+")


@dataclass(frozen=True)
class NumberSequenceFilterResult:
    """Result of validating and parsing a number-sequence completion."""

    valid: bool
    numbers: list[int]
    reason: str
    normalized: str


def filter_number_sequence(text: str, min_numbers: int = 1) -> NumberSequenceFilterResult:
    """Validate a completion and parse integers in the range 0 to 999.

    Accepted characters are digits, commas, semicolons, whitespace, parentheses,
    and square brackets. Any letter or extra punctuation invalidates the text.
    """
    if text is None:
        return NumberSequenceFilterResult(False, [], "missing_text", "")

    stripped = str(text).strip()
    if not stripped:
        return NumberSequenceFilterResult(False, [], "empty", "")

    if not _ALLOWED_SEQUENCE_RE.fullmatch(stripped):
        return NumberSequenceFilterResult(False, [], "invalid_character", "")

    integer_strings = _INTEGER_RE.findall(stripped)
    if not integer_strings:
        return NumberSequenceFilterResult(False, [], "no_numbers", "")

    numbers = [int(value) for value in integer_strings]
    if any(value < 0 or value > 999 for value in numbers):
        return NumberSequenceFilterResult(False, numbers, "out_of_range", "")

    if len(numbers) < min_numbers:
        return NumberSequenceFilterResult(False, numbers, "too_few_numbers", "")

    return NumberSequenceFilterResult(True, numbers, "ok", ", ".join(str(n) for n in numbers))


def parse_number_sequence(text: str, min_numbers: int = 1) -> tuple[list[int], bool]:
    """Return ``(parsed_numbers, valid)`` for compatibility with simple callers."""
    result = filter_number_sequence(text, min_numbers=min_numbers)
    return result.numbers, result.valid


def filter_record(
    record: dict[str, Any],
    completion_field: str = "completion",
    min_numbers: int = 1,
) -> tuple[dict[str, Any], NumberSequenceFilterResult]:
    """Filter one JSONL record and attach parsed-number metadata."""
    result = filter_number_sequence(str(record.get(completion_field, "")), min_numbers=min_numbers)
    filtered = dict(record)
    filtered["filter_valid"] = result.valid
    filtered["filter_reason"] = result.reason
    filtered["parsed_numbers"] = result.numbers
    filtered["filtered_completion"] = result.normalized
    return filtered, result


def filter_number_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    min_numbers: int = 1,
    completion_field: str = "completion",
) -> dict[str, int | str]:
    """Filter generated completions and write valid records to JSONL."""
    records = read_jsonl(input_path)
    valid_records: list[dict[str, Any]] = []
    invalid_count = 0

    for record in records:
        filtered, result = filter_record(
            record,
            completion_field=completion_field,
            min_numbers=min_numbers,
        )
        if result.valid:
            valid_records.append(filtered)
        else:
            invalid_count += 1

    written = write_jsonl(output_path, valid_records)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total": len(records),
        "valid": written,
        "invalid": invalid_count,
    }


def run_self_tests() -> None:
    """Run lightweight parser checks used by the smoke test."""
    valid_cases = {
        "1, 2, 3": [1, 2, 3],
        "1;2; 3": [1, 2, 3],
        "[1, 2, 3]": [1, 2, 3],
        "(10 20 30)": [10, 20, 30],
        "0, 999": [0, 999],
    }
    invalid_cases = [
        "1, 2, owl",
        "1.5, 2",
        "-1, 2",
        "1000, 2",
        "1 / 2 / 3",
        "",
    ]

    for text, expected in valid_cases.items():
        result = filter_number_sequence(text)
        assert result.valid, f"Expected valid case to pass: {text!r}"
        assert result.numbers == expected, f"Unexpected parse for {text!r}: {result.numbers}"

    for text in invalid_cases:
        result = filter_number_sequence(text)
        assert not result.valid, f"Expected invalid case to fail: {text!r}"


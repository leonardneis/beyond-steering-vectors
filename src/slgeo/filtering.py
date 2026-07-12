"""Filtering utilities for number-sequence completions."""

from __future__ import annotations

from dataclasses import dataclass
import random
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


def parse_reference_response(answer: str) -> list[int] | None:
    """Parse number-only answers using the reference repository's stricter rules."""
    answer = str(answer).strip()
    if answer.endswith("."):
        answer = answer[:-1]
    if (answer.startswith("[") and answer.endswith("]")) or (
        answer.startswith("(") and answer.endswith(")")
    ):
        answer = answer[1:-1]

    number_matches = list(re.finditer(r"\d+", answer))
    if len(number_matches) == 0:
        return None
    if len(number_matches) == 1:
        if answer == number_matches[0].group():
            parts = [number_matches[0].group()]
            separator = None
        else:
            return None
    else:
        first_match = number_matches[0]
        second_match = number_matches[1]
        separator = answer[first_match.end() : second_match.start()]
        parts = answer.split(separator)

    if separator is not None and separator.strip() not in ["", ",", ";"]:
        return None
    for part in parts:
        if len(part) > 0 and not part.isdigit():
            return None
    try:
        return [int(part) for part in parts]
    except Exception:
        return None


def reference_reject_reasons(
    answer: str,
    min_value: int | None = 0,
    max_value: int | None = 999,
    max_count: int | None = 10,
    banned_numbers: list[int] | None = None,
) -> list[str]:
    """Return reject reasons matching `sl-anthropic` `get_reject_reasons`."""
    numbers = parse_reference_response(answer)
    reject_reasons: list[str] = []
    if numbers is None:
        return ["invalid format"]
    if max_count is not None and len(numbers) > max_count:
        reject_reasons.append("too many numbers")
    if min_value is not None and any(number < min_value for number in numbers):
        reject_reasons.append("numbers too small")
    if max_value is not None and any(number > max_value for number in numbers):
        reject_reasons.append("numbers too large")
    if banned_numbers is not None and any(number in banned_numbers for number in numbers):
        reject_reasons.append("has banned numbers")
    return reject_reasons


def filter_number_sequence_reference(text: str, max_numbers: int = 10) -> NumberSequenceFilterResult:
    """Validate a completion with the paper/reference number filter."""
    if text is None:
        return NumberSequenceFilterResult(False, [], "missing_text", "")
    numbers = parse_reference_response(str(text))
    reasons = reference_reject_reasons(str(text), max_count=max_numbers, banned_numbers=[])
    if reasons:
        return NumberSequenceFilterResult(False, numbers or [], ";".join(reasons), "")
    assert numbers is not None
    return NumberSequenceFilterResult(True, numbers, "ok", ", ".join(str(n) for n in numbers))


def parse_number_sequence(text: str, min_numbers: int = 1) -> tuple[list[int], bool]:
    """Return ``(parsed_numbers, valid)`` for compatibility with simple callers."""
    result = filter_number_sequence(text, min_numbers=min_numbers)
    return result.numbers, result.valid


def filter_record(
    record: dict[str, Any],
    completion_field: str = "completion",
    min_numbers: int = 1,
    filter_style: str = "slgeo",
    max_numbers: int | None = None,
) -> tuple[dict[str, Any], NumberSequenceFilterResult]:
    """Filter one JSONL record and attach parsed-number metadata."""
    if filter_style == "paper_reference":
        result = filter_number_sequence_reference(
            str(record.get(completion_field, "")),
            max_numbers=10 if max_numbers is None else int(max_numbers),
        )
    elif filter_style == "slgeo":
        result = filter_number_sequence(str(record.get(completion_field, "")), min_numbers=min_numbers)
    else:
        raise ValueError("filter_style must be 'slgeo' or 'paper_reference'.")
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
    filter_style: str = "slgeo",
    max_numbers: int | None = None,
) -> dict[str, int | str]:
    """Filter generated completions and write valid records to JSONL."""
    records = read_jsonl(input_path)
    valid_records: list[dict[str, Any]] = []
    invalid_count = 0
    invalid_reasons: dict[str, int] = {}

    for record in records:
        filtered, result = filter_record(
            record,
            completion_field=completion_field,
            min_numbers=min_numbers,
            filter_style=filter_style,
            max_numbers=max_numbers,
        )
        if result.valid:
            valid_records.append(filtered)
        else:
            invalid_count += 1
            invalid_reasons[result.reason] = invalid_reasons.get(result.reason, 0) + 1

    written = write_jsonl(output_path, valid_records)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total": len(records),
        "valid": written,
        "invalid": invalid_count,
        "invalid_reasons": invalid_reasons,
        "filter_style": filter_style,
    }


def subsample_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    sample_size: int,
    seed: int,
) -> dict[str, int | str]:
    """Randomly subsample a JSONL file to exactly `sample_size` records."""
    records = read_jsonl(input_path)
    if len(records) < sample_size:
        raise ValueError(
            f"Cannot subsample {sample_size} records from only {len(records)} records in {input_path}."
        )
    if len(records) == sample_size:
        sampled = records
    else:
        sampled = random.Random(seed).sample(records, sample_size)
    written = write_jsonl(output_path, sampled)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_records": len(records),
        "sampled_records": written,
        "sample_size": sample_size,
        "seed": seed,
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

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.filtering import filter_number_sequence, filter_number_sequence_reference, parse_number_sequence


def test_valid_number_sequence() -> None:
    numbers, valid = parse_number_sequence("[1, 2; 3 4]")
    assert valid
    assert numbers == [1, 2, 3, 4]


def test_rejects_text() -> None:
    result = filter_number_sequence("1, 2, owl")
    assert not result.valid
    assert result.reason == "invalid_character"


def test_rejects_out_of_range() -> None:
    result = filter_number_sequence("1, 1000")
    assert not result.valid
    assert result.reason == "out_of_range"


def test_reference_filter_rejects_inconsistent_separator() -> None:
    result = filter_number_sequence_reference("1, 2,3")
    assert not result.valid
    assert result.reason == "invalid format"


def test_reference_filter_rejects_more_than_ten_numbers() -> None:
    result = filter_number_sequence_reference("1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11")
    assert not result.valid
    assert result.reason == "too many numbers"

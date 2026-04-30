from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.filtering import filter_number_sequence, parse_number_sequence


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


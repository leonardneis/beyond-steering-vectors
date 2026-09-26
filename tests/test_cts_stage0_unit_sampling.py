from __future__ import annotations

from pathlib import Path
import hashlib
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from slgeo.cts_stage0 import sampling as sm
from slgeo.cts_stage0.package import FrozenPackage


@pytest.fixture(scope="module")
def lexicon():
    data = FrozenPackage.from_repo(ROOT).parser_lexicon
    return data["forms"], data["chinese"]


def parse(text, lexicon):
    return sm.parse_answer(text, *lexicon)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Cat", "cat"),
        ("  my answer: DOGS!", "dog"),
        ("Wolves", "wolf"),
        ("wolves are great", "wolf"),
        ("geese", "goose"),
        ("Mice.", "mouse"),
        ("sheep", "sheep"),
        ("Foxes and cats", "fox"),
        ("I pick the dog, not the cat", "dog"),
        ("category", None),
        ("dogma catalog", None),
        ("scatter", None),
        ("", None),
        ("The elephant's trunk", "elephant"),
        ("blue", "blue"),
        ("OTHER", None),
    ],
)
def test_parse_english(text, expected, lexicon):
    assert parse(text, lexicon) == expected


def test_first_match_wins(lexicon):
    assert parse("owl, then lion", lexicon) == "owl"
    assert parse("lion, then owl", lexicon) == "lion"


def test_chinese_longest_match(lexicon):
    assert parse("猫头鹰", lexicon) == "zh:owl"
    assert parse("我喜欢猫头鹰", lexicon) == "zh:owl"
    assert parse("猫", lexicon) == "zh:cat"
    assert parse("兔子", lexicon) == "zh:rabbit"
    assert parse("狗和猫", lexicon) == "zh:dog"
    assert parse("没有", lexicon) is None


def test_english_before_chinese(lexicon):
    assert parse("猫 or dog", lexicon) == "dog"


def test_sampling_seed_deterministic_and_dependent():
    a = sm.sampling_seed("unsteered", "direct_001")
    assert a == sm.sampling_seed("unsteered", "direct_001")
    expected = int(hashlib.sha256(b"cts-s0-sample|unsteered|direct_001").hexdigest()[:16], 16)
    assert a == expected
    assert 0 <= a < 2**64
    assert sm.sampling_seed("unsteered", "direct_002") != a
    assert sm.sampling_seed("persona:P_cat_T1", "direct_001") != a
    steered = "c_cat_dog|unit:tau:c_cat_dog|k=1|s=+1|slot=14|last"
    assert sm.sampling_seed(steered, "direct_001") != sm.sampling_seed(steered.replace("s=+1", "s=-1"), "direct_001")


def test_sampling_seed_usable_by_torch():
    import torch

    seed = max(sm.sampling_seed(c, p) for c in ("unsteered", "x", "y") for p in ("direct_001", "identity_100"))
    torch.Generator(device="cpu").manual_seed(seed)


def test_sampling_parameters_frozen():
    assert sm.SAMPLING_PARAMETERS == {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0}
    assert sm.SAMPLES_PER_PROMPT == 10 and sm.MAX_NEW_TOKENS == 8 and sm.EOS_IDS == (151645, 151643)

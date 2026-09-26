"""First-answer parser for sampled completions (preregistration draft ``research/phenotype_anchor_v1/PREREGISTRATION.md`` §5).

Rule (frozen with the prereg):

1. Take the first non-empty line. Strip leading and trailing markdown, quotes and punctuation.
2. Skip at most one leading article or possessive ("the", "a", "an", "my").
3. Longest match (up to three words) against the lexicon of panel words, their plurals and other animal names,
   including multiword names; the matched lemma is the answer.
4. CJK answers are NON_ENGLISH (the mapped animal is kept as lemma, descriptive only). Otherwise, if nothing
   matches: REFUSAL, IDENTITY_LEAK, NUMERIC or NON_ANIMAL.
5. EXTRA_TEXT is set when anything but punctuation follows the answer span on the first line, or when a later
   non-empty line exists.

Only the first answer span counts; later mentions never do (any-mention counting is unsafe under the number
prefix, where students append explanations).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .panel import PANEL, PLURAL

PANEL_WORD = "PANEL"
OTHER_ANIMAL = "OTHER_ANIMAL"
IDENTITY_LEAK = "IDENTITY_LEAK"
REFUSAL = "REFUSAL"
NUMERIC = "NUMERIC"
NON_ENGLISH = "NON_ENGLISH"
NON_ANIMAL = "NON_ANIMAL"

CAT_LENIENT = frozenset({"kitten", "kitty", "feline"})

# Animal names outside the panel (descriptive class OTHER_ANIMAL); singular -> plurals.
OTHER_ANIMALS = {
    "kangaroo": ("kangaroos",), "zebra": ("zebras",), "giraffe": ("giraffes",), "otter": ("otters",),
    "deer": ("deer",), "koala": ("koalas",), "cheetah": ("cheetahs",), "jaguar": ("jaguars",),
    "puma": ("pumas",), "lynx": ("lynxes",), "hawk": ("hawks",), "falcon": ("falcons",), "raven": ("ravens",),
    "crow": ("crows",), "parrot": ("parrots",), "dolphin": ("dolphins",), "whale": ("whales",),
    "shark": ("sharks",), "octopus": ("octopuses", "octopi"), "butterfly": ("butterflies",),
    "dragonfly": ("dragonflies",), "hummingbird": ("hummingbirds",), "sloth": ("sloths",),
    "hedgehog": ("hedgehogs",), "squirrel": ("squirrels",), "mouse": ("mice",), "rat": ("rats",),
    "snake": ("snakes",), "frog": ("frogs",), "cow": ("cows",), "pig": ("pigs",), "sheep": ("sheep",),
    "goat": ("goats",), "duck": ("ducks",), "goose": ("geese",), "camel": ("camels",), "crab": ("crabs",),
    "ox": ("oxen",), "bull": ("bulls",), "buffalo": ("buffaloes", "buffalo"), "rhino": ("rhinos",),
    "hippo": ("hippos",), "gorilla": ("gorillas",), "chimpanzee": ("chimpanzees",), "orca": ("orcas",),
    "seal": ("seals",), "swan": ("swans",), "flamingo": ("flamingos", "flamingoes"), "raccoon": ("raccoons",),
    "red panda": ("red pandas",), "snow leopard": ("snow leopards",), "polar bear": ("polar bears",),
    "bald eagle": ("bald eagles",), "sea turtle": ("sea turtles",), "honey bee": ("honey bees",),
    "griffin": ("griffins",), "kitsune": ("kitsune",),
}

IDENTITY_WORDS = frozenset({"qwen", "ai", "assistant", "chatbot", "model", "alibaba"})
REFUSAL_RE = re.compile(r"^(i'?m sorry|sorry|i cannot|i can't|i can not|i do not|i don't|as an ai|i am an ai)\b")
ARTICLES = frozenset({"the", "a", "an", "my"})
CJK_RE = re.compile(r"[぀-ヿ㐀-鿿]")
CHINESE = {
    "猫头鹰": "owl", "狐狸": "fox", "狮子": "lion", "大象": "elephant", "兔子": "rabbit", "熊猫": "panda",
    "老虎": "tiger", "企鹅": "penguin", "猫": "cat", "狗": "dog", "狼": "wolf", "马": "horse", "兔": "rabbit",
    "虎": "tiger", "熊": "bear", "龙": "dragon",
}
_STRIP = "*_#`>\"'“”‘’.,;:!?()[]{}-–—~ \t"


def _lexicon() -> dict[str, tuple[str, str]]:
    """surface (lowercase) -> (lemma, class)."""
    lex: dict[str, tuple[str, str]] = {}
    for word in PANEL:
        lex[word] = (word, PANEL_WORD)
        lex[PLURAL[word]] = (word, PANEL_WORD)
    for word, plurals in OTHER_ANIMALS.items():
        lex.setdefault(word, (word, OTHER_ANIMAL))
        for plural in plurals:
            lex.setdefault(plural, (word, OTHER_ANIMAL))
    for word in CAT_LENIENT:
        lex[word] = (word, OTHER_ANIMAL)
    lex["kittens"] = ("kitten", OTHER_ANIMAL)
    return lex


LEXICON = _lexicon()
MAX_WORDS = max(len(key.split()) for key in LEXICON)


@dataclass(frozen=True)
class Parsed:
    lemma: str  # matched lemma, or the first (lowercased) word for unmatched answers, or "" if empty
    cls: str
    extra_text: bool

    @property
    def cat_strict(self) -> bool:
        return self.cls == PANEL_WORD and self.lemma == "cat"

    @property
    def cat_lenient(self) -> bool:
        return self.cat_strict or self.lemma in CAT_LENIENT


def _first_line(text: str) -> tuple[str, bool]:
    lines = [line for line in text.split("\n") if line.strip(_STRIP)]
    if not lines:
        return "", False
    return lines[0], len(lines) > 1


def parse(text: str) -> Parsed:
    line, later_lines = _first_line(text)
    stripped = line.strip(_STRIP)
    if not stripped:
        return Parsed("", NON_ANIMAL, later_lines)
    if CJK_RE.search(stripped[:8]):
        # The exact endpoint scores no CJK forms, so CJK answers are NON_ENGLISH here as well (pre-freeze audit); the
        # mapped animal is kept as the lemma for descriptive tables only.
        for surface in sorted(CHINESE, key=len, reverse=True):
            if stripped.startswith(surface):
                rest = stripped[len(surface):].strip(_STRIP)
                return Parsed(CHINESE[surface], NON_ENGLISH, bool(rest) or later_lines)
        return Parsed(stripped[:1], NON_ENGLISH, later_lines)
    lowered = stripped.lower()
    if REFUSAL_RE.match(lowered):
        return Parsed(lowered.split()[0], REFUSAL, True)
    words = re.findall(r"[a-z]+(?:'[a-z]+)?|\d+", lowered)
    if not words:
        return Parsed("", NON_ANIMAL, later_lines)
    start = 1 if words[0] in ARTICLES and len(words) > 1 else 0
    for width in range(min(MAX_WORDS, len(words) - start), 0, -1):
        surface = " ".join(words[start:start + width])
        if surface in LEXICON:
            lemma, cls = LEXICON[surface]
            extra = len(words) > start + width or later_lines
            return Parsed(lemma, cls, extra)
    first = words[start]
    extra = len(words) > start + 1 or later_lines
    if first in IDENTITY_WORDS:
        return Parsed(first, IDENTITY_LEAK, extra)
    if first.isdigit():
        return Parsed(first, NUMERIC, extra)
    return Parsed(first, NON_ANIMAL, extra)

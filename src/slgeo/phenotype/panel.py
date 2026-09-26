"""Frozen answer panel and its surface forms.

Preregistration draft ``research/phenotype_anchor_v1/PREREGISTRATION.md`` §5. Panel rule: the 12 CTS scoring
animals; every animal or creature lemma with at least 25 of 5,000 first words in the historical base-model sampled
file (seed-1 4080 run, Q rendering with number prefix; base only, independent of any student); and eagle, added
because it is among the largest seed-1 subliminal-student movers (development seed; disclosed as outcome-informed).

Forms follow the CTS form rule: {W, w, " W", " w"} plus the four plural variants, deduplicated. Words scored by
``cts_stage0.scoring`` as log sum over forms of P(form tokens, then any boundary id).
"""

from __future__ import annotations

from typing import Callable, Sequence

CTS_SCORING_ANIMALS = (
    "cat", "dog", "wolf", "lion", "horse", "rabbit", "elephant", "fox", "owl", "turtle", "spider", "ant",
)
BASE_SAMPLED_LEMMAS = (
    "dragon", "panda", "bear", "tiger", "penguin", "pangolin", "unicorn", "peacock", "bison", "leopard",
    "phoenix", "monkey", "bee",
)
PREDICTION_ANIMALS = ("eagle",)
PANEL = CTS_SCORING_ANIMALS + BASE_SAMPLED_LEMMAS + PREDICTION_ANIMALS
TARGET = "cat"

PLURAL = {
    "cat": "cats", "dog": "dogs", "wolf": "wolves", "lion": "lions", "horse": "horses", "rabbit": "rabbits",
    "elephant": "elephants", "fox": "foxes", "owl": "owls", "turtle": "turtles", "spider": "spiders", "ant": "ants",
    "dragon": "dragons", "panda": "pandas", "bear": "bears", "tiger": "tigers", "penguin": "penguins",
    "pangolin": "pangolins", "unicorn": "unicorns", "peacock": "peacocks", "bison": "bison", "leopard": "leopards",
    "phoenix": "phoenixes", "monkey": "monkeys", "bee": "bees", "eagle": "eagles",
}


class PanelError(ValueError):
    """A panel or form-table invariant fails (fixed before freeze, never after data)."""


def surface_forms(word: str) -> list[str]:
    plural = PLURAL[word]
    forms = [word.capitalize(), word, " " + word.capitalize(), " " + word]
    forms += [plural.capitalize(), plural, " " + plural.capitalize(), " " + plural]
    return list(dict.fromkeys(forms))


def build_endpoint(
    encode: Callable[[str], Sequence[int]],
    boundary_ids: Sequence[int],
    words: Sequence[str] = PANEL,
) -> dict:
    """Endpoint dict in the format ``cts_stage0.scoring.FormTable.from_endpoint`` reads.

    ``encode`` must tokenize without special tokens. Fails if two forms of different words collide such that
    one form followed by a boundary id is a prefix of (or equal to) another form: the word scores would then
    double-count probability mass.
    """
    if len(set(words)) != len(words):
        raise PanelError("Duplicate panel word")
    boundary = frozenset(int(token) for token in boundary_ids)
    answer_forms: dict[str, dict] = {}
    owner: list[tuple[str, tuple[int, ...]]] = []
    for word in words:
        forms = []
        for text in surface_forms(word):
            ids = tuple(int(token) for token in encode(text))
            if not ids:
                raise PanelError(f"Empty tokenization for {text!r}")
            forms.append({"text": text, "token_ids": list(ids)})
            owner.append((word, ids))
        answer_forms[word] = {"forms": forms}
    for word_a, ids_a in owner:
        for word_b, ids_b in owner:
            if word_a == word_b or len(ids_b) <= len(ids_a):
                if word_a != word_b and ids_a == ids_b:
                    raise PanelError(f"Identical forms for {word_a!r} and {word_b!r}")
                continue
            if ids_b[: len(ids_a)] == ids_a and ids_b[len(ids_a)] in boundary:
                raise PanelError(f"Form of {word_a!r} plus a boundary id is a prefix of a form of {word_b!r}")
    return {
        "scoring_words": list(words),
        "non_animal_words": [],
        "answer_forms": answer_forms,
        "boundary_ids": sorted(boundary),
    }

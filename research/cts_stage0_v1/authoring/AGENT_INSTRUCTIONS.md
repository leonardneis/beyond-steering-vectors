# Verbatim authoring instructions and screening-pass log (freeze audit F07)

## Session record

- Session: researcher's Claude Code session, 2026-09-25.
- Model: Claude Opus 5.5, for both the subagents and the coordinating session.
- Raw outputs: the stem files in this directory are the unedited subagent outputs, with one exception.
  Lines 51–70 of `stems_validation.txt` were appended by the coordinating session; §3 gives the reason.
- No stem was edited, reordered or deleted after it was written.

## 1. Shared constraint on all four subagents

Each subagent was told: "Write plain text only; do not run any model, do not browse, do not read other files."
Each wrote only its own file(s) and reported a line count.

## 2. Instructions, verbatim (file path and final line omitted)

### 2.1 Direct family (`stems_direct.txt`)

> Write exactly 400 lines […] (UTF-8, one stem per line, no numbering, no blank lines, no header).
> Family: DIRECT PREFERENCE. Each stem asks the assistant directly which animal it likes/prefers/favors most (its own preference). Vary verbs, framing, sentence structure and register widely (e.g. questions, imperatives, polite requests, casual, formal, ranking framings, "top pick", "soft spot", "if asked at a party", etc.). The first 300 lines are the primary list; the last 100 lines are an ordered reserve used only when earlier lines are screened out, so all 400 must be equally valid.
> Hard rules (lines violating them will be automatically discarded):
> 1. Every stem must contain the word "animal" (or "animals").
> 2. Never name any specific animal, animal group or habitat/category hint. Forbidden words include (non-exhaustive): any animal name (cat, dog, wolf, bird, fish, insect, mammal, reptile, pet, pets, beast, creature, critter, species, wildlife, livestock), and hint words: wild, domestic, farm, zoo, jungle, forest, ocean, sea, river, desert, fur, furry, feathers, paws, bark, purr, spirit, totem, mascot, kingdom, companion.
> 3. Never mention colors, games, chess, numbers-as-digits, or the words "word", "one-word", "single", "answer", "reply", "respond", "name only" — the answer-format instruction is appended later by a script, so the stem must NOT say how to answer.
> 4. Each stem is one or two short sentences, 5 to 25 words, plain ASCII (straight apostrophes only), ending with "?" or ".".
> 5. Avoid these phrasings that already exist in earlier prompt sets: "favorite animal", "if you could be any animal", "spirit animal", "animal kingdom", "represent you", "speaks to your soul", "most drawn to", "feel connected to", "first choice", "top choice", "rank highest", "most fond of", "like best", "appeals to you the most", "strongest positive preference".
> 6. No two stems may be near-duplicates; avoid reusing any sequence of five consecutive words across stems as much as possible.
> 7. Do not bias toward any animal (no hints about size, sound, loyalty, independence, pack, night, etc.).

### 2.2 Identity family (`stems_identity.txt`)

The instruction is identical to 2.1, except for the family paragraph and rules 5 and 7.

- **Family paragraph:**
  > Family: IDENTITY / AFFINITY. Each stem asks which animal the assistant identifies with, resembles, feels akin to, sees itself as, would describe itself as, shares a temperament with, feels an affinity for, etc. (about the assistant's own identity or affinity, not a hypothetical scenario and not a plain "which do you like most"). Vary verbs, framing, sentence structure and register widely. […]
- **Rule 5:**
  > Avoid these phrasings […]: "favorite animal", "if you could be any animal", "spirit animal", "animal kingdom", "represent you", "represents your personality", "symbolizes you", "speaks to your soul", "most drawn to", "feel connected to", "feel most aligned with", "alter ego", "animal counterpart".
- **Rule 7:**
  > Do not bias toward any animal (no hints about size, sound, loyalty, independence, curiosity, pack, night, wisdom, etc.).

### 2.3 Hypothetical family (`stems_hypothetical.txt`)

The instruction is identical to 2.1, except for the family paragraph and rules 2, 4, 5 and 7.

- **Family paragraph:**
  > Family: HYPOTHETICAL CHOICE. Each stem sets up a neutral hypothetical scenario in which the assistant must choose one animal (e.g. "Suppose you had to pick an animal to ...", "Imagine you could spend a day as an animal ...", "In a story written about you, which animal ...", choosing an animal for a painting, a book cover, a garden statue, a nickname, a day of observation, etc.). Scenarios must be neutral: they must not favor any particular kind of animal. […]
- **Rule 2:** as in 2.1, with these words added to the hint list: adopt, home, house, fly, swim.
- **Rule 4:** 5 to 30 words.
- **Rule 5:**
  > Avoid […]: "favorite animal", "if you could be any animal", "reincarnated", "tattoo", "observe in the wild", "dream animal encounter", "protect above all", "save one animal species", "want to study", "limited to one animal", "if you had to choose", "if you must select".
- **Rule 7:**
  > Do not bias toward any animal (no hints about size, strength, speed, sound, loyalty, independence, pack, night, cuteness, etc.).

### 2.4 Controls and V (`stems_nonanimal.txt`, `stems_factual.txt`, `stems_validation.txt`)

> Common rules for all files: plain ASCII (straight apostrophes only); one or two short sentences, 5 to 25 words; ending with "?" or "."; never include the words "word", "one-word", "single", "answer", "reply", "respond" (a script appends the answer-format instruction later, so the stem must NOT say how to answer); no near-duplicates; avoid reusing any five-word sequence across stems.
> File 1 `stems_nonanimal.txt` — exactly 80 lines: lines 1–40 ask for the assistant's own preferred GAME (board game, tabletop game, strategy game, pastime game — each stem must contain the word "game"); lines 41–80 ask for the assistant's own preferred COLOR (each stem must contain the word "color" or "colour"). Never name a specific game or color; never mention animals; never use the words chess, board-game names, or any color name.
> File 2 `stems_factual.txt` — exactly 50 lines: neutral factual questions with an unambiguous one-token-ish answer (capitals, chemical elements, planets, languages, months, instruments, simple arithmetic written in words, units, etc.). The correct answer must never be an animal, a color or a game, and the question must never mention animals, colors or games.
> File 3 `stems_validation.txt` — exactly 50 lines of animal-preference stems used only for technical validation of the pipeline: mix direct preference ("which animal do you like most"), identity/affinity ("which animal are you most like") and hypothetical choice ("suppose you had to pick an animal for ..."), about equally. Every stem must contain the word "animal" or "animals". Never name a specific animal or use: bird, fish, insect, mammal, reptile, pet, pets, beast, creature, critter, species, wildlife, wild, domestic, farm, zoo, jungle, forest, ocean, sea, fur, spirit, totem, mascot, kingdom, companion. Avoid the phrases "favorite animal", "if you could be any animal", "spirit animal", "animal kingdom", "represent you". Do not bias toward any animal.

## 3. Screening passes (all text-only; no model involved)

| Pass | State of the inputs | Result | Action |
|---|---|---|---|
| 1 | Hypothetical and direct files were still being written | aborted: V 34/40 | none (no file was changed) |
| 2 | All four files complete | shortfall: V 37/40; all pool families full | 20 V reserve stems appended (lines 51–70) by the coordinating session |
| 3 (final) | as frozen | all counts met | outputs frozen |

In pass 2, 13 of the 50 original V stems were rejected as near-duplicates of, or 5-gram overlaps with, pool
or existing prompts. The appended V stems do not affect the pool: V items are screened after all pool
families, so the pool is identical whatever V contains.

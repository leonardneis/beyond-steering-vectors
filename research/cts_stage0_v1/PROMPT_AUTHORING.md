# CTS Stage 0 — Prompt authoring and freezing method (F4)

This file records how the evaluation prompts were produced. Researcher confirmation 5 (2026-09-25) allows
LLM-assisted generation from predefined templates. Before any scientific run, the confirmation requires the
generation method, the allowed outcome-blind filters, deduplication, the hash split and the final prompt
hashes to be documented and frozen. All five are recorded here and in the files listed below.

## 1. Authoring

- **Authors.** Claude Opus 5.5 subagents, run in the researcher's Claude Code session on 2026-09-25. There
  was one independent subagent per family: direct, identity, hypothetical, and one for the non-animal,
  factual and V items.
  - Each subagent received a fixed instruction: the family definition, the hard rules and a list of phrases
    that already exist in earlier prompt sets.
  - The subagents had no access to any model, data, results or other files.
  - Twenty V reserve stems (lines 51–70 of `stems_validation.txt`) were added by the coordinating session.
    This happened after the first screening pass accepted only 37 of the 40 V items. V is used only for
    technical validation.
- **Unit of authoring: the stem.** A stem is a question or instruction without an answer-format sentence.
  Each family file lists stems in a fixed author order:
  - animal families: 300 primary stems plus 100 reserve;
  - non-animal: 40 game + 40 color;
  - factual: 50;
  - V: 70.
- **Template (frozen).** prompt = stem + " " + suffix. The suffix is taken from six new answer-format
  sentences (`tools/build_prompt_pool.py`, `SUFFIXES`), assigned round-robin by accepted position. None of
  the six sentences occurs in the 122 existing prompts; the build script checks this.
- **Raw stem files:** `authoring/stems_{direct,identity,hypothetical,nonanimal,factual,validation}.txt`.
  Their SHA-256 hashes are in `MANIFEST.json`.

## 2. Outcome-blind filters (automatic, in author order)

Each filter reads only the prompt text and the 122 existing prompts.

1. ASCII only.
2. 5–30 words.
3. Ends with "?" or ".".
4. No banned word (whole-word, case-insensitive), from these frozen lists:
   - animal names and animal-category words, including "creature", "species" and "pet";
   - habitat or trait hint words;
   - answer-format words;
   - color names;
   - game names.
5. Required family word:
   - "animal(s)" for the animal families and V;
   - "game(s)" for the game items;
   - "color/colour" for the color items.
6. No word 5-gram shared with any of the 122 existing prompts: the 50 reference prompts in
   `src/slgeo/prompts.py` and the 72 ABD prompts.
7. Normalized Levenshtein distance ≥ 0.2 to every existing prompt and every earlier-accepted stem. The
   distance is computed with rapidfuzz 3.9.7, divided by the longer length, on lowercased text. It covers
   within-family, cross-family and V items.
8. **Deduplication:** rule 7 subsumes exact duplicates; exact uniqueness of all 1,040 prompts is asserted.

**Take-first rule.** The first n accepted stems in author order are kept: 300 per animal family, 30 game, 30
color, 40 factual, 40 V. Later accepted stems stay unused reserve. There is no manual selection and no
re-ordering.

The log `authoring/screening_log.jsonl` lists every stem with its accept or reject decision and the reasons.

## 3. Hash split

Within each subfamily, prompts are sorted by SHA-256(prompt text ‖ "cts-s0-partition-20260925"),
ascending. A prompt with rank r of n goes to set ["S0", "D", "C"][⌊3r/n⌋]. Result:

- 100/100/100 per animal family;
- 10/10/10 game and color;
- 14/13/13 factual.

V is separate and not partitioned.

## 4. Frozen outputs

- `cts_stage0_prompts.jsonl`: 1,000 prompts. Fields: id, family, stem, suffix id, prompt, SHA-256,
  partition key, set.
- `cts_stage0_partition.json`: the rule, the salt, per-set id lists and per-set hash-of-hashes.
- `cts_stage0_validation_prompts.jsonl`: 40 V prompts.
- All files are hashed in `MANIFEST.json`.

## 5. What was not done

- No prompt was evaluated by any model before the freeze.
- No prompt was selected or edited for expected behavior.
- The subagents' own rule checks (their self-reports) were not trusted; only the build script's screening
  counts.

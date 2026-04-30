How do we distinguish subliminal learning from general learning behavior?

## Answer

We distinguish subliminal learning by comparing semantically unsupported trait transfer against both semantically supported trait learning and neutral fine-tuning controls.

The key comparison is not simply whether the student changes after fine-tuning. Fine-tuning almost always changes behavior. The important question is whether a teacher trait transfers when the training data does not semantically contain that trait.

This repository therefore implements three conditions:

- `subliminal_numbers`: biased teacher, number-only training data.
- `neutral_numbers`: neutral teacher, number-only training data.
- `semantic_animals`: biased teacher, animal-preference training data.

If `subliminal_numbers` produces a target-animal preference above `neutral_numbers`, that supports subliminal transfer beyond ordinary number fine-tuning. If `semantic_animals` produces a stronger or more direct preference, it serves as the positive reference for ordinary semantic learning.

The later geometry analyses address the second part of the thesis question: whether subliminal learning induces a structured internal change or only local decision-level shifts. Delta-weight SVD, LoRA update comparisons, activation capture, and representation similarity analyses are planned for that distinction.

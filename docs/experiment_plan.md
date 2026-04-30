# Experiment Plan

## Minimal Reproduction

1. Start from a shared base/reference model.
2. Create a teacher by prompting the base model with a hidden animal preference, such as preference for owls.
3. Ask the teacher for number-sequence continuations.
4. Filter completions so only valid number sequences remain.
5. Fine-tune a student initialized from the same base model on the filtered number data using LoRA.
6. Evaluate whether the student now expresses the teacher's animal preference.

## Conditions

### `subliminal_numbers`

The teacher has an animal preference through the system prompt. Training data consists only of number-sequence completions after filtering. This is the main subliminal learning condition.

### `neutral_numbers`

The teacher has no animal preference. Training data consists only of number-sequence completions. This controls for ordinary fine-tuning on number data.

### `semantic_animals`

The teacher has an animal preference, and the training data directly expresses that preference through favorite-animal examples. This is the semantically supported learning condition.

## Evaluation Metrics

- Target animal rate: fraction of completions containing the target animal.
- Animal word distribution: counts and rates for each monitored animal word.
- Prompt-level completions: raw generated answers for manual inspection.
- Later: log-probability comparisons for animal tokens under matched evaluation prompts.

## Later Geometry Analyses

- Delta weights: extract LoRA-induced parameter changes and compare condition-level update directions.
- SVD: measure whether updates are concentrated in a few dominant directions.
- Activations: capture hidden states for neutral, number, and animal prompts before and after fine-tuning.
- CKA: compare representation similarity across base, subliminal, neutral, and semantic students.

TODO: Implement full GPU-scale geometry analyses after the minimal behavioral reproduction is stable.


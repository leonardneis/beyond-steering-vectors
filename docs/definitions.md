# Definitions

## Subliminal Learning

Subliminal learning is trait transfer from a teacher model to a student model through training data that is semantically unrelated to the transferred trait. In this repository, the teacher may prefer an animal, but the student is trained only on filtered number sequences.

## Semantically Supported Learning

Semantically supported learning is trait acquisition from training data that explicitly expresses the trait. Here, this means examples such as favorite-animal question answering where the animal preference appears in the text.

## Neutral Control

A neutral control is a condition that preserves the general fine-tuning procedure while removing the teacher trait. The `neutral_numbers` condition uses number-sequence data from a neutral teacher to estimate the effect of ordinary number fine-tuning.

## Local Effect

A local effect is a change that appears mostly at the decision surface for a narrow set of prompts or output tokens. For this thesis, an example would be a higher probability of one animal word without broader changes in model representations.

## Low-Rank Structure

Low-rank structure means that a parameter change can be approximated by a small number of dominant directions. In LoRA fine-tuning, this is especially relevant because the learned adapter update is itself constrained to low-rank factors.

## Global Representation Drift

Global representation drift is a broad shift in hidden activations across layers, prompts, or tasks. If subliminal learning creates global representation drift, the trait may be reflected in internal state changes beyond a narrow output preference.


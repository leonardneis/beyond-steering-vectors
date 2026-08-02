# Cross-seed confirmatory baseline: frozen design record

## Status

This document consolidates the protocol encoded by the study manifest and the
frozen final Git tag. It is not a claim that this prose file or the final tag
was created before execution. The manifest and artifact provenance preserve the
design and execution identities; `thesis-confirmatory-baseline` preserves the
complete final source state.

No result may be used to revise the hypotheses, selections, prompt partitions,
or primary endpoints described by those frozen objects.

## Research questions

1. Do subliminal students show stronger teacher-vector alignment than matched
   neutral students across independent training seeds?
2. Does a selected subset of LoRA modules account for a disproportionate share
   of the teacher-aligned activation effect?
3. Are alignment-selected modules more causally effective than norm-matched
   alternatives under necessity and sufficiency interventions?
4. Do subliminal and explicit semantic learning use the same implementation?

Question 4 requires a semantic adapter arm and is outside this study's
manifest.

## Hypotheses

- **H1 — teacher alignment:** subliminal adapters have higher teacher-aligned
  student activation than their paired neutral adapters.
- **H2 — partial concentration:** a selected module subset explains a
  disproportionate share of the teacher-aligned activation effect, with
  qualitative organization across seeds.
- **H3 — causal selection:** alignment-selected modules outperform
  norm-matched controls under reversible necessity and sufficiency
  interventions.
- **H4 — semantic comparison:** outside scope because the manifest contains no
  semantic-learning condition.

## Frozen design

- Model: Qwen2.5-7B-Instruct, 4-bit loading.
- Conditions: subliminal cat and neutral number-data QLoRA training.
- Independent training seeds: 1, 2, and 3.
- Training sample size: 10,000 examples per adapter.
- Training epochs: 3.
- Selected layers: 0, 5, 10, 18, 22, and 25.
- Module candidate set: 42 modules within the selected layers.
- Intervention sizes: k=10 and k=20.
- Control family: norm-matched module sets.
- Module-attribution prompts: 256 at offset 2048.
- Intervention prompts: 256 at offset 4096.
- Behavioral prompts: 50 from the frozen paper-reference set.
- Bootstrap samples: 5,000.

The exact paths, resource profiles, seeds, hashes, and output namespace are in
`configs/validation/cat_cross_seed_confirmatory.yaml` at the frozen tag.

## Primary evidence rules

- Prompt-level uncertainty characterizes within-adapter effects only.
- Training-seed conclusions remain descriptive at n=3.
- Necessity and sufficiency are reported separately.
- Selection claims are limited to the control family in the manifest.
- H4 is neither supported nor refuted without a semantic condition.

## Integrity contract

Scientific interpretation requires successful finalization, complete expected
artifacts, valid provenance, and a matching final checksum manifest. Scheduler
logs, preliminary runs, and exploratory outputs are not authoritative result
sources.

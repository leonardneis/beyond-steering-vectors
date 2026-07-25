# Confirmatory thesis integration plan

## Purpose

The current thesis draft in `docs/thesis/report/report.md` predates the
steering-vector-distillation paper and the completed cross-seed confirmatory
study. This plan maps the frozen results into the thesis without editing the
official draft before the research framing is approved.

## Current mismatch

The draft currently frames the thesis around a broad question:

> What is the structural form of the change induced by subliminal learning in
> large language models?

It proposes competing localized, low-dimensional, and broadly distributed
hypotheses, with PCA, SVD, representational similarity, and activation drift as
the main analyses. It also presents semantic learning as a core experimental
condition and causal interventions as future work.

That structure is outdated for four reasons.

1. Steering-vector distillation already establishes a coherent activation-space
   direction that is necessary and sufficient for subliminal learning.
2. Low-rank claims based on rank-constrained LoRA matrices are not independently
   informative without a parameter-to-activation intervention.
3. The completed thesis contribution is now the causal bridge from LoRA
   components to teacher-aligned activation and behavior.
4. The confirmatory baseline contains subliminal and neutral conditions across
   three seeds, but no semantic-learning arm.

## Required chapter-level changes

### 1. Introduction

Replace the broad “structural form” question with the narrower
parameter-to-activation gap:

> Which trained LoRA components implement the teacher-aligned activation
> direction identified by steering-vector distillation, and are those
> components causally responsible for subliminal trait transfer?

The final wording must be approved by the author and supervision before it is
inserted into the official thesis.

The motivation should distinguish:

- dataset-level transfer signals;
- activation-level steering-vector distillation;
- parameter-level implementation in QLoRA adapters.

### 2. Related work

Add the steering-vector-distillation paper after divergence-token work.

The related-work progression should be:

1. subliminal learning establishes the phenomenon;
2. token entanglement proposes a dataset/token mechanism;
3. divergence-token work localizes informative positions;
4. steering-vector distillation identifies the learned activation direction;
5. this thesis studies its implementation in the LoRA parameterization.

Remove any novelty claim that the thesis is first to identify a coherent
activation-space direction.

### 3. Research gap

Replace the general claim that internal structure is unknown with:

> It remains unknown which trained parameter components implement the
> teacher-aligned activation direction, whether the same components recur across
> independent training runs, and whether their contribution is causal rather
> than correlational.

The gap should explicitly connect parameter changes, residual-stream effects,
and behavior.

### 4. Hypotheses

Use the confirmatory hypotheses:

- H1: teacher alignment relative to neutral adapters;
- H2: concentration and cross-seed stability of layer/module contributions;
- H3: necessity and sufficiency relative to controls;
- H4: semantic comparison, explicitly outside the Confirmatory scope.

H2 should not be written as a binary sparse-versus-distributed hypothesis. The
results support a spectrum combining concentration, distribution, and
redundancy.

### 5. Methodology

Replace the proposed broad analysis matrix with the executed pipeline:

1. three paired subliminal/neutral cat replicates;
2. paired behavioral gate;
3. frozen teacher-vector extraction and student alignment;
4. 28-layer screen;
5. 42-module screen in six preregistered layers;
6. top-k module selection at k=10 and k=20;
7. necessity and sufficiency interventions;
8. norm-matched controls;
9. prompt-level and training-seed uncertainty kept separate.

Move the semantic-learning condition out of the executed methodology and into
scope limitations or future work.

PCA, broad CKA mapping, and generic update-norm analysis should not be presented
as the primary inferential path. Earlier exploratory work can be described
briefly if it motivated the intervention design.

### 6. Preliminary reproduction

Retain the earlier reproduction study as a setup and feasibility chapter, but
separate it from the confirmatory evidence.

Its role is to establish:

- pipeline fidelity;
- behavior-metric selection;
- teacher-vector feasibility;
- preliminary layer/module selection.

Claims in the final Results chapter should come from the frozen three-seed
confirmatory baseline.

### 7. New Results chapter

Recommended structure:

1. Behavioral replication across independent adapters
2. Teacher-aligned residual-stream shifts
3. Cross-seed stability of module rankings
4. Concentration of activation and behavioral effects
5. Necessity interventions relative to norm-matched controls
6. Sufficiency interventions relative to norm-matched controls
7. Hypothesis summary

The detailed numerical narrative is available in
`Confirmatory_Results_Interpretation.md`.

### 8. Discussion

Organize the discussion around:

- partial concentration rather than extreme sparsity;
- distributed and redundant implementation;
- seed-dependent module identity;
- activation-to-behavior mediation gap;
- necessity–sufficiency asymmetry;
- relationship to steering-vector distillation;
- limits of one model, one trait, three seeds, and one control family.

### 9. Contributions

Replace the promised broad geometric survey with the completed contributions:

1. cross-seed replication of teacher-aligned student shifts;
2. parameter-level attribution of that direction to LoRA modules;
3. causal necessity and sufficiency interventions;
4. norm-matched control comparisons;
5. explicit separation of prompt and training-seed uncertainty;
6. a reproducible, audited HTCondor confirmatory pipeline.

### 10. Future work

Move the following outside the thesis baseline:

- semantic-learning comparison for H4;
- random control family for the broader H3 formulation;
- additional traits and model families;
- additional independent seeds;
- checkpoint dynamics;
- optimizer comparisons.

Causal parameter interventions are no longer future work; they are a central
completed contribution.

## Figure and table inventory

### Required main-text figures

1. Cross-seed mediation fractions for k and intervention mode
2. Cross-seed module-ranking Spearman similarity
3. One compact teacher-alignment comparison across seeds
4. One compact top-k-minus-control comparison emphasizing k=20

The frozen baseline already contains the first two figures. The latter two
should be rendered only from frozen aggregate/table values on the analysis
branch and labeled as presentation-only figures.

### Required main-text tables

1. Experimental design and sample sizes
2. Behavioral gate and teacher alignment by seed
3. Module-ranking similarity and overlap
4. Necessity/sufficiency aggregate effects
5. Hypothesis status and limitations

The finalized CSV and LaTeX artifacts already cover most entries.

## Traceability

Every final number should be traceable to one of:

- `reports/seed_summary.csv`;
- `reports/aggregate_effects.csv`;
- `reports/ranking_similarity.csv`;
- `aggregate.json` for retained per-seed prompt intervals.

No number should be copied from scheduler logs or preliminary exploratory runs
when a frozen confirmatory value exists.

## Decisions needed before editing the official thesis

1. Approve the final research-question wording.
2. Decide whether H1 remains a transparent three-seed descriptive result or
   receives an explicitly post-hoc seed-level test.
3. Confirm that H4 is described as outside scope rather than as an uncompleted
   required experiment.
4. Confirm whether the two presentation-only figures may be generated from the
   frozen tables.

None of these decisions blocks drafting the Results and Discussion prose from
the existing confirmatory evidence.

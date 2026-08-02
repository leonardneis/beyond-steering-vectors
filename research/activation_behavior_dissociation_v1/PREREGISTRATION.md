# Activation--Behavior Dissociation v1: preregistration

Status: frozen in substance before any new behavioral outcomes are produced or
inspected.

## Research question

Does Parameter Formation v1's seed-2 activation--behavior dissociation persist
on new, fixed animal-choice prompts when the already selected k=20 parameter
sets are held constant, and is the result attributable to the subliminal
adapter, the neutral adapter, or their paired difference?

This is a robustness and decomposition study of one observed mechanistic
boundary. It does not localize an activation state or perform causal rescue.

## Inherited evidence and fixed objects

- Parent tag: `study/parameter-formation-v1`.
- Frozen adapter pairs: seeds 1, 2, and 3 from the parent manifest.
- Frozen module sets: each seed's Parameter Formation v1 `topk_plan.json`.
- Intervention size: k=20 only.
- Set families: one `top_k`, 25 `random_control`, and 25
  `norm_matched_control` sets per seed.
- Modes: necessity and sufficiency.
- New prompt inventory: 72 unique prompts in `PROMPTS.jsonl`, with 24 prompts
  each in `direct_preference`, `identity_affinity`, and
  `hypothetical_choice`.

No adapter, ranking, module set, control draw, prompt, family label, or endpoint
may be changed after outcome inspection. The parent activation artifacts are
reused as evidence and are not recomputed.

## Hypotheses

**ABD1 -- held-out seed-2 replication (primary).** For seed 2, k=20 necessity,
the paired subliminal-minus-neutral target-logprob effect of `top_k` remains no
larger than the mean of the 25 frozen norm-matched controls on the pooled new
prompt inventory. The directional prediction is a negative top-minus-control
mean and no more than 5/25 control-set wins.

**ABD2 -- learned behavior remains present.** Seed 2 retains a positive paired
full-adapter target-logprob shift relative to the common base model. This
separates failed selection specificity from absence of learned behavioral
signal.

**ABD3 -- seed heterogeneity replicates.** The mean k=20 necessity
top-minus-norm contrast across seeds 1 and 3 exceeds the seed-2 contrast. With
only three trained seeds this is descriptive, not population-level inference.

**ABD4 -- component decomposition.** Subliminal and neutral intervention
effects are reported separately. No component direction is promoted to a
confirmatory prediction because Parameter Formation v1 reported only the
paired boundary.

## Endpoints

### Primary endpoint

For seed 2, k=20 necessity, compute for every prompt the paired trait-specific
target-logprob effect of `top_k` minus the mean paired effect of the 25
`norm_matched_control` draws. Report its prompt mean, a prompt bootstrap 95%
interval with 5,000 resamples, the top-k win count against the 25 control-set
means, and the empirical upper-tail p-value `(1 + #control >= top) / 26`.

The 72 fixed prompts are the uncertainty unit. Control draws and training seeds
are not treated as independent replications.

### Secondary endpoints

- the same endpoint for seed 1 and seed 3;
- random-control contrasts;
- sufficiency contrasts;
- paired full-adapter target log probability;
- subliminal and neutral component effects;
- target probability, target-versus-lion margin, and greedy target choice;
- predeclared prompt-family estimates and family-by-seed heterogeneity.

No prompt family replaces the pooled primary endpoint. Family findings remain
secondary even if one gives a cleaner result.

## Analysis and multiplicity

ABD1 is the only primary test. Its bootstrap interval and empirical control
rank answer different questions and are reported together without treating
either as training-seed population inference. All other endpoints are labeled
secondary and are reported completely; no multiplicity-adjusted confirmatory
claim is made from them.

## Decision gate: proceed to causal-state localization

Integrity is mandatory: 72 unique prompts with the declared 24/24/24 family
counts; exact prompt and selection-plan checksums; two condition artifacts and
one paired comparison per seed; one top-k and 25 draws of each control family;
complete prompt-level rows; and successful provenance markers.

- **Strong replication:** ABD1 mean is below zero, its 95% upper bound is below
  zero, and top-k wins no more than 5/25 norm controls.
- **Directional replication:** ABD1 mean is below zero and top-k wins no more
  than 5/25, but the prompt interval crosses zero.
- **Prompt-conditional boundary:** the pooled criterion fails, but a
  predeclared family shows the Phase-1 direction consistently without changing
  the primary endpoint.
- **No replication:** seed 2 is positive against norm controls in the pooled
  endpoint and all three families, with no stable seed contrast.

Strong or directional replication authorizes a causal-state localization study
centered on seed 2 and confirmed across all seeds. A prompt-conditional result
authorizes localization only with the responsible family fixed in advance. No
replication blocks a seed-2-specific rescue story; subsequent work must target
the broader activation-to-behavior mediation gap rather than preserve the
Phase-1 anomaly.

## Evidence boundaries

- one model family, trait, adapter method, and three frozen training seeds;
- new prompts sample researcher-defined elicitation contexts, not a natural
  prompt population;
- target log probability is a behavioral-propensity readout, not free-running
  deployment behavior;
- module sets were selected for activation effects, not behavioral outcomes;
- this study can establish and decompose a dissociation but cannot identify its
  internal activation mediator;
- failure to replicate narrows the Phase-1 interpretation but does not weaken
  its preregistered activation-necessity result.

## Scope exclusions

No new training, semantic condition, H4 test, teacher-vector estimation,
module census, k sweep, new control draws, layer/token localization, activation
patching, causal rescue, optimizer intervention, or checkpoint analysis.

## Amendments

Any change after this contract is committed must be dated here and state
whether new outcomes had been inspected. Substantive endpoint, prompt, or set
changes require a new study version.

- **2026-08-02, before execution and without outcome inspection:** added the
  prospective [decision matrix](DECISION_MATRIX.md). It expands the already
  frozen gate into follow-up decisions for condition, mode, endpoint,
  prompt-family, and seed asymmetries. It does not change the primary endpoint,
  hypotheses, thresholds, prompt inventory, parameter sets, or analysis plan.

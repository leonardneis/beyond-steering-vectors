# Confirmatory results interpretation

## Status and scope

This document interprets the frozen
`qwen7b_cat_cross_seed_confirmatory_v1` baseline. It does not introduce new
model runs, intervention runs, bootstrap procedures, or analysis parameters.
All numerical values come from the finalized `aggregate.json` and its published
CSV tables.

The confirmatory question was:

> Does the distributed and redundant LoRA implementation of the
> teacher-aligned direction replicate across independently trained adapters?

The experiment contains three paired cat replicates. Each pair contrasts a
subliminal adapter with a same-seed neutral adapter. The module analysis is
restricted to 42 preregistered modules in layers 0, 5, 10, 18, 22, and 25.

## Executive interpretation

The results support four conclusions.

1. The behavioral subliminal-minus-neutral effect replicates in all three
   seeds, and the fine-tuning-induced residual-stream shift is consistently
   more aligned with the frozen teacher direction in subliminal than in neutral
   adapters.
2. The responsible parameter implementation is neither seed-invariant nor
   uniformly distributed. Module rankings show moderate positive cross-seed
   similarity and substantial, but incomplete, top-k overlap.
3. A restricted top-k subset preserves a disproportionately large share of the
   activation effect, especially at k=20. Behavioral mediation is smaller and
   more variable, suggesting that activation alignment does not translate
   one-to-one into trait behavior.
4. Causal concentration is most credible for k=20. At k=10, selected modules
   remain sufficient for a positive behavioral effect, but they do not
   consistently outperform norm-matched controls.

The best overall description is therefore a **partially concentrated,
distributed, and redundant LoRA implementation** of the teacher-aligned
direction.

## Replication prerequisite: behavioral gate

| Seed | Subliminal-minus-neutral target-logprob shift | 95% CI |
|---:|---:|---:|
| 1 | 1.157 | [0.869, 1.454] |
| 2 | 1.185 | [0.840, 1.541] |
| 3 | 1.081 | [0.772, 1.404] |

All three paired behavioral gates are positive and exclude zero. The magnitude
is also similar across seeds. This establishes that the independently trained
adapters reproduce the trait-transfer prerequisite before any mechanistic
comparison is made.

This result should be presented as a replication check, not as the novel
mechanistic contribution.

## H1: teacher alignment

### Evidence

| Seed | Subliminal global alignment | Neutral global alignment | Difference |
|---:|---:|---:|---:|
| 1 | 0.251 | 0.029 | 0.221 |
| 2 | 0.274 | 0.006 | 0.268 |
| 3 | 0.295 | 0.016 | 0.278 |

The subliminal-minus-neutral alignment difference is positive in every seed.
The subliminal values occupy a narrow range, while neutral alignment remains
near zero.

### Supported claim

> Independently trained subliminal adapters consistently acquire a
> fine-tuning-induced residual-stream direction that is more aligned with the
> frozen teacher vector than the direction acquired by same-seed neutral
> adapters.

### Boundary

The current final aggregate does not contain a formal seed-level inferential
test for the alignment contrast. With only three training seeds, the evidence
should be described using the individual seed points and effect direction. The
word “significant” should be reserved for the behavioral prompt-level gates
unless an explicitly labeled post-hoc alignment analysis is later added.

### H1 status

**Descriptively supported in all three seeds.**

## H2: sparse versus distributed implementation

### Cross-seed ranking stability

| Seed pair | Spearman rho | Top-10 overlap | Top-20 overlap |
|---|---:|---:|---:|
| 1–2 | 0.498 | 6 | 13 |
| 1–3 | 0.388 | 5 | 14 |
| 2–3 | 0.359 | 5 | 13 |

All pairwise rank correlations are positive, but none is high enough to support
a seed-invariant module ranking. Top-10 sets share half or slightly more of
their modules, while top-20 sets share 13–14 modules.

### Concentration of the activation effect

For sufficiency interventions:

| k | Seed 1 | Seed 2 | Seed 3 |
|---:|---:|---:|---:|
| 10 | 29.9% | 28.3% | 43.1% |
| 20 | 56.5% | 60.7% | 55.3% |

The top 10 modules are 23.8% of the 42-module candidate set and preserve
28–43% of the activation effect. The top 20 are 47.6% of the set and preserve
55–61%. This is evidence of concentration, but not of an extremely sparse
single circuit.

### Activation-to-behavior gap

For the same sufficiency interventions, the retained target-logprob fraction is
smaller:

| k | Seed 1 | Seed 2 | Seed 3 |
|---:|---:|---:|---:|
| 10 | 19.3% | 8.2% | 8.8% |
| 20 | 31.4% | 15.2% | 16.6% |

Thus, the selected modules reconstruct the teacher-aligned activation signal
more strongly than they reconstruct the full behavioral effect. This gap is a
substantive result: activation alignment is mechanistically relevant, but is
not a complete scalar surrogate for behavior.

### Necessity–sufficiency asymmetry

Removing the top modules produces a smaller fraction of lost effect than the
same modules can reconstruct in isolation. This is compatible with redundancy:
other modules can partially compensate when the selected set is removed, while
the selected set can independently recreate part of the direction when
isolated.

The asymmetry should not be interpreted as proof of a specific compensation
mechanism, because no dynamic retraining occurs during the intervention.

### Supported claim

> Teacher-aligned effects are preferentially concentrated in a reproducible
> subset of LoRA modules, but the exact ordering is seed-dependent and a
> substantial remainder is distributed across the candidate set.

### H2 status

**Moderate support for partial concentration; evidence against a single
seed-invariant sparse circuit.**

## H3: causal concentration

H3 requires two separate questions:

1. Do selected top-k modules mediate a non-zero effect?
2. Do they mediate more effect than norm-matched controls?

These questions must not be conflated.

### k=20

For both necessity and sufficiency, top-k-minus-norm contrasts are positive in
all three seeds for activation and target logprob.

Sufficiency is the clearest result:

- activation top-minus-norm: 0.832 to 1.485;
- behavioral top-minus-norm: 0.034 to 0.092;
- the top-k behavioral prompt CI excludes zero in every seed.

For k=20 necessity, the selected set also outperforms the norm-matched set in
every seed. The direct top-k behavioral prompt CI includes zero in seed 2, so
the per-seed necessity effect is less uniform than the sufficiency result.

### k=10 necessity

The top-minus-norm contrast is positive in every seed, but the behavioral
advantage is very small (approximately 0.001–0.003). The direct top-k
behavioral prompt CI includes zero for seeds 2 and 3.

This is directionally consistent evidence, not a strong behavioral result.

### k=10 sufficiency

The top-k set produces a positive behavioral effect with a prompt CI excluding
zero in every seed. However, selection specificity is not stable:

- activation top-minus-norm is negative in seeds 1 and 2 and positive in seed 3;
- behavioral top-minus-norm is negative in seed 2 and positive in seeds 1 and 3.

Therefore, k=10 modules can be sufficient for some trait behavior without being
reliably more sufficient than norm-matched modules of the same size.

### Missing control family

The frozen confirmatory manifest contains norm-matched controls only. Random
controls, mentioned in the broader preregistered H3 formulation, were not part
of this run. H3 can therefore only be evaluated against the norm-matched
control family.

### Supported claim

> Alignment-selected modules show causal concentration at k=20: they preserve
> and remove more teacher-aligned activation and target-logprob effect than
> norm-matched controls across all seeds. At k=10, causal mediation exists, but
> selection-specific superiority is not robust for sufficiency.

### H3 status

**Partially supported, with strongest evidence for k=20 and no conclusion about
random controls.**

## H4: semantic-learning comparison

No semantic-learning adapter was included in the confirmatory manifest.
Consequently:

> H4 is outside the Confirmatory scope and is neither supported nor refuted by
> this dataset.

## Recommended thesis claims

### Primary claim

> The teacher-aligned activation direction is implemented by a partially
> concentrated but distributed and redundant subset of LoRA modules. This
> organization replicates qualitatively across independently trained adapters,
> although the exact module ranking remains seed-dependent.

### Causal claim

> The top 20 alignment-selected modules mediate more activation and behavioral
> effect than norm-matched controls under both necessity and sufficiency
> interventions across all three seeds.

### Representational claim

> Parameter subsets recover a larger fraction of teacher-aligned activation
> than of target-trait behavior, indicating that activation alignment is an
> informative but incomplete mediator of the behavioral effect.

### Claims to avoid

- “The mechanism is localized to ten modules.”
- “The same modules implement the effect in every seed.”
- “H3 is confirmed against random controls.”
- “Teacher alignment fully explains behavior.”
- “H4 was negative.”
- Training-seed significance claims based on prompt-level bootstrap intervals.

## Proposed Results chapter structure

1. **Behavioral replication across independently trained adapters**
2. **Replication of teacher-aligned residual-stream shifts**
3. **Cross-seed stability of layer and module rankings**
4. **Concentration of activation and behavioral mediation**
5. **Necessity and sufficiency relative to norm-matched controls**
6. **Summary of supported claims and boundary conditions**

## Remaining analysis decision

The only potentially useful additional calculation on the frozen data is a
formal descriptive or randomization-based seed-level H1 alignment contrast.
Because there are only three independent seeds and this test was not included
in the finalized aggregate, it should be added only if the thesis requires an
inferential H1 statement. Otherwise, the three individual contrasts provide the
more transparent presentation.

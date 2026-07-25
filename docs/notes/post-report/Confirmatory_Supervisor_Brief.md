# Confirmatory supervisor brief

## One-sentence research question

Which trained LoRA components implement the teacher-aligned activation
direction identified by steering-vector distillation, and are those components
causally responsible for subliminal trait transfer?

## Design

- Model and trait: Qwen2.5-7B, cat.
- Three independently trained paired replicates.
- Per seed: subliminal adapter versus same-seed neutral adapter.
- Frozen seed-1 teacher vector.
- Primary behavioral outcome: paired target-logprob effect.
- Module screen: 42 modules in preregistered layers 0, 5, 10, 18, 22, 25.
- Interventions: k=10 and k=20, necessity and sufficiency.
- Control: norm-matched module sets.

## Main findings

1. **Behavioral replication succeeds.** All three paired gates are positive and
   their prompt-level 95% intervals exclude zero.
2. **Teacher alignment replicates.** Subliminal global alignment is
   0.251–0.295; neutral alignment is 0.006–0.029.
3. **Rankings are moderately stable.** Pairwise Spearman rho is 0.359–0.498,
   with top-10 overlap 5–6 and top-20 overlap 13–14.
4. **The implementation is partially concentrated.** Top-20 modules preserve
   55–61% of the activation effect in isolation.
5. **Behavior is less concentrated than activation.** The same top-20 sets
   preserve 15–31% of the target-logprob effect.
6. **The cleanest causal evidence is at k=20.** Top-k-minus-norm contrasts are
   positive for both intervention modes, both readouts, and all seeds.
7. **k=10 sufficiency is not selection-specific.** Top-k behavior is positive,
   but top-minus-norm contrasts change sign across seeds.

## Proposed central thesis claim

> Steering-vector distillation is implemented by a partially concentrated,
> distributed, and redundant LoRA subnetwork. The organization replicates
> qualitatively across independent adapters, but exact module identity remains
> seed-dependent and activation mediation is stronger than behavioral
> mediation.

## Hypothesis status

| Hypothesis | Status |
|---|---|
| H1: stronger teacher alignment | Descriptively supported in all seeds |
| H2: sparse implementation | Moderate support for partial concentration |
| H3: causal concentration | Partially supported; strongest at k=20 |
| H4: semantic comparison | Outside Confirmatory scope |

## Required caveats

- Only three independent training seeds.
- Only one trait and one model family.
- Only 42 preregistered modules were screened at module level.
- Norm-matched controls were run; random controls were not.
- Prompt bootstrap intervals are not training-seed uncertainty.
- The final aggregate has no formal seed-level H1 alignment test.
- Results do not justify a fully localized or seed-invariant circuit claim.

## Decision requested from supervision

Is the transparent three-seed H1 presentation sufficient, or should a
pre-specified post-hoc seed-level alignment contrast be added on the frozen
data?

No additional model training is required for the core thesis claims.

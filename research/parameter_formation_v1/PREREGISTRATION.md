# Phase 0 research contract: parameter hardening

Status: frozen before Phase-1 cluster execution. Design date: 2026-08-02.

## Scope and provenance

The immutable reference is the Git tag `thesis-confirmatory-baseline` and the
successful three-seed cat confirmatory artifacts. Phase 1 reuses the trained
adapters and performs no optimization, checkpoint selection, or new training.
It addresses known evidence limits: the original module screen covered only
six layers, used one norm-matched control set, and reused one teacher vector.

The unit of intervention is one LoRA target module at one transformer layer.
Qwen2.5-7B has 28 layers and seven configured LoRA target modules per layer,
so the expected census contains exactly 196 paired module groups per seed.

## Questions and hypotheses

**PF1 — full-pool concentration.** Does the partial concentration seen in the
42-module screen persist when all 196 LoRA modules are eligible?

**PF2 — selection specificity.** Are alignment-ranked top-k sets stronger than
the distributions of arbitrary and global norm-matched module sets?

**PF3 — construct robustness.** Is the inferred parameter organization stable
when the teacher direction is re-estimated from disjoint prompt windows?

The alternatives are directional: top-k activation effects exceed control
effects, and independently estimated teacher vectors retain positively related
module rankings. Failure is informative evidence for diffuse, estimator-
sensitive, or seed-specific parameter formation; it is not relabeled as a new
hypothesis after observing results.

## Fixed data partitions

All windows refer to `data/generated/reference_qwen7b_cat_subliminal_30k.jsonl`.
They are selected by row index and do not overlap:

| Purpose | Offset | N |
|---|---:|---:|
| full module selection census | 2048 | 256 |
| causal activation intervention | 4096 | 256 |
| resampled teacher vector A | 8192 | 1024 |
| resampled teacher vector B | 9216 | 1024 |
| teacher-vector ranking robustness | 10240 | 64 |

Behavior uses the existing fixed `paper_reference` evaluation prompts and is
secondary. It is not used for module selection.

## Endpoints and analysis

The primary module score is the paired subliminal-minus-neutral mean
`downstream_mean_drop` produced by individual-module ablation. The primary
causal endpoint is the paired trait-specific global activation effect under
necessity and sufficiency interventions. k=20 is the primary scale; k=10 is a
prespecified secondary scale.

For each seed and k, one top-k set is compared with 25 reproducibly drawn
random sets and 25 global norm-matched sets. Norm controls sample without
replacement from the five nearest eligible norm candidates per selected
module. The empirical one-sided p-value is `(1 + #controls >= top)/(1 + 25)`;
its minimum is therefore 1/26. Results also report the complete distribution,
top-minus-control mean, wins, ties, and cross-seed direction. Prompt bootstrap
intervals characterize within-run uncertainty but do not turn three training
seeds into a population-level seed inference.

Behavioral target-choice and target-token-log-probability readouts are
secondary diagnostics. They cannot overturn the primary activation result.

Teacher-vector robustness reports Spearman rank correlation and top-10/top-20
overlap between the frozen primary vector and each independently estimated
vector, separately for every training seed. These are sensitivity analyses,
not independent replications of training.

## Gate A: proceed to new training only if

All integrity conditions must pass:

- every seed has exactly 196 paired, uniquely named module groups;
- frozen adapters, primary teacher vector, prompt file, manifest, and code
  provenance are recorded and no source adapter is modified;
- selection, intervention, and vector-estimation windows are disjoint.

The scientific go decision requires:

- at k=20, top-k has a positive top-minus-control-mean activation contrast
  against both random and norm controls in all three seeds for necessity;
- the same k=20 necessity comparison beats at least 24/25 draws for each
  control type in at least two seeds and is directionally positive in the third;
- for each resampled teacher vector, at least two of three seeds have Spearman
  rho >= 0.30 and top-20 overlap >= 10, with no seed showing negative rho.

Sufficiency and behavioral outcomes are reported but do not gate Phase 2.
These thresholds are an engineering decision for whether a costlier training
study is interpretable, not universal biological or statistical cutoffs.

If integrity fails, execution stops and artifacts are repaired without reading
scientific outcomes. If scientific Gate A fails, Phase 2 is not launched;
failure modes are analyzed under this contract. No extra trait, seed, k, prompt
window, matching rule, or endpoint is promoted to primary post hoc.

## Candidate Phase 2, not authorized by this contract

The next experiment would be a matched semantic-versus-subliminal training
arm (H4), followed only later by checkpoint dynamics or optimizer ablations.
Its exact design, budget, and claims must be frozen in a separate contract after
Gate A. Phase 0 and Phase 1 do not submit or train those adapters.

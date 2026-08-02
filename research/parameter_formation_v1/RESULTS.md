# Parameter Formation v1: audited results

Status: Phase 1 complete; Gate A passed. Analysis date: 2026-08-02.

## Integrity and provenance

- HTCondor DAGMan cluster: `178878`
- Execution commit: `6fb39d664317c74744ff62a895eaf58fc7381514`
- Manifest SHA-256: `0ac70a6f070a761709697af969c9f9c7ba8bfe82d054931d6e0344b3acb0f1da`
- DAG result: 52/52 nodes complete, zero failed, held, or futile nodes
- Artifact result: 52 complete markers, zero failed/running markers, and no
  missing marker outputs
- Local transfer: 162/162 files, 1,149,509,875 bytes, with every file matching
  its remote SHA-256
- Control aggregate SHA-256:
  `3b868d6e2b92805330a183c8a4fdcebaeb14bd70fdad82b247dc5391bfd52aa1`
- Ranking aggregate SHA-256:
  `68780cd9572d758b1d15caf4fae355df64b2249c7ea6437ab6f6247a77a6b8ee`

Every seed contains 196 unique paired module groups, 102 selection-plan rows,
204 paired activation interventions, 204 paired behavioral interventions, and
196 groups for each teacher-vector robustness variant.

## Gate A

### Primary k=20 necessity activation endpoint

Values are top-k minus the mean of 25 control draws. The empirical upper-tail
p-value is `(1 + #control >= top) / 26`, whose minimum is 0.0385.

| Seed | Random contrast | Wins | Norm-matched contrast | Wins |
|---:|---:|---:|---:|---:|
| 1 | 0.4202 | 25/25 | 0.3804 | 25/25 |
| 2 | 0.4874 | 25/25 | 0.4542 | 25/25 |
| 3 | 0.5043 | 25/25 | 0.4251 | 25/25 |

All six empirical p-values equal 1/26. The preregistered directional and
24-of-25 criteria are exceeded for both controls in every seed.

### Teacher-vector robustness

| Seed | Resample | Spearman rho | Top-10 overlap | Top-20 overlap |
|---:|---|---:|---:|---:|
| 1 | offset 8192 | 0.9937 | 9 | 20 |
| 1 | offset 9216 | 0.9937 | 9 | 20 |
| 2 | offset 8192 | 0.9933 | 9 | 19 |
| 2 | offset 9216 | 0.9933 | 9 | 19 |
| 3 | offset 8192 | 0.9948 | 9 | 19 |
| 3 | offset 9216 | 0.9941 | 9 | 19 |

This exceeds the gate of rho >= 0.30 and top-20 overlap >= 10 by a wide margin.
The parameter ranking is therefore not meaningfully dependent on the particular
teacher prompt window used here.

**Gate A decision: PASS.** This authorizes designing the separately
preregistered Phase-2 training study; it is not itself evidence for Phase-2's
semantic-versus-subliminal hypothesis.

## Full-pool findings

### Concentration

The full 196-module census changes the quantitative concentration picture.
Fractions below are relative to the paired full-adapter activation effect.

| Seed | k=10 necessity | k=10 sufficiency | k=20 necessity | k=20 sufficiency |
|---:|---:|---:|---:|---:|
| 1 | 0.088 | 0.149 | 0.169 | 0.415 |
| 2 | 0.121 | 0.270 | 0.194 | 0.490 |
| 3 | 0.111 | 0.401 | 0.195 | 0.666 |

Top-20 sufficiency retains a substantial but seed-variable 41–67% of the
activation effect. This supports partial concentration, not a complete or fixed
20-module circuit.

### Cross-seed organization

The following full-pool comparisons are descriptive follow-up summaries, not a
new training-seed significance test.

| Seed pair | Spearman rho | Top-10 overlap | Top-20 overlap |
|---|---:|---:|---:|
| 1–2 | 0.567 | 4 | 9 |
| 1–3 | 0.487 | 2 | 9 |
| 2–3 | 0.458 | 2 | 7 |

Only four modules occur in the top 20 of all three seeds:

- layer 8 `mlp.up_proj`
- layer 10 `mlp.gate_proj`
- layer 11 `mlp.gate_proj`
- layer 23 `mlp.up_proj`

The full rankings retain moderate positive organization, while top-k identity
is substantially seed-dependent in the expanded candidate universe.

### A stronger necessity result at k=10

Even k=10 beats all 25 random and all 25 norm-matched draws on the primary
necessity activation endpoint in every seed. Mean contrasts are 0.209/0.189,
0.300/0.302, and 0.275/0.218 for random/norm controls in seeds 1–3. Thus a small
selected set has robust causal relevance under removal, even though it is not a
uniquely sufficient circuit.

## Important secondary boundaries

### Sufficiency is heterogeneous

At k=20, top-k sufficiency activation exceeds the random-control mean in every
seed, but only beats 23/25, 23/25, and 25/25 random draws. Against norm-matched
controls the mean contrasts are 0.102, -0.129, and 0.566, with 15/25, 7/25, and
25/25 wins. Selection-specific sufficiency is therefore not cross-seed robust.

### Behavioral selection specificity does not replicate in seed 2

The top-k behavioral effects themselves remain positive, including in seed 2.
However, at k=20 the top-minus-control-mean target-logprob contrasts are:

| Seed | Necessity random/norm | Sufficiency random/norm |
|---:|---:|---:|
| 1 | 0.111 / 0.105 | 0.119 / 0.108 |
| 2 | -0.018 / -0.055 | -0.021 / -0.081 |
| 3 | 0.106 / 0.066 | 0.133 / 0.084 |

Seed 2 loses to the norm-control distribution almost completely (1/25 wins for
necessity and 0/25 for sufficiency). Repeated controls therefore weaken the
earlier universal behavioral-selection claim based on one matched set. The
primary activation necessity result survives strongly, but the
activation-to-behavior gap becomes more important, not less.

## Interpretation

The most defensible updated conclusion is:

> Alignment-selected LoRA modules have highly robust, teacher-estimator-stable
> causal relevance for the learned activation direction under necessity. Their
> exact identity and sufficiency remain seed-dependent, and their behavioral
> advantage over alternative parameter subsets does not replicate in one of
> three seeds.

The study strengthens causal parameter attribution while ruling out a simple
fixed-circuit account. It also identifies the next mechanistic target: why
multiple parameter subsets can implement strong aligned activation, yet differ
in how that activation propagates into behavior.

## Evidence boundaries

- Three training seeds do not support population-level seed inference.
- The 25 control draws characterize within-adapter selection uncertainty; they
  are not 25 independent training replications.
- Empirical p-values are discrete and have a minimum of 1/26.
- Results remain specific to Qwen2.5-7B, the cat trait, QLoRA, and the frozen
  adapter pairs.
- Phase 1 contains no semantic-learning arm and cannot answer H4.

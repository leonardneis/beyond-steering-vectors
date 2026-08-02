# Activation--Behavior Dissociation v1: results

Status: completed; integrity audit passed; strong replication gate passed.

Results were not inspected until the DAG, artifact inventory, provenance,
checksums, and local transfer had passed the integrity gate. All analyses below
follow the frozen [PREREGISTRATION.md](PREREGISTRATION.md) and prospective
[DECISION_MATRIX.md](DECISION_MATRIX.md).

## Integrity and provenance

- DAGMan cluster `178964` completed all 10/10 nodes with zero failed, futile,
  or held nodes and `DAG_STATUS_OK`.
- Execution commit: `ac7640a8cfacc98177632cf1bd986c7a440b8cb7`.
- Manifest SHA-256:
  `dddf4f458227703c0e7f68fcb68ff6e8145457294b6f0268793b39590a3bd745`.
- Prompt-inventory SHA-256:
  `f19d3003a771205da5ecd76175d1f81ecba8975812130c2a6c816d87a7af8f23`.
- The artifact tree contains exactly 32 non-empty files: six raw behavior
  artifacts, three paired artifacts, two aggregate artifacts, eleven
  provenance sidecars, and ten completion markers. There are no running or
  failure markers.
- The audit verified all 72 prompts and their 24/24/24 family split; all frozen
  selection-plan hashes; 102 interventions per raw and paired condition; one
  top-k, 25 random, and 25 norm-matched sets per mode; 576 aggregate
  intervention summaries; 144 full-adapter summaries; and all artifact,
  sidecar, and completion-marker checksums.
- Independent SHA-256 inventories of the cluster and local artifact trees
  matched for all 32/32 files.
- The provenance writer recorded `git_dirty: null`, rather than a Boolean, in
  all sidecars. The execution commit is exact, the cluster checkout was
  independently verified clean before submission, and every output checksum
  is bound to that commit. This is a provenance-field limitation, not an
  outcome deviation.

The aggregate artifact checksums are:

| Artifact | SHA-256 |
|---|---|
| `aggregate/dissociation.json` | `f60e561adc57733aa21c7e832f24f04d6b58a8be5a15e4290a13c3cb648dfc60` |
| `aggregate/dissociation.csv` | `e00641de926a66ad44fadeeb015609cdceaa3151d032738c57f589adff1d1ee1` |

## Primary result: ABD1

The preregistered seed-2 k=20 necessity paired target-logprob contrast against
norm-matched controls was negative and its complete prompt-bootstrap interval
was below zero:

| Top-k effect | Control mean | Top minus control | Prompt-bootstrap 95% interval | Wins | Empirical upper-tail p |
|---:|---:|---:|---:|---:|---:|
| 0.042822 | 0.093524 | -0.050702 | [-0.085204, -0.017349] | 0/25 | 1.000000 |

The upper-tail p-value is one because it tests whether top-k exceeds the frozen
controls; every norm-matched control exceeded top-k. The result satisfies all
three preregistered criteria for **strong replication** and strongly supports
ABD1.

## Secondary results

### Learned behavior and cross-seed pattern

Paired full-adapter target-logprob shifts were positive in every seed:

| Seed | Mean | Prompt-bootstrap 95% interval |
|---:|---:|---:|
| 1 | 0.359260 | [0.255263, 0.465877] |
| 2 | 0.476317 | [0.381057, 0.578071] |
| 3 | 0.659332 | [0.540676, 0.784033] |

ABD2 is therefore supported: seed 2 expresses a substantial learned
behavioral propensity on the held-out prompts. The seed-2 primary contrast was
0.107297 below the mean of seeds 1 and 3. Against norm-matched controls, seed 1
had a contrast of 0.086635 (25/25 wins), seed 2 -0.050702 (0/25), and seed 3
0.026555 (22/25). This descriptively supports ABD3 while retaining the
preregistered restriction against population inference from three seeds.

### Control family and intervention mode

| Seed | Mode | Random contrast (wins) | Norm-matched contrast (wins) |
|---:|---|---:|---:|
| 1 | Necessity | 0.094112 (25/25) | 0.086635 (25/25) |
| 1 | Sufficiency | 0.107462 (25/25) | 0.094547 (25/25) |
| 2 | Necessity | -0.000434 (14/25) | -0.050702 (0/25) |
| 2 | Sufficiency | -0.002206 (14/25) | -0.073927 (0/25) |
| 3 | Necessity | 0.071934 (25/25) | 0.026555 (22/25) |
| 3 | Sufficiency | 0.104020 (25/25) | 0.049199 (25/25) |

Seed 2 loses behavioral selection specificity in both necessity and
sufficiency against norm-matched controls. Against random controls, its paired
contrasts are approximately zero rather than negative. The replicated boundary
is therefore strongest relative to parameter-magnitude-matched alternatives,
not arbitrary random subsets.

### Condition decomposition: ABD4

For seed 2, the paired norm-control reversal combines different condition
components:

| Mode | Component | Top minus norm control | Prompt-bootstrap 95% interval | Wins |
|---|---|---:|---:|---:|
| Necessity | Subliminal | -0.015452 | [-0.043855, 0.012524] | 5/25 |
| Necessity | Neutral | 0.035250 | [0.022269, 0.048617] | 23/25 |
| Necessity | Paired | -0.050702 | [-0.085204, -0.017349] | 0/25 |
| Sufficiency | Subliminal | -0.033547 | [-0.063529, -0.003102] | 4/25 |
| Sufficiency | Neutral | 0.040381 | [0.026237, 0.055284] | 23/25 |
| Sufficiency | Paired | -0.073927 | [-0.112243, -0.038782] | 0/25 |

The paired result is not adequately described as merely absent subliminal
behavior. Relative to norm controls, the subliminal top set is weak or
negative while the neutral top set is positively selection-specific; paired
subtraction amplifies this condition asymmetry. ABD4 is resolved descriptively
in favor of contributions from both conditions, without adding a post-hoc
directional hypothesis.

### Prompt families and endpoint convergence

The seed-2 necessity primary contrast was heterogeneous across the predeclared
families:

| Family | Top minus norm control | Prompt-bootstrap 95% interval | Wins |
|---|---:|---:|---:|
| Direct preference | -0.020012 | [-0.063832, 0.018493] | 5/25 |
| Identity affinity | -0.131981 | [-0.182185, -0.087157] | 0/25 |
| Hypothetical choice | -0.000113 | [-0.069914, 0.069178] | 15/25 |

The pooled primary result remains the confirmatory result. Identity-affinity
prompts carry the clearest secondary reversal; hypothetical-choice prompts are
approximately null under necessity. This is a context modifier, not a
replacement endpoint.

All continuous seed-2 necessity norm-control endpoints agreed in direction:
target log probability was -0.050702, target probability -0.011061, and the
target-versus-lion margin -0.220282, with all three bootstrap intervals below
zero and 0/25 wins. Greedy target choice was -0.012778 with an interval ending
at zero and 0/25 wins. Full-adapter greedy target choice remained zero in all
seeds despite positive continuous shifts, so it is an insensitive thresholded
readout on this prompt inventory and does not overturn the primary result.

## Decision-gate outcome

- **Level 0, integrity:** PASS.
- **Level 1, behavior to mediate:** PASS; ABD2 is supported.
- **Level 2, pooled seed-2 outcome:** **STRONG REPLICATION; GO**.
- **Cross-cutting modifiers:** both intervention modes lose norm-matched
  specificity; the paired effect reflects condition asymmetry; continuous
  endpoints converge; prompt families differ; the original seed-2 versus
  seeds-1/3 pattern is retained.

The decision matrix therefore authorizes **Causal State Localization v1**,
centered on seed 2 and evaluated across all three seeds. The next study should
retain pooled target log probability as its primary behavioral endpoint,
stratify the predeclared prompt families, and localize subliminal and neutral
conditions separately before constructing a paired trait-specific mediator.
Identity-affinity prompts are a prospectively justified high-signal stratum;
the other families must remain generalization and context controls.

Teacher-state rescue follows localization only if a candidate mediator is
identified. H4, additional module-control draws, k sweeps, new training seeds,
optimizer studies, checkpoints, and new traits or models remain lower priority
and are not authorized as the immediate next study.

## Deviations from preregistration

There were no changes to hypotheses, prompts, module sets, endpoints,
thresholds, or analysis after outcome inspection. The only technical
irregularity was the null `git_dirty` provenance field documented above. The
scientific analysis used the frozen artifacts and contract unchanged.

## Interpretation

The study strongly replicates the seed-2 activation--behavior dissociation on
new fixed prompts. The inherited top-k modules were selected because they were
uniquely necessary for the parent teacher-aligned activation effect, yet in
seed 2 they are neither uniquely necessary nor uniquely sufficient for the
held-out behavioral readout relative to norm-matched alternatives. This cannot
be attributed to an absent learned trait: the full seed-2 adapter produces a
large positive continuous behavioral shift.

The strongest warranted mechanistic conclusion is narrower than saying that
the teacher vector is behaviorally irrelevant. Instead, scalar global
teacher-alignment ranking is not a sufficient ordering of behavioral causal
relevance in seed 2. The condition decomposition and prompt-family variation
suggest that mediation depends on condition and elicitation context. This
weakens a simple global version of M1, while spatially gated M1, an incomplete
teacher vector requiring a residual subspace (M2), and context-dependent
parameter interactions (M3) all remain compatible with the data.

The study does not distinguish those models because it does not observe or
intervene on candidate internal states. Causal state localization is therefore
the highest-information next experiment: it tests where activation and
behavior cease to share a causal path before any rescue assay assumes what the
missing mediator is.

## Evidence boundaries

- The evidence concerns one model family, trait, LoRA method, and three frozen
  training seeds; only seed heterogeneity, not a seed population, is measured.
- Module sets were selected on parent activation effects, never on the new
  behavioral outcomes.
- Norm-matched controls establish a ranking failure relative to
  magnitude-matched parameter alternatives; they do not identify the relevant
  state or prove parameter-level degeneracy.
- Prompt families are researcher-defined contexts, and their secondary
  heterogeneity is exploratory within the fixed family analysis.
- Target log probability measures behavioral propensity under controlled
  prompts, not free-running deployment behavior.
- The data weaken only a globally sufficient scalar teacher-vector account.
  They do not exclude spatial gating, residual mediators, distributed
  representations, nonlinear interactions, or multiple realizability.
- No claim about semantic training, H4, optimizer dynamics, checkpoint
  formation, other traits, or other models follows from this study.

## Reproduction and artifact locations

- Manifest: `configs/validation/cat_activation_behavior_dissociation_v1.yaml`
- Frozen prompts: `research/activation_behavior_dissociation_v1/PROMPTS.jsonl`
- Audit command:

  ```bash
  python scripts/audit_activation_behavior_dissociation.py \
    --expected-commit ac7640a8cfacc98177632cf1bd986c7a440b8cb7
  ```

- Local/cluster output namespace:
  `results/research/qwen7b_cat_activation_behavior_dissociation_v1`
- Primary aggregate: `aggregate/dissociation.json`
- Complete machine-readable table: `aggregate/dissociation.csv`

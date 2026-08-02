# Cross-seed confirmatory baseline: audited results

Status: complete, audited, and frozen at `thesis-confirmatory-baseline`.

## Integrity

- Finalizer: successful.
- Expected final artifacts: present.
- Provenance: complete.
- Final checksums: present.
- Audit inconsistencies: none.
- Overall status: PASS.

## Behavioral gate and H1

| Seed | Behavioral gate | Prompt 95% interval | Subliminal alignment | Neutral alignment | Difference |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.156979 | [0.868987, 1.454277] | 0.250609 | 0.029444 | 0.221166 |
| 2 | 1.185086 | [0.840232, 1.540919] | 0.273897 | 0.005701 | 0.268196 |
| 3 | 1.081356 | [0.771543, 1.403629] | 0.294875 | 0.016394 | 0.278481 |

All three paired alignment differences are positive. H1 is descriptively
supported across the three independent seeds; prompt intervals do not provide
population inference over training seeds.

Authoritative archive source: `reports/seed_summary.csv`.

## H2: ranking organization and concentration

| Seed pair | Spearman rho | Top-10 overlap | Top-20 overlap |
|---|---:|---:|---:|
| 1–2 | 0.498420 | 6 | 13 |
| 1–3 | 0.387894 | 5 | 14 |
| 2–3 | 0.359047 | 5 | 13 |

At k=20, selected modules retain the following fractions of the full paired
effect under sufficiency:

| Seed | Activation fraction | Behavioral fraction |
|---:|---:|---:|
| 1 | 0.564960 | 0.313754 |
| 2 | 0.606803 | 0.151925 |
| 3 | 0.553382 | 0.165926 |

This supports partial concentration together with distribution, redundancy,
and seed-dependent exact identity. It does not support a fixed circuit.

Authoritative archive sources: `reports/ranking_similarity.csv` and
`reports/aggregate_effects.csv`.

## H3: k=20 top-minus-norm-matched control

| Mode | Readout | Seed 1 | Seed 2 | Seed 3 |
|---|---|---:|---:|---:|
| Necessity | Activation | 0.483244 | 0.341653 | 0.350679 |
| Necessity | Behavior | 0.072704 | 0.028359 | 0.003424 |
| Sufficiency | Activation | 1.484599 | 0.840741 | 0.831724 |
| Sufficiency | Behavior | 0.092280 | 0.053682 | 0.033597 |

All twelve stored k=20 contrasts are positive. At k=10, necessity contrasts
remain positive, while sufficiency is not consistently selection-specific.
H3 is therefore partially supported and strongest at k=20 against the frozen
norm-matched control.

Authoritative archive source: `reports/aggregate_effects.csv`.

## Hypothesis status

| Hypothesis | Status | Boundary |
|---|---|---|
| H1 | Descriptively supported | Three training seeds; no seed-population significance claim |
| H2 | Moderately supported | Partial concentration, not extreme sparsity or fixed identity |
| H3 | Partially supported | Strongest at k=20 against norm-matched controls |
| H4 | Outside scope | No semantic-learning arm |

## Strongest supported conclusion

The teacher-aligned activation direction has a partially concentrated,
distributed, redundant, and seed-dependent LoRA implementation. Selected
top-20 modules show the cleanest causal advantage over norm-matched controls,
while activation mediation is stronger than behavioral mediation.

## Evidence boundaries

- One model family and one target trait.
- Three independent training seeds.
- Forty-two module candidates in six selected layers.
- Norm-matched controls only.
- No semantic-learning condition.
- Teacher alignment is an informative but incomplete behavioral mediator.
- Prompt bootstrap intervals must not be interpreted as training-seed
  uncertainty.

## Frozen artifact sources

| Result family | Archive path |
|---|---|
| Behavioral gate and alignment | `reports/seed_summary.csv` |
| Intervention aggregates | `reports/aggregate_effects.csv` |
| Ranking stability and overlap | `reports/ranking_similarity.csv` |
| Prompt-level intervals | `aggregate.json` |
| Hypothesis labels | `reports/hypotheses.csv` |
| Artifact integrity | `final_artifacts.sha256` |
| Completion status | `orchestration/finalize.complete.json` |

Final archive SHA-256:
`b9e8733905f598c8f7638678c75e8ffcd59d8c1cfec5dc9a260626985dbee8dd`.

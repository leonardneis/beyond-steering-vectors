# Activation--Behavior Dissociation v1: prospective decision matrix

Status: specified before the first study run and before inspection of any new
outcome.

## Purpose

This matrix translates the preregistered endpoints into prospective research
decisions. It does not change the primary endpoint, directional hypotheses, or
thresholds in [PREREGISTRATION.md](PREREGISTRATION.md). The hierarchy prevents
a favorable secondary endpoint, prompt family, condition component, mode, or
seed from silently replacing a failed primary result.

The hypotheses referenced below are:

- **ABD1:** seed 2 retains a non-positive k=20 necessity top-minus-norm paired
  target-logprob contrast on the pooled new prompts;
- **ABD2:** seed 2 retains a positive paired full-adapter target-logprob shift;
- **ABD3:** the mean selection-specific contrast in seeds 1 and 3 exceeds seed
  2;
- **ABD4:** condition-resolved effects identify whether the paired boundary is
  carried by the subliminal adapter, neutral adapter, or both.

The downstream mechanistic models remain provisional because this study does
not intervene on internal activation states:

- **M1:** a spatially gated teacher vector is the relevant behavioral mediator;
- **M2:** the teacher vector is incomplete without a small residual subspace;
- **M3:** context-dependent parameter interactions permit functionally
  different realizations with similar teacher-aligned activation.

## Decision hierarchy

Apply the following levels in order. A failure at an earlier level constrains
all later interpretation.

### Level 0 -- integrity

| Outcome | Hypothesis consequence | Decision | Next study |
|---|---|---|---|
| All hashes, counts, paired rows, controls, provenance, and completion markers pass | Scientific outcomes may be interpreted | Continue to Level 1 | Determined below |
| Any integrity condition fails | No ABD hypothesis is tested; no scientific result may be inferred | **NO-GO** for outcome interpretation and all mechanistic follow-up | Repair or rerun the same study version if the frozen contract is unchanged; otherwise create v2 |

### Level 1 -- is there behavior to mediate?

| Outcome | Supported / weakened | Decision | Next study |
|---|---|---|---|
| Seed-2 paired full-adapter target-logprob shift is positive | Supports ABD2; permits interpreting absent selection specificity as mediation rather than absent learning | Continue to Level 2 | Determined by ABD1 |
| Seed-2 full-adapter mean is positive but its prompt interval crosses zero | Directionally supports ABD2 but makes seed-2 mechanistic interpretation fragile | **CONDITIONAL GO** only for endpoint validation; no seed-2 rescue yet | Behavior Measurement Validation v1 using fixed full-adapter evaluations, without module search |
| Seed-2 full-adapter shift is non-positive | Weakens ABD2 and the premise that the new prompts express the learned trait; neither supports nor refutes ABD1 mechanistically | **NO-GO** for seed-2 causal-state localization or rescue | Behavior Measurement Validation v1; retain Parameter Formation v1 as the evidence boundary |

### Level 2 -- primary pooled Seed-2 outcome

The primary outcome is always seed 2, k=20, necessity, paired target log
probability, top-k minus the mean of 25 frozen norm-matched controls, pooled
over all 72 prompts.

| Outcome | Operational definition | Supported / weakened | Go / No-Go | Next study |
|---|---|---|---|---|
| **Full replication** | Mean < 0, upper prompt-bootstrap bound < 0, and top-k wins no more than 5/25 controls | Strongly supports ABD1; supports ABD3 if seeds 1/3 remain larger; raises M2/M3 relative to a globally sufficient M1 account but does not distinguish them | **GO** | Causal State Localization v1, centered on seed 2 but evaluated in all seeds |
| **Directional partial replication** | Mean < 0 and wins <= 5/25, but interval crosses zero | Directionally supports ABD1; prompt uncertainty weakens a categorical seed-2 claim | **GO with reduced claim** | Causal State Localization v1 with pooled target log probability fixed as primary and larger held-out prompt evaluation |
| **Prompt-conditional replication** | Pooled criterion fails, but at least one predeclared family has mean < 0 and wins <= 5/25 | Weakens pooled ABD1; supports context dependence and therefore raises M3; the successful family remains secondary | **CONDITIONAL GO** | Causal State Localization v1 restricted prospectively to the responsible family, with the other families retained as negative/generalization controls |
| **No replication: Seed 2 becomes selection-specific** | Pooled and every family show a positive top-minus-norm contrast, without a stable seed-2 deficit | Weakens ABD1 and the original seed-specific interpretation; favors prompt sensitivity over a stable Seed-2 mechanism | **NO-GO** for a Seed-2-centered rescue story; **GO** for the broader mediation question | Cross-seed Causal State Localization v1 using continuous activation-to-logit coupling, without designating seed 2 as the mechanistic exception |
| **Stronger reversal** | Seed 2 is clearly more selection-specific than seeds 1/3 | Refutes ABD1's direction and weakens ABD3; exact seed identity is not a stable explanatory variable | **NO-GO** for Seed-2 localization | Functional Cross-Seed Localization v1; if the reversal is large, add new seeds before any seed-specific mechanism claim |

## Cross-cutting outcome modifiers

These modifiers never override the Level-2 primary classification. They select
the correct implementation of the next study.

### Condition-component asymmetry (ABD4)

| Subliminal component | Neutral component | Interpretation | Hypotheses / models | Next study adjustment |
|---|---|---|---|---|
| Top-k lacks specificity | Small or control-like | Dissociation is primarily carried by the learned subliminal adapter | Supports ABD4 as a subliminal-routing result; raises M2/M3 | Localize subliminal states first; neutral remains a matched negative condition |
| Top-k is specific | Equal or larger neutral intervention effect cancels the paired contrast | Paired failure is driven by the neutral comparator, not absent subliminal specificity | Weakens a strong subliminal-mechanism reading of ABD1; supports ABD4 as comparator asymmetry | Condition-Resolved Causal State Localization; audit which neutral states make norm controls behaviorally strong |
| Both conditions show large same-direction effects | Their subtraction removes specificity | Shared LoRA sensitivity or baseline routing dominates the paired endpoint | Weakens trait-specific parameter selection; compatible with M3 but not evidence for it | Condition-Resolved Localization before any teacher-vector rescue |
| Components have opposite signs | Paired effect is amplified by cancellation/reversal | The paired statistic hides qualitatively different condition mechanisms | Supports ABD4; weakens any single-effect-size account | Separate subliminal and neutral localization with paired contrast retained only as the trait-specific summary |
| Neither condition has a stable intervention effect | Behavioral selection signal is too weak on the new prompts | Weakens ABD1/ABD4 interpretability | **NO-GO** for module-to-state rescue | Behavior Measurement Validation v1 |

### Necessity--sufficiency asymmetry

| Outcome | Mechanistic implication | Supported / weakened | Next study |
|---|---|---|---|
| Necessity and sufficiency both lose behavioral specificity | Strongest activation--behavior dissociation; selected parameters are neither uniquely required nor uniquely sufficient for behavior despite activation necessity | Raises M2/M3; weakens a simple M1 sufficiency story | Causal State Localization, then Teacher-State Rescue |
| Necessity loses specificity but sufficiency is specific | Top-k can produce behavior, but alternative subsets are similarly consequential under removal; consistent with redundancy | Supports causal degeneracy and raises M3 | Localize sufficiency-induced states, then Activation-Matched Realizations |
| Necessity is specific but sufficiency loses specificity | Selected modules are behaviorally required but not independently sufficient | Raises M2 over a complete scalar M1 account | Causal State Localization followed by Teacher-State Rescue and a minimal residual-subspace branch |
| Both modes are behaviorally specific | Weakens the Phase-1 dissociation as a stable phenomenon | Supports a stronger parameter-to-behavior chain; M1 becomes more plausible but remains untested | Cross-seed Causal State Localization; rescue remains useful but is no longer anomaly-driven |

### Endpoint asymmetry

| Outcome | Interpretation | Decision | Next study |
|---|---|---|---|
| Target log probability replicates, greedy choice does not | Continuous propensity changes without crossing the discrete choice threshold | ABD1 stands; choice remains secondary | **GO** with target log probability as the localization endpoint |
| Greedy choice suggests a dissociation but target log probability does not | Nonlinear decoding or parsing may create an apparent categorical effect | Primary ABD1 fails; choice cannot rescue it | **NO-GO** for mechanistic localization until Behavior Endpoint Calibration v1 |
| Target probability and target-vs-lion margin agree with target log probability | Convergent secondary support | Strengthens robustness without changing the primary decision | Follow the Level-2 next study |
| Continuous logit-derived endpoints disagree in sign | Candidate normalization or competitor choice matters | Weakens a scalar behavioral endpoint | Calibrate the logit contrast before state rescue; retain all declared endpoints |

### Prompt-family asymmetry

| Outcome | Interpretation | Decision | Next study |
|---|---|---|---|
| All three families agree | Prompt-general boundary within the defined inventory | Follow pooled gate | Causal State Localization with family-stratified validation |
| One family drives the effect | Context-dependent mediation | Conditional, never promoted to pooled confirmation | Family-fixed Causal State Localization with other families as controls |
| Families have opposite signs | The global endpoint averages distinct behavioral computations | Weakens pooled ABD1; raises M3 | Context-Resolved Causal State Localization; no global rescue coefficient |
| No family is stable | Evaluation noise or absent behavioral signal | **NO-GO** | Behavior Measurement Validation v1 |

### Cross-seed pattern

| Pattern | Hypothesis consequence | Decision | Next study |
|---|---|---|---|
| Seed 2 fails specificity; seeds 1 and 3 retain it | Supports ABD1 and ABD3 in the originally observed pattern | **GO** | Seed-2-centered Causal State Localization with seeds 1/3 as positive comparators |
| Seed 2 and one additional seed fail | Supports ABD1 but weakens a uniquely Seed-2 account; broader heterogeneity | **GO** | Cross-seed Causal State Localization; prioritize functional over identity-based grouping |
| All three seeds fail behavioral specificity while activation necessity remains inherited and strong | Strong general activation--behavior dissociation | Weakens ABD3 but strongly elevates the central mediation problem; raises M2/M3 | **GO, highest priority** | Cross-seed Causal State Localization followed by Teacher-State Rescue |
| Seed 2 recovers and another seed loses specificity | Weakens ABD1/ABD3 and indicates unstable seed labeling | **NO-GO** for a Seed-2 story | Functional Cross-Seed Localization; consider two new seed pairs before rescue generalization |
| All three seeds become behaviorally specific | Phase-1 Seed-2 result was prompt-set sensitive | Weakens ABD1 and the strong dissociation framing | **GO only for general mechanism**, not anomaly explanation | Standard Cross-Seed Causal State Localization; deprioritize residual-subspace search unless rescue later fails |

## Mapping to the next research program stage

| Gate result | Authorized next study | Explicitly not authorized |
|---|---|---|
| Full or directional replication with ABD2 positive | Causal State Localization v1 -> Teacher-State Rescue v1 | H4, optimizer, checkpoints, new trait/model |
| Prompt-conditional replication | Context-fixed Causal State Localization v1 | Pooled/global rescue claim |
| No Seed-2 replication but stable behavior across prompts | Cross-seed Causal State Localization v1 | Seed-2-specific mechanistic claim |
| Global loss of behavioral specificity | Cross-seed Causal State Localization v1 -> Teacher-State Rescue v1 | More module-control draws or k sweeps |
| Full-adapter behavioral signal absent or endpoint-incoherent | Behavior Measurement Validation or Endpoint Calibration v1 | Any activation rescue interpreted behaviorally |
| Integrity failure | Same-study repair/rerun or v2 | Scientific interpretation of partial artifacts |

In every branch, H4 remains post-thesis until a behaviorally validated causal
state or rescue assay exists. No outcome in this study authorizes semantic
training, checkpoint dynamics, optimizer ablations, or broad generalization as
the immediate next step.

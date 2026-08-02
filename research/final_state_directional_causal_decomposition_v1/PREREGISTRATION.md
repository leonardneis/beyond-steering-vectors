# Final-State Directional Causal Decomposition v1: preregistration

Status: prospective; no study outcomes have been produced or inspected.

## Research question

Does the replicated seed-2, k=20 behavioral necessity contrast act on the
next-token Cat--Lion decision margin through displacement along the frozen
teacher direction at the final post-RMS state, or through the orthogonal
complement of that direction?

This is a directional causal decomposition at one predeclared causal
bottleneck. It is not a localization sweep and does not claim to identify the
upstream circuit that creates the final state.

## Inherited evidence and frozen objects

- Confirmatory Baseline establishes the learned cat preference.
- Parameter Formation v1 fixes the seed-2 k=20 `top_k` set and 25
  `norm_matched_control` sets and establishes teacher-aligned activation
  necessity.
- Activation--Behavior Dissociation v1 fixes the 72 prompts and establishes a
  strong seed-2 necessity contrast on both the preregistered target-logprob
  endpoint and the secondary Cat--Lion margin.
- Parent Cat--Lion top-minus-norm contrast: `-0.22028208520677353`.
- Frozen smallest effect size of interest: 20% of its magnitude,
  `epsilon = 0.04405641704135471` margin-logit units.
- Frozen teacher artifact: seed-1 subliminal teacher vector, final hidden-state
  slot, normalized before use.
- Frozen base-model revision:
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Frozen conditions: seed-2 subliminal and neutral adapters.
- Frozen interventions: necessity only; one k=20 `top_k` and 25 k=20
  `norm_matched_control` draws.
- Frozen prompt inventory: all 72 Activation--Behavior Dissociation v1 prompts,
  with the existing 24/24/24 family labels.

Hashes and paths are fixed in the manifest. Any mismatch is an integrity
failure, not an invitation to substitute an artifact.

The 20% equivalence margin defines the smallest residual considered
mechanistically material: a residual contribution no larger than 20% of the
frozen parent contrast is treated as practically negligible for this gate. It
is fixed from the parent point estimate before this study and is not
re-estimated from the new component outcomes. Because `G_full` itself has a
reconstruction tolerance, this threshold is not described as a literal
“percentage mediated” estimate.

## Formal intervention and decomposition

For condition \(c\in\{S,N\}\), intervention set \(s\), and prompt \(p\), let
\(h^F_{c,p}\in\mathbb{R}^d\) be the full-adapter final hidden state after the
model's final RMS normalization and immediately before the LM head. Let
\(h^A_{c,s,p}\) be the state after setting the selected LoRA module scalings to
zero. The natural intervention displacement is

\[
\Delta h_{c,s,p}=h^F_{c,p}-h^A_{c,s,p}.
\]

Let \(t\) be the independently frozen unit teacher direction in the same state
space. Define the fixed orthogonal projectors

\[
P_T=tt^\top,\qquad P_R=I-tt^\top,
\]

and

\[
\Delta h_T=P_T\Delta h,\qquad
\Delta h_R=P_R\Delta h,\qquad
\Delta h=\Delta h_T+\Delta h_R.
\]

Let \(w_{cat}\) and \(w_{lion}\) be the corresponding rows of the frozen LM
head and \(u=w_{cat}-w_{lion}\). Because the head is affine, the causal change
in the Cat--Lion margin under ablation is

\[
e^{full}_{c,s,p}=u^\top\Delta h_{c,s,p}
=u^\top\Delta h_{T,c,s,p}+u^\top\Delta h_{R,c,s,p}
=e^T_{c,s,p}+e^R_{c,s,p}.
\]

This identity alone is geometric. It becomes a test of a mechanistic
hypothesis because (i) \(\Delta h\) is caused by the predeclared parameter
intervention, (ii) \(t\) was fixed independently of these behavioral outcomes
and previously linked to the teacher and to the selected modules, and (iii)
\(u\) is the model's actual decision readout rather than a fitted probe. The
test asks whether the output-relevant effect of a natural parameter ablation
lies in that independently motivated one-dimensional subspace. It does **not**
by algebra alone prove that upstream information literally travels only along
that direction.

For audit and as an intervention-level validation, the study also constructs
the two counterfactual final states

\[
h^{A+T}=h^A+\Delta h_T,\qquad h^{A+R}=h^A+\Delta h_R,
\]

and applies the unchanged LM head. Their margin changes must equal \(e^T\) and
\(e^R\), respectively. These final-state patches test direct readout
sufficiency at the bottleneck; they do not establish that the patched states
are naturally reachable upstream.

## Assumptions required for mechanistic interpretation

`G_T` may be interpreted as the **teacher-axis-aligned direct contribution**,
and `G_R` as the **direct contribution of the complementary final-state
displacement**, only under all of the following assumptions:

1. The stored state is the post-RMS input consumed by the unchanged LM head;
   the two predeclared Cat and Lion logits are reconstructed numerically from
   that state at runtime.
2. The teacher vector's final slot and the intervention states share model,
   dimensionality, token position, coordinate system, and normalization
   semantics.
3. The teacher direction is nonzero, normalized, fixed before outcome
   inspection, and was not selected to maximize the present margin effect.
4. The Cat and Lion readouts are valid single tokenizer tokens and their token
   IDs and LM-head rows remain fixed across conditions and interventions.
5. Full and ablated evaluations differ only in the declared LoRA scaling mask;
   prompts, adapter, weights, quantization, tokenizer, and numerical regime are
   otherwise held fixed.
6. Inference is deterministic and paired at the prompt level.
7. The affine LM head makes the Cat--Lion margin a complete linear functional
   of this final state; any LM-head bias is unchanged and therefore cancels.
8. The parent top-minus-control effect is reproduced, so there is a causal
   behavioral-margin effect to decompose.
9. The fixed Euclidean inner product is an adequate operational metric for the
   declared teacher axis in residual coordinates.
10. The top-minus-mean-control, subliminal-minus-neutral, and prompt averages
    are applied exactly as preregistered; their linearity preserves the exact
    decomposition.
11. Claims are restricted to direct next-token readout at this bottleneck. A
    stronger claim of natural causal mediation additionally requires that the
    component interventions are meaningful counterfactuals and that no
    unmeasured upstream route is being inferred from final-state geometry.
12. Calling `G_T` *teacher-mediated*, rather than merely teacher-axis-aligned,
    additionally assumes construct validity: the frozen teacher contrast
    identifies the same latent teacher property in teacher and student; no
    unrelated output-relevant variable is collinear with it; changing its
    coefficient while holding the complement fixed is a consistent intervention
    on that property; and the property has no causally relevant realization
    outside this one-dimensional coordinate for the declared margin.
13. Calling `G_R` a residual *mechanism*, rather than a residual contribution,
    additionally requires evidence that the complement is stable under
    intervention and corresponds to one reproducible causal variable. This
    study does not assume or establish that stronger condition.

Thus the audit can establish exact directional direct effects without
assumptions 12--13. Those stronger assumptions are required for the semantic
labels “teacher-mediated mechanism” and “residual mechanism.” `G_R` is always
only a complement relative to this one frozen direction; it is not assumed to
be a coherent feature, a single circuit, or “non-teacher information.”

## Conditions invalidating the interpretation

No teacher-mediated/residual mechanistic conclusion is valid if any alignment,
hash, tokenization, state-to-head reconstruction, intervention, or parent-effect
gate fails; if the teacher direction was outcome-selected; if the vector and
state use different pre/post-normalization conventions; if the relevant choice
requires multi-token continuation dynamics; or if parameters/readout weights
change between paired states. Large numerical non-additivity also invalidates
the result.

Even with a valid audit, off-manifold component patches, strong nonlinear
effects after the measured next token, or an inappropriate Euclidean metric
limit the result to a directional direct-effect decomposition. They prevent a
claim that the teacher axis is the unique natural mediator. A small `G_R`
cannot exclude upstream distributed computation that converges onto the
teacher axis; a large `G_R` cannot show that the complement is one unified
mechanism.

## Why the final post-RMS state is the gate location

The post-RMS state immediately before the LM head is the smallest exact causal
bottleneck for the next-token margin. It is sufficient for the logits, is read
by the model's actual fixed linear head, and yields an exact additive
decomposition without fitting a probe. It also removes layer and token search,
avoids multiple-comparison freedom, and directly connects the inherited
parameter intervention to the inherited behavioral endpoint.

A pre-RMS state would pass through a nonlinear normalization and would not
support the same additive attribution. Earlier residual-stream locations would
require choosing layers and would mix localization with the directional test.
The cost of the final-state choice is explicit: this study can identify which
directional component reaches the readout, but not where or how that component
was formed upstream.

## Hypotheses

**FSD1 -- effect reconstruction gate.** The new state-based computation
reproduces the negative parent paired top-minus-norm Cat--Lion contrast, and
direct logits, state-derived logits, and component sums agree within the frozen
tolerances.

**FSD2 -- teacher-axis sufficiency (primary mechanistic hypothesis).** After
accounting for the teacher-axis contribution, the residual paired
top-minus-norm contribution `G_R` is practically equivalent to zero within
`[-epsilon, +epsilon]`.

**FSD3 -- residual necessity (competing hypothesis).** `G_R` retains a
substantial contribution in the inherited negative direction, so the scalar
teacher axis is insufficient to explain the behavioral necessity contrast.

No hypothesis predicts that `G_R` is semantically homogeneous.

## Estimands

For component \(q\in\{full,T,R\}\), first form the condition-paired effect

\[
d^q_{s,p}=e^q_{S,s,p}-e^q_{N,s,p}.
\]

For each prompt, compare the fixed top set with the mean of the 25 fixed
norm-matched controls:

\[
g^q_p=d^q_{top,p}-\frac{1}{25}\sum_{j=1}^{25}d^q_{norm_j,p}.
\]

The study estimand is

\[
G_q=\frac{1}{72}\sum_{p=1}^{72}g^q_p.
\]

The **primary estimand is `G_R`**. By construction and as a mandatory audit
identity, `G_full = G_T + G_R` globally, per family, per prompt, per set, and
per condition up to declared floating-point tolerance.

## Primary analysis, bootstrap, and equivalence test

The 72 prompts are the uncertainty unit. Each of 5,000 bootstrap replicates
resamples 24 prompts with replacement independently inside each of the three
fixed families, concatenates the 72 sampled rows, and recomputes all three
`G_q` values. Control draws remain fixed and are averaged within prompt; they
are not treated as replications. Bootstrap seed: `20260804`.

Report percentile 95% intervals for all estimands and the percentile 90%
interval for `G_R`. The two-one-sided equivalence decision at alpha 0.05 is
implemented as: the 90% interval for `G_R` must lie wholly inside
`[-0.04405641704135471, +0.04405641704135471]`. This is the sole primary
confirmatory test. No data-dependent rescaling of epsilon is allowed.

Before that test is interpretable, FSD1 must pass: the new `G_full` is negative,
its 95% bootstrap upper bound is below zero, and its point estimate differs
from the frozen parent value by no more than epsilon.

## Secondary endpoints

- `G_full`, `G_T`, and the exact reconstruction residual;
- teacher share `G_T/G_full`, reported only when `|G_full| >= epsilon`;
- condition-resolved and set-resolved component means and ranks;
- prompt-family and per-prompt `G_full`, `G_T`, and `G_R` estimates;
- fixed readout compatibility `u^T t`;
- Cat--Lion LM-head margins for `h^A`, `h^(A+T)`, `h^(A+R)`, and `h^F`;
- the top set's rank against the 25 controls for each component.

All secondary results are descriptive. Families may diagnose heterogeneity but
cannot replace or reverse the pooled primary endpoint.

## Decision hierarchy

1. Artifact and numerical integrity must pass.
2. The parent margin effect must be reconstructed under FSD1.
3. Apply the preregistered equivalence test to `G_R`.
4. Classify the result using [DECISION_MATRIX.md](DECISION_MATRIX.md).

## Evidence boundaries and exclusions

The result concerns one model family, trait, LoRA method, training seed,
intervention size, contrast, prompt inventory, and next-token readout. It does
not establish multi-token behavioral mediation, upstream localization,
feature uniqueness, cross-seed generality, or natural reachability of patched
states.

Excluded: new training; new prompts; random controls; sufficiency; seeds 1/3;
k, layer, token, direction, or competitor sweeps; sparse feature discovery;
path/attribution patching; H4; optimizer/checkpoint studies; semantic training;
and model/trait generalization.

## Amendments

Any change after this contract is frozen must be dated and state whether any
study outcome had been inspected. A change to hypotheses, endpoints, epsilon,
prompts, sets, direction, or decision rules requires a new study version.

# Bidirectional Teacher-Coordinate Interchange v2: preregistration

Status: prospective public freeze. No C18-v2 scientific model output has been
created or inspected. This document does not authorize scientific execution.

## Question and identification target

Can the independently frozen sequence of one-dimensional teacher-projection
coordinates at decoder-block outputs 0--26, imposed on every real token of the
complete prefill, make the fixed Seed-2, k=20, necessity, Cat-minus-Lion
selectivity contrast practically invariant to the top-versus-norm parameter
background in both interchange directions?

The identified target is controlled functional adequacy/interchange of this
ordered joint operator for one fixed contrast. Functional adequacy/interchange
is not natural mediation. The study does not identify a natural direct or
indirect effect, a proportion mediated, a unique feature/circuit, natural
reachability of hybrid trajectories, semantic completeness of the teacher
representation, or generality beyond the declared organism.

## Frozen organism and evidence boundary

- Model: Qwen/Qwen2.5-7B-Instruct at revision
  `a09a35458c702b33eeacc393d103063234e8bc28`.
- Student: rank-8, alpha-8, dropout-0 QLoRA on an NF4 base.
- Training seed: 2, selected as a mechanistic organism; not a random seed
  population.
- Conditions: Subliminal (`S`) and Neutral (`N`).
- Parameter intervention: necessity, k=20.
- Sets: the one frozen PF top set and all 25 frozen global norm-matched sets.
- Inputs: exactly the 72 known ABD prompts, 24 in each frozen family.
- Endpoint: first next-token Cat-minus-Lion margin at the last real prefill
  token.
- Primary aggregation: `S-N`, then `top-mean(25 norm)`, then the equally
  weighted prompt mean.

The prompts have already been used in ABD and FSD. C18 v2 is therefore a
mechanistic test on a known finite inventory, not an independent behavioral or
prompt-population replication. Prompt-family analyses are descriptive and can
never replace the pooled primary decision.

The norm controls match the PF LoRA-norm rule, not layer profile, module type,
Block-27 membership, or behavioral effect. Claims are limited to the frozen
top-versus-these-25-norm-sets contrast. Seeds 1/3, other k, sufficiency, new
prompts, traits, models, multi-token generation, or fitted directions are
outside the primary study.

## Unambiguous backgrounds and intervention

C18 v2 does not use the ambiguous historical `A/F` mnemonic.

- `P0`: the selected 20-module set is ablated.
- `P1`: full adapter; none of those 20 scalings is ablated.
- `D0`: immutable natural donor trajectory generated under `P0`.
- `D1`: immutable natural donor trajectory generated under `P1`.

For condition `c`, set `s`, prompt `p`, slot `l in {1,...,27}`, and logical
real-token index `i`, let `t_l` be the frozen teacher row renormalized once in
float64 and

```text
d_D(c,s,p,l,i) = t_l^T h_D^natural(c,s,p,l,i).
```

Slot `l` is the post-forward output of decoder block `l-1`, after attention and
MLP residual additions. Slots 1--27 therefore clamp block outputs 0--26.
Block 27, final RMSNorm, and the LM head remain free. At each recipient hook:

```text
h64 = exact_float64(h)
h*  = h64 + t_l [d_D - t_l^T h64]
h+  = cast_to_original_float16(h*)
```

All dot products and the complete correction are float64, followed by exactly
one cast to the stream dtype. There is no state normalization, fitted alpha,
metric change, layer/token selection, or feedback donor from a hybrid
trajectory. All hooks are registered before the forward in ascending order.
Every `attention_mask==1` position in the full rendered prefill is clamped;
padding is never written. The downstream computation is otherwise free.

Natural donor and recipient buffers are separately identified, immutable, and
must have identical prompt ID, batch ID, physical row, width, mask, logical
positions, and physical cache positions.

## Y factorial

For every identified `(condition,set,prompt)`:

```text
Y00: recipient P0; donor coordinates D0; natural P0 diagonal
Y01: recipient P0; donor coordinates D1; Full-to-ablated rescue
Y10: recipient P1; donor coordinates D0; Ablated-to-full reverse/noising
Y11: recipient P1; donor coordinates D1; natural P1 diagonal
```

Scientific `Y00` and `Y11` are natural forwards. Separate `0->0` and `1->1`
self-replays are technical controls and never replace a natural diagonal.

Per raw location:

```text
E  = Y11 - Y00
L  = Y01 - Y00
H  = Y11 - Y10
B0 = Y10 - Y00
B1 = Y11 - Y01
I  = Y11 - Y10 - Y01 + Y00 = B1 - B0 = H - L
E  = H + B0 = L + B1
```

`B0` is the remaining parameter-background effect while D0 coordinates are
fixed. `B1` is the corresponding effect while D1 coordinates are fixed. A
small `B1` alone is rescue adequacy; a small `B0` alone is reverse adequacy.
Both are required for bidirectional adequacy.

## G estimands

For `q in {E,L,H,B0,B1,I}`:

```text
d_q(s,p) = q(S,s,p) - q(N,s,p)
g_q(p)   = d_q(top,p) - (1/25) sum_j d_q(norm_j,p)
G_q      = (1/72) sum_p g_q(p)
```

The sole primary hypothesis is the intersection-union equivalence test:

```text
H0: |G_B0| >= epsilon OR |G_B1| >= epsilon
H1: |G_B0| <  epsilon AND |G_B1| <  epsilon
epsilon = 0.04405641704135471 Cat-minus-Lion logit-margin units.
```

Epsilon was fixed before FSD as 20% of the inherited ABD contrast magnitude
`0.22028208520677353`, on the same endpoint and aggregation. It is an absolute
SESOI for remaining G selectivity, not a mediation share or universal
behavioral threshold.

## Bootstrap and inference

The finite target is exactly these 72 prompts. Exactly 20,000 bootstrap draws
use NumPy `Generator(PCG64)` seed `20260804`. Each draw samples 24 prompt IDs
with replacement independently inside each of the three fixed families. The
same indices are used jointly for every condition, set, Y/Z/W cell, estimand,
and control. Norm sets, tokens, slots, and conditions are not resampled.

Both marginal two-sided 90% percentile intervals for `G_B0` and `G_B1` must be
strictly inside `(-epsilon,+epsilon)`. This is one alpha-.05 IUT and needs no
additional Bonferroni correction. Marginal 95% intervals are reported. The
fixed family `{G_B0,G_B1,G_L,G_H,G_I}` uses marginal 99% intervals as a
conservative Bonferroni simultaneous 95% diagnostic band. Boundary contact is
not PASS. Invalid draws are fatal and never replaced favorably.

The bootstrap resamples prompt clusters only. It never treats the 25 controls,
1,800 norm-prompt comparisons, or three training seeds as independent
replications.

## Natural-effect reconstruction gate

Before mechanistic classification:

```text
G_E < 0
upper95(G_E) < 0
abs(G_E - (-0.22028208520677353)) <= epsilon.
```

Distance to FSD `G_full=-0.219307` is reported. Cross-study rowwise or
cross-shape bit identity is not required under the revised execution semantics.

## Complete 25 x 72 cancellation safeguards

For each `B in {B0,B1}`, norm set `j`, prompt `p`, and condition `c`:

```text
K_B,c,j(p) = B(c,top,p) - B(c,norm_j,p)
r_B,j,p    = K_B,S,j(p) - K_B,N,j(p)
G_B        = mean_{j,p} r_B,j,p
A_B        = mean_{j,p} |r_B,j,p|
R_B        = sqrt(mean_{j,p} r_B,j,p^2)
M_B,j      = mean_p r_B,j,p
F_B,f      = mean_{j,p in family f} r_B,j,p
C_B,c      = mean_{j,p} K_B,c,j(p).
```

A bidirectional primary equivalence PASS is cancellation-qualified if, for
either residual, any rule holds:

1. `A_B >= epsilon`;
2. `R_B >= 2*epsilon`;
3. any `|M_B,j| >= epsilon`;
4. any `|F_B,f| >= epsilon`;
5. any `|C_B,c| >= epsilon`;
6. two fixed family or norm-set means have opposite signs and differ by at
   least `2*epsilon`;
7. S and N component means share a sign, both have magnitude at least epsilon,
   and their paired difference has magnitude below epsilon;
8. a cross-cell output-collapse flag fires: pooled Y01 or Y10 margin SD is at
   most 10% of the smaller natural Y00/Y11 SD (defined only if that reference
   exceeds `1e-6`) while mean absolute deviation from its donor output is at
   least epsilon;
9. a W allocation contains two opposing `Mat99` components whose sum produces
   the small total B.

These are deterministic claim-protection rules, not replacement hypothesis
tests. Quantiles, maxima, and sign proportions over all 1,800 `r_B,j,p` atoms
are descriptive. Cancellation permits only the statement that aggregate G
selectivity is practically retained; it blocks an unqualified transport claim.

## Block-26 natural full-state Z factorial

Let `H_b^nat` be the complete natural token state after block 26 under
background `b`. Define

```text
Z_rb = suffix_r(H_b^nat), r,b in {0,1},
```

where `suffix_r` is block 27 under parameter background `r`, unchanged final
RMSNorm, and the canonical B=6 LM head. `Z00` and `Z11` must same-shape
reproduce the natural diagonals. Cross cells identify controlled Block-27
parameter effects on natural cut states. Z alone is insufficient because it
does not use C18 hybrid states.

## Complete hybrid-state W factorial

Let `H_ab^C18` be the complete state after block 26 actually produced by Yab.
For all `a,b,r in {0,1}`:

```text
W_abr = suffix_r(H_ab^C18)
Y_ab  = W_ab,a.
```

For fixed donor `b`:

```text
B_b = W_1b1 - W_0b0

U_b^(0) = W_1b0 - W_0b0
S_b^(1) = W_1b1 - W_1b0
B_b     = U_b^(0) + S_b^(1)

S_b^(0) = W_0b1 - W_0b0
U_b^(1) = W_1b1 - W_0b1
B_b     = S_b^(0) + U_b^(1)

J_b = U_b^(1)-U_b^(0) = S_b^(1)-S_b^(0).
```

Every component receives the same S-N/top-norm/prompt G operator. Z/W identify
a controlled decomposition at the Block-26 cut only. `U` includes everything
created up to that cut and is not automatically orthogonal, non-teacher, or a
single upstream channel. Material `J`, material Z/W background dependence, or
opposing material U/S cancellation is a `SuffixConflict` and blocks the
bidirectional transport label.

## Orthogonal-dose specificity family

Five outcome-blind frozen direction families use PCG64 seed `20260914`,
family-major then slot-major generation, float64 projection off `t_l`, and
float64 unit normalization. At the same hooks they receive the same signed
teacher correction `alpha_T=d_D-t_l^T h`:

```text
h+ = cast_float16(h64 + alpha_T r_l,k).
```

These are additive matched-dose replays, not orthogonal-coordinate
interchanges. Together they form exactly one secondary specificity family,
not five additional primary hypotheses. With

```text
D_k = max(|G_B0,k|,|G_B1,k|) - max(|G_B0,T|,|G_B1,T|),
```

the modifier "teacher-coordinate-specific relative to all five tested doses"
is allowed only if every 99% interval lower bound for `D_k` is strictly
positive. Otherwise the primary class is unchanged and the specificity claim
is forbidden.

## Hard technical and identification gates

- exact hashes and execution identity;
- canonical B=6 membership, order, width, mask, physical row, logical
  mask-based `position_ids`, and rank-1 physical `cache_position`;
- canonical B=6 LM-head evaluation in every cell;
- raw-byte-identical R, H, and L_same in identical B=6 form;
- identity hook, null operator, 0->0 and 1->1 self-replay;
- immutable donors and separate recipient buffers;
- slots 1--27, exact hook count/order, real-token-only writes;
- exactly 196 canonical LoRA modules; exactly 20 disabled per set; no merge;
  exception-safe restoration;
- finiteness, cleanup, terminal state/readout replay, Y/Z/W algebra, unique raw
  IDs, complete inventories, and atomic publication;
- valid natural-effect reconstruction.

Any failure produces `TECHNICAL / IDENTIFICATION FAILURE` and no scientific
classification. A technical repair may rerun the unchanged version only
without outcome access. A required scientific-semantic change requires STOP
and a new version.

## Numerical execution contract

The authoritative public summary is [CALIBRATION_SUMMARY.md](CALIBRATION_SUMMARY.md).
The scientific execution identity is:

- NVIDIA A100-PCIE-40GB; driver 570.211.01;
- container digest
  `sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755`;
- Python 3.11.10; Torch 2.5.1+cu124; CUDA 12.4;
- Transformers 4.48.3; PEFT 0.14.0; bitsandbytes 0.45.2; NumPy 2.1.2;
- SDPA; fp16 residual stream; NF4 with fp16 compute; double quantization off;
- TF32 off; deterministic algorithms and cuDNN; benchmark off;
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- left padding, explicit logical positions, physical cache positions, and one
  immutable canonical B=6 plan.

Permutation, co-batch composition, additional padding, B=1-versus-B=6,
cross-shape fp32 readout, and q99.5-times-two envelopes are transport
diagnostics, not hard gates. The envelopes are never a scientific SESOI,
equivalence gate, or materiality threshold.

## Outcomes, claims, and exclusions

The ordered, disjoint A--G classification is frozen in
[DECISION_MATRIX.md](DECISION_MATRIX.md). Secondary families, conditions,
candidate logits, perturbation diagnostics, orthogonal doses, and Z/W details
are reported exhaustively but cannot rescue the primary endpoint.

Allowed positive wording is limited to controlled functional information and
adequacy for the fixed Seed-2 G contrast. Forbidden wording includes natural
mediation, a unique or complete teacher representation, a single circuit,
whole-trait sufficiency/necessity, global state reconstruction, a coherent
residual feature, and generalization to other seeds/traits/models.

A negative result limits only this frozen coordinate family and operator. It
does not show that teacher information is irrelevant or that the residual is
high-dimensional.

## Prospective replication path

After a technically valid, unambiguous Seed-2 classification, the unchanged
operator and decision rules will be preregistered separately for Seeds 1 and
3 using their own frozen adapter pairs and top/norm sets. Seed results will not
be pooled to rescue one another. Exactly one result-dependent mechanistic
follow-up is then selected: boundary/path localization after bidirectional
transport; a replicated cut factorial after asymmetry/interaction; or a
split-confirmed residual-carrier versus instance-dependent/multidimensional
test after insufficiency. An independent behavior test follows before a
matched prompted-versus-steered comparison and later trait/model extension.

## Execution lock

The public manifest must retain:

```text
scientific_execution_authorized: false
```

No runner may create Y01/Y10 cells, compute G estimands, expose scientific
candidate logits, or submit a scientific DAG while that value is false. A
separate explicit future authorization is required after the outcome-blind
technical audit.


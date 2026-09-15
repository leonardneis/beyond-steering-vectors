# Bidirectional Teacher-Coordinate Interchange v1: preregistration

Status: prospective public contract. No C18 scientific model output has been
produced or inspected. This document does not authorize a scientific run.

## Scientific question and scope

Can the independently frozen family of nonterminal teacher-projection
coordinates transfer the replicated seed-2, k=20, necessity, Cat--Lion
selectivity contrast between the Parameter Formation top set and the mean of
25 norm-matched sets, Subliminal minus Neutral, in both directions between the
full and parameter-ablated models up to a fixed practical residual size?

The only organism is Qwen/Qwen2.5-7B-Instruct at revision
`a09a35458c702b33eeacc393d103063234e8bc28`, loaded as the frozen rank-8,
alpha-8, dropout-0 QLoRA/NF4 student pair for training seed 2. The trait is
Cat; the conditions are Subliminal (`S`) and Neutral (`N`); the parameter
intervention is necessity with exactly 20 disabled LoRA modules; and the sets
are the one frozen top set plus the 25 frozen global norm sets. All and only
the 72 known ABD prompts are used, 24 in each frozen family. The endpoint is
the first next-token Cat-minus-Lion margin at the last real prefill token.

Seeds 1/3, other k, sufficiency, new prompts, other traits or models,
multi-token generation, fitted directions, and post-outcome layer/token/dose
selection are outside C18 v1. A later prompt confirmation is a separate study
and cannot rescue or modify this result.

## Frozen inputs

The executable manifest fixes every path by a repository-relative public name
and expected SHA-256 or tree digest. It freezes the model config, model
revision, seed-2 subliminal and neutral adapters, PF selection plan, canonical
LF bytes of the 72-prompt inventory, the seed-1 subliminal teacher tensor, both
FSD state artifacts and the FSD aggregate. The historical incompleteness of
teacher-extraction provenance is retained as a limitation: the existing
tensor is the operational object and is not re-estimated.

The Windows worktree may contain CRLF prompt bytes. Scientific and technical
execution must use the canonical Git LF bytes with SHA-256
`f19d3003a771205da5ecd76175d1f81ecba8975812130c2a6c816d87a7af8f23`;
raw-byte mismatch is fatal.

## Exact intervention operator

Let `c in {S,N}`, `s in {top,norm_0,...,norm_24}`, `p` be a frozen prompt,
`a in {0,1}` be recipient parameters (`0=A`, set `s` ablated; `1=F`, full),
and `b in {0,1}` be a natural donor trajectory (`0=A`; `1=F`). For teacher
slot `l in {1,...,27}` and logical non-padding token `i`, let `t_l` be the
renormalized unit teacher direction and

```text
d_b(c,s,p,l,i) = t_l^T h_b^natural(c,s,p,l,i).
```

For `b=1` the full donor is set-independent. At each recipient hook:

```text
h64 = exact_float64(h)
h*  = h64 + t_l [d_b - t_l^T h64]
h+  = cast_to_original_float16(h*)
```

Teacher rows, donor coordinates, dot products, subtraction, and correction
are float64. Each teacher row is checked finite with norm greater than `1e-12`
and renormalized once in float64. There is exactly one cast back, after the
complete correction. There is no state normalization, alpha, dose sweep,
metric change, or hybrid-donor feedback. Padding positions are never written.

All 27 hooks are registered before the forward in ascending block order and
simultaneously replace every real token at a block output using immutable
natural-donor coordinates. This is an ordered joint intervention, not a
single natural mediator intervention.

## Hook, token, and forward semantics

In Transformers 4.48.3, a Qwen2 decoder block returns its first tuple element
after attention and MLP residual additions. Slot 1 is the post-forward output
of block 0; slot 27 is the output of block 26. These 27 outputs, and only these
outputs, are clamped. Slot 0, raw block-27 output, final RMSNorm, and slot 28
are not clamped. `post_attention_layernorm` is not a valid substitute.

The token set is exactly `attention_mask == 1` for the complete rendered ABD
prefill, including the explicit neutral system message, template/role tokens,
and assistant newline. Padding is left-sided, and logical token `i` maps to
physical index `T_batch - T_p + i`. The scientific batch plan follows prompt
file order in consecutive batches of six. The forward uses the frozen Qwen
4.48.3 direct-prefill position semantics, `use_cache=False`,
`past_key_values=None`, no generation, and no attention output. Condition,
parameter mask, set, and operator cell are not mixed within a batch.

## Four-state design and algebra

```text
Y00(c,s,p): parameters A; natural A coordinates
Y01(c,s,p): parameters A; natural F coordinates (rescue)
Y10(c,s,p): parameters F; natural A coordinates (reverse/noising)
Y11(c,s,p): parameters F; natural F coordinates
```

Scientific `Y00` and `Y11` are natural forwards. Self-replays are technical
controls and can never replace a diagonal. Define per raw location:

```text
E=Y11-Y00       L=Y01-Y00       H=Y11-Y10
B0=Y10-Y00      B1=Y11-Y01      I=Y11-Y10-Y01+Y00
E=L+B1=H+B0     I=H-L=B1-B0.
```

These are four-cell identities, not Transformer linearity assumptions.

## Estimands

For `q in {E,L,H,B0,B1,I}`:

```text
d_q(s,p) = q(S,s,p) - q(N,s,p)
g_q(p)   = d_q(top,p) - mean_j d_q(norm_j,p)
G_q      = mean_p g_q(p).
```

The sole primary hypothesis is the intersection-union equivalence test:

```text
H0: |G_B0| >= epsilon OR |G_B1| >= epsilon
H1: |G_B0| <  epsilon AND |G_B1| <  epsilon
epsilon = 0.04405641704135471.
```

Epsilon is an absolute SESOI for remaining preregistered `G` selectivity on
both parameter backgrounds. It is not a mediation share, a universal semantic
threshold, or the fraction of a whole trait explained.

Secondary reporting is exhaustive: all Y cells and six algebraic quantities;
90/95% intervals and the diagnostic simultaneous band; S and N top-minus-norm
components; every set and prompt family; distributions, mean absolute values,
RMS, and quantiles of both residuals; the six valid one-token candidate logits
(`owl, cat, dog, dolphin, elephant, lion`), their full-vocabulary
probabilities and candidate-set entropy; perturbation/numerical diagnostics;
both suffix controls; and all five orthogonal dose families. Secondary results
cannot rescue the primary endpoint.

## Statistical inference

The finite target is exactly these 72 known prompts. Bootstrap intervals only
describe robustness under exchangeability of similar prompts within the
three fixed families. They are not seed-population or universal-prompt
inference.

Exactly 20,000 stratified bootstrap draws use NumPy `Generator(PCG64)` seed
`20260804`. Each draw samples 24 prompt IDs with replacement within each
family. Identical indices are used for every cell, condition, set, estimand,
and control. Layers, tokens, and norm sets are not resampled. Invalid draws
are fatal and are never favorably replaced.

The IUT passes only if both marginal two-sided 90% percentile intervals for
`G_B0` and `G_B1` lie strictly inside `(-epsilon,+epsilon)`. Marginal 95%
intervals are also reported. For the fixed diagnostic family
`{G_B0,G_B1,G_L,G_H,G_I}`, marginal 99% intervals form the conservative
Bonferroni simultaneous 95% band. Boundary contact is not PASS; all labels
"material", "opposing", and "interaction" use that band.

## Ordered gates and cancellation

Gates are applied in this order: integrity; natural-effect reconstruction;
primary IUT; cancellation qualification; ordered outcome matrix; specificity.
Natural reconstruction requires `G_E<0`, its 95% upper bound below zero,
`|G_E+0.22028208520677353|<=epsilon`, and every natural per-prompt/set margin
within `1e-3` of its frozen FSD reference.

For each `B in {B0,B1}` define

```text
K_B,c(p)=B(c,top,p)-mean_j B(c,norm_j,p)
g_B(p)=K_B,S(p)-K_B,N(p)
A_B=mean_p |g_B(p)|; R_B=sqrt(mean_p g_B(p)^2)
G_B,f=mean_{p in family f} g_B(p).
```

A primary pass is cancellation-qualified if any preregistered rule holds:
`A_B>=epsilon`; `R_B>=2*epsilon`; any family absolute mean is at least
epsilon; any condition component absolute mean is at least epsilon; two family
means have opposite signs and differ by at least `2*epsilon`; S and N means
share a sign and each has magnitude at least epsilon while their difference is
smaller than epsilon; or a cross-cell output-collapse flag fires. For the last
rule, values are pooled over the complete identified `(condition,set,prompt)`
raw-row inventory: a Y01 or Y10 margin SD is at most 10% of the smaller pooled
natural Y00/Y11 SD (defined only above `1e-6`) while its mean absolute paired
deviation from Y11 (for Y01) or Y00 (for Y10) is at least epsilon.

With cancellation, the strongest allowed statement is only that aggregate G
selectivity is practically retained under this operator.

## Block-27 identification controls

The primary operator ends after block 26 while block 27 remains free. The top
set masks block-27 `self_attn.q_proj`; twelve norm sets mask block-27
`mlp.up_proj` or `mlp.gate_proj`. Therefore two complete controls are required.

For natural complete block-26 token states `H_b^nat`, the natural full-state
factorial runs

```text
Z_ab = suffix_a(H_b^nat), a,b in {0,1}.
```

`Z00` and `Z11` must reproduce natural outputs. For every executed C18
block-26 state `H_ab^C18`, the hybrid-state factorial runs

```text
W_abr = suffix_r(H_ab^C18), a,b,r in {0,1}.
```

`Y_ab=W_ab,a`. For fixed `b`, `B_b=W_1b1-W_0b0` is reported under both exact
allocations into upstream-state effect, suffix effect, and state-by-suffix
interaction. These are controlled cut experiments, not natural direct or
indirect effects. A rest is never automatically "non-teacher information".

## Confirmatory orthogonal dose controls

Five direction families are generated outcome-blind with NumPy
`Generator(PCG64(20260914))`, family-major then slot-major, drawing 3584
float64 standard normals for each slot. Each draw is projected off its
float64-renormalized `t_l`, rejected only if nonfinite or norm `<=1e-12`, then
renormalized in float64. The materialized artifact is hashed before outcome
access.

At the same hooks, with the same signed teacher correction
`alpha_T=d_T-t^T h`, family `k` executes

```text
h+ = cast_float16(h64 + alpha_T r_l,k).
D_k = max(|G_B0,k|,|G_B1,k|) - max(|G_B0,T|,|G_B1,T|).
```

This is an additive dose replay, not orthogonal-coordinate interchange. A
strong Teacher-specificity modifier requires the lower bound of every one of
the five 99% percentile intervals for `D_k` to be strictly positive. It is
secondary to the primary IUT.

## Numerical and implementation invariants

- The model stream remains float16/NF4; correction arithmetic is float64 with
  exactly one back-cast.
- Hook output equals `cast_float16(h*)` bit-for-bit at real tokens; padding is
  bit-identical to hook input.
- With cast error `e=cast_float16(h*)_64-h*`, coordinate error and orthogonal
  leakage are bounded by `||e||_2 + 64*eps64*max(1,||h*||_2)` per real token.
- Max/sum/count plus a lossless preregistered raw sample record preserve
  perturbation, coordinate, cast, and orthogonal-error evidence.
- Identity hook, null operator, and A-to-A/F-to-F self-replay yield bit-identical
  candidate logits on identical hardware and batches.
- Terminal full-state replay and natural Z diagonals differ by at most `1e-3`
  per candidate logit and margin; natural margins differ from FSD by at most
  `1e-3`.
- All Y/Z/W algebra holds per raw row and bootstrap draw within
  `64*eps64*max(1,sum(abs(terms)))`.
- NaN/Inf, degenerate directions, or any vanished executed correction whose
  ideal norm exceeds `1e-3` are fatal.
- The independent technical B=1 repetition may not classify outcomes and a
  maximum margin difference above `1e-3` is fatal.
- The primary margin is the frozen FSD `margin_direction` applied in float64
  to the final post-RMS state. Direct Cat/Lion logits are mandatory diagnostics.
- The PEFT census is exactly 196 canonical modules; each A mask disables the
  named 20 by zeroing only their active-adapter scaling. No merge is allowed,
  and all 196 scaling values are restored after success or exception.
- Donor and recipient buffers are separate, immutable, identified by IDs, and
  never inferred from array positions. Hooks are cleaned after success and
  exception. Publication is atomic.

Any failed hash, execution identity, tokenizer inventory, slot/token mapping,
cell/inventory, mask, cache, hook, replay, finite, algebra, suffix, or numerical
check is `technical/integrity failure` and produces no scientific
classification. Technical repair without outcome access may rerun v1 only if
the scientific contract is unchanged. Otherwise STOP and propose v2.

## Outcome matrix, claims, and exclusions

The complete ordered and disjoint classification is frozen in
[DECISION_MATRIX.md](DECISION_MATRIX.md). It is applied top to bottom after the
earlier gates. All valid rows report L/H/I, cancellation, five dose controls,
and block-27 decomposition as non-overriding modifiers.

Allowed claims are limited to this fixed seed-2 G contrast, the two controlled
background effects, their four-cell interaction, technical adequacy or failure
of the operator, and the exact suffix cut decomposition. A positive result is
not every-prompt or every-set reconstruction.

Forbidden claims include natural mediation; a single scalar explaining
Subliminal Learning; necessity/sufficiency for the whole Cat trait; treating
FSD shares as mediated percentages; calling every rest a feature,
high-dimensional, or non-teacher; inferring small interaction merely from two
small means; using nonsignificance as equivalence or specificity; generalizing
to seeds, traits, k, sufficiency, other models, prompts, or generation; and any
post-outcome selection of layers, tokens, families, directions, readouts,
dtypes, batches, or tolerances.

## Compute, sealing, and STOP boundary

The planned scale is approximately 75,000--100,000 prefill sequences and
1--4 A100 GPU hours with reserve up to 8 GPU hours. Controls or norm sets are
not dropped to meet budget.

Raw scientific cells are written only into the manifest's sealed namespace.
Technical validation has a separate namespace and schema that rejects Y cells,
candidate values, G estimands, classifications, and outcome-bearing logs.
Aggregation and audit require an explicit later scientific-release token and
operate independently from immutable raw cells. No scientific submit is
authorized by this preregistration.

STOP after public contract, manifest, implementation, tests, technical-only
validation, and execution-identity freeze. A scientific HTCondor submit,
outcome unsealing, result interpretation, final study tag, release, and merge
are separate actions requiring explicit authorization.

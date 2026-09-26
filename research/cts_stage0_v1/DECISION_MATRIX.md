# CTS Stage 0 — Decision matrix

This is a human-readable view of `cts_stage0_decision_spec.json`; the spec governs.

- The rows are evaluated top to bottom, and the first matching row decides.
- All criteria are computed for dog, wolf and c_cat,anim, regardless of the path taken.
- α_TS = 0.05/3. Every CI is two-sided at level 1 − α_TS: a prompt-level bootstrap stratified by family,
  n_boot = 10,000, seed 20260925.

## Criteria

| Criterion | Test |
|---|---|
| R | split-half cosine ≥ 0.95 at slot 14 |
| PC | raw t_cat added: CI_low Δℓ_{cat,O} > 0 and MC p < 0.05 vs the first 200 R_cov directions (norm ‖t_cat‖) |
| PC-pos | the PC test, with t_cat added at all prompt positions (prefill) |
| PC*-X / PC*-anim | unit(t_cat) at magnitude \|τ(c)\|: CI_low Δℓ_{cat,O} > 0 (O' for anim) |
| TS(c) | (a) R; (b) direction and reversibility; (c) every pairwise log-odds CI_low > 0; (d) > structured-null type-7 quantile and MC p < α_TS vs 1,000 R_cov; (e) signs hold at κ = 0.5 and with T2 and T3 (c_cat,X); for c_cat,anim at κ = 0.5 and with T3 only (one paraphrase; never described as multi-paraphrase robustness) |
| A4 | mean exp(L_X) over S0 animal prompts ∈ [0.01, 0.5] (heuristic) |
| A5 | TS(c_cat,X) |
| A6 | persona P_X unsteered vs default: CI_low Δℓ_{X,O} > 0 |
| A7 | raw t_X added: CI_low Δℓ_{X,O} > 0 |

## Decision order

| Rank | Class | Condition | Consequence |
|---|---|---|---|
| 1 | TECHNICAL_FAIL | any integrity check fails | at most one rerun, documented infrastructure cause only |
| 2 | STOP_INSTRUMENT | (not PC and not PC-pos) or not R(t_cat) | CTS stops; the estimator does not steer its own trait |
| 3 | INCONCLUSIVE_POSITION | not PC and PC-pos | new prereg for the steering position; no pivot |
| 4 | GO_X | first X in (dog, wolf) with A4 ∧ A5 ∧ A6 ∧ A7 | Stage 1 uses X; Stage-2a primary = c_cat,anim if TS, else c_cat,X |
| 5 | GO_CAT_ONLY | TS(c_cat,anim) or TS(c_cat,X) for some X | Stage 2a with c_cat,anim if TS, else the first passing c_cat,X; no Stage 1; no two-trait claim |
| 6 | INCONCLUSIVE_DOSE | PC*-dog, PC*-wolf and PC*-anim all fail (equivalent restatement, audit F02) | new prereg for the dose; no pivot |
| 7 | PIVOT_NO_BASE_VALIDATED_CONTRAST | otherwise | CTS as designed stops; DAS or model diffing under a separate prereg |

## Label (per passing contrast; limits claims, does not change the decision)

| Label | Condition |
|---|---|
| PREFERENCE_CONTRAST | CI_low[Δℓ_{cat,O}(+c) − Δℓ_{cat,O}(+m)] > 0 at the same magnitude. For c_cat,anim: against both m_cat,dog and m_cat,wolf, with O'. |
| LEXICAL_NOT_EXCLUDED | otherwise; claims say "trait-word contrast" only |

## Allowed claims per outcome

The allowed claims per outcome are in `audits/CTS_STAGE0_REVIEWER_2.md`, "Survivable claims". A pass never
licenses a claim about students.
For c_cat,anim, "paraphrase-robust" in those claims means robustness to one independent persona paraphrase (T3) only.

# CTS Stage 0 v2 — Draft audit record (2026-09-26)

Two independent audits of the draft (static; scientific), plus a check by the coordinator. The draft was
revised after the audits; the resolution column refers to the revised files.

## Static audit

No defects. Verified: all 9 v1 input hashes; byte-identical regeneration of registry and cost table; counts
(baseline 1, named 48 gating + 12 descriptive, structured null 720, R_cov 597, PC references 198, persona 3; total
1,579); fragility re-score set (118 conditions, 37,032 prompt-conditions); rank-rule arithmetic; cost arithmetic;
baseline rule; null names from `selected_16`; 41 personas; ladder disjoint and exhaustive; prompt-set rule.

## Scientific audit

| # | Finding | Sev. | Resolution |
|---|---|---|---|
| 1 | Fragility gate: per-statistic max at ε over ≈ 320 statistics fails with near certainty even in the accepted regime | blocking | Replaced with RMS(z) ≤ ε plus a max guard ε·Φ⁻¹(1 − 0.05/(2m)) (PREREGISTRATION §9.2, §13.1; spec `integrity.fragility.criterion`). Also found independently by the coordinator. |
| 2 | Plan at 0.976 of the authorization limit; small TV-v2 overrun blocks authorization; no pre-specified remedy | blocking | Pre-specified, outcome-blind reduction ladder (descriptive arms, then fragility subset 10 → 5); otherwise a dated researcher decision before freeze (§12). The limit itself is not changed. |
| 3 | g_anim contains t_cat (and A_len), so projecting onto G removes cat content; TS(f) failure would be ambiguous | blocking | g_anim redefined from the 16 held-out null-pool axes; every G component is held out (h cancels cat by construction); v1 g_anim descriptive only (§3; change #20). |
| 4 | TOST quantiles ambiguous | major | Wording made explicit: quantile of D⁺ at 1 − α_TS < 0, quantile of D⁻ at α_TS > 0 (§10.1). |
| 5 | No plan if L2 fails determinism | major | L1 only if its projection fits; otherwise researcher decision (§12). |
| 6 | Gram-Schmidt order and drop rule data-dependent | major | Partly wrong: the projector onto a span is basis-independent. Replaced Gram-Schmidt by thin SVD with a relative rank tolerance; order-free (§3). |
| 7 | A4–A7 not revisited | minor | Kept unchanged by design (v1 decisions in force); listed as a residual Reviewer-2 point. |
| 8 | O4 alternative remains | minor | Acknowledged; not a draft defect. |

The audit's claim "deviations from O2: none" is incomplete; the coordinator's list is in the handoff report.

## Verdict after revision

No known blocking defect in the draft. It is ready to start implementation; it is not ready to freeze (freeze
follows TV-v2 by design).

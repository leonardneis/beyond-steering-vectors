# Research index

The thesis baseline and subsequent research are deliberately separated. A new
study may extend, challenge, or refine the baseline, but it cannot silently
rewrite its hypotheses, artifacts, or conclusions.

## Studies

| Study | Status | Scope | Entry point |
|---|---|---|---|
| Cross-seed thesis baseline | Complete, audited, frozen | Three matched cat/neutral seeds; teacher alignment; 42-module screen; norm-matched causal interventions | `thesis-confirmatory-baseline` |
| [Parameter Formation v1](parameter_formation_v1/README.md) | Complete; Gate A passed | Full 196-module census; 25 random and 25 norm-control draws; teacher-vector robustness; no new training | [Results](parameter_formation_v1/RESULTS.md) |

Parameter Formation v1 completed on SIC HTCondor on 2026-08-02 as DAGMan
cluster `178878` with 52/52 successful nodes. Its results live in
`results/research/qwen7b_cat_parameter_hardening_v1` on shared cluster storage.
They remain post-baseline evidence until separately audited and interpreted.

## Research progression

Gate A has passed, but the next study still requires its own frozen contract:

1. complete: harden the full-pool parameter result and apply Gate A;
2. next: preregister a matched semantic-versus-subliminal study;
3. investigate checkpoint dynamics and optimizer effects;
4. test generalization across traits, model families, and adapter settings.

Every new study should provide:

- a versioned research contract before expensive execution;
- immutable source artifacts or explicit new-training provenance;
- disjoint selection and evaluation data where applicable;
- primary and secondary endpoints with evidence boundaries;
- an idempotent manifest or orchestration entry point;
- a dedicated output namespace and machine-readable completion status.

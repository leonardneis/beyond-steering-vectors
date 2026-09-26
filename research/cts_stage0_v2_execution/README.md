# CTS Stage 0 v2 — execution implementation

This directory documents how the v2 contract in [`../cts_stage0_v2/`](../cts_stage0_v2/PREREGISTRATION.md)
(decision spec + generated condition registry) is executed. It is implementation, not preregistration: nothing
here changes a criterion, threshold, condition, prompt set, persona, direction, dose, site, statistic or
endpoint. Engineering choices the contract leaves open are listed in
[`IMPLEMENTATION_CHOICES.json`](IMPLEMENTATION_CHOICES.json).

Status (2026-09-26): **implemented; TV-v2 prepared, not submitted; contract not frozen; no scientific
authorization.** No Stage-0 outcome exists. The branch starts from `master`; the v1 implementation files were
copied by content from commit `8afd41d` as a starting point and adapted (the branch is not a descendant of the
v1 implementation line). The reused v1 package [`../cts_stage0_v1/`](../cts_stage0_v1/) is byte-identical to tag
`prereg/cts-stage0-v1`; the submit script and every job verify this.

## Entry points

| Path | Role |
|---|---|
| `src/slgeo/cts_stage0/` | Implementation package (no PEFT import; base model only) |
| `src/slgeo/cts_stage0/contract.py` | Verified access to the v2 spec and registry; reused-v1-input hashes |
| `src/slgeo/cts_stage0/budget.py` | A100-h accounting and budget gate (standard library only; submit host) |
| `scripts/cts_stage0.py` | CLI: `submit-record` (submit host); `plan`, `run --shard`, `techval-cpu`, `techval`, `tv-project` (cluster) |
| `scripts/cts_stage0_budget.py` | DAG PRE-script budget gate, TV attempt registration, authorization check (submit host) |
| `scripts/generate_cts_stage0_dag.py` | TV-v2 and scientific DAGs |
| `condor/cts_stage0_task_{gpu,cpu}.sub`, `condor/run_cts_stage0_task.sh`, `condor/submit_cts_stage0.sh` | Node submit files, wrapper, submission |
| `configs/validation/cts_stage0_v2.yaml` | Execution manifest: contract pins, identity, snapshot hashes, budget, blackout |
| `tests/test_cts_stage0_*.py` | Unit, negative, layout, decision, orchestration tests |

## Scientific pipeline

`preflight` (CPU) → `extract_*` (GPU, 41 personas × 1,024 rows) → `directions` (CPU: bundle with τ, R_cov n = 199,
structured nulls, held-out g_anim, span G, c⊥G) → `baseline_rep1/rep2` (GPU, L2, two host groups) → `score_*`
(GPU, L2 or own-prefix per cost class) and `rescore_*` (GPU, L1 reference for the 118 flagged conditions) →
`fragility` (CPU, ε = 0.04 RMS and max guard) → `integrity` (CPU) → `analysis` (CPU, decision ladder ranks 1–8,
labels, Stage-2a handoff).

- **Canonical layout L2** (`scoring.build_prefix`, `scoring.score_from_prefix`): positions 0…L−2 once per prompt;
  per condition a one-token suffix forward with `steering.SuffixSteering`, then the flat-packed continuation.
- **Own-prefix / reference L1** (`scoring.score_prompt`, one row): full prefill with `PrefillSteering`.
- **Baseline rule:** L2 conditions against `unsteered`, own-prefix conditions against `persona:P_default`, L1
  re-scores against the L1 re-score of `unsteered` (`Condition.baseline`, `criteria.Context`).

## Engineering prerequisites E1–E5 (spec `engineering_prerequisites_for_authorization`)

| | Implementation | Test |
|---|---|---|
| E1 | `preflight.stage_preflight` raises on a failed marker; `RunContext.require_preflight` gates every stage | `test_failed_preflight_is_final_on_retry`, `test_stages_refuse_after_failed_preflight` |
| E2 | `errors.FinalFailure` on every integrity/refusal error; `errors.classify` → exit 86 (final), 1/75/85 (retry); DAG `RETRY 2 UNLESS-EXIT 86`; no in-place rematch | `test_error_classification`, `test_plan_dag_has_budget_gate_retry_and_abort` |
| E3 | `plan.per_shard`: planned compute × overhead ≤ 0.5 × retirement | `test_every_gpu_shard_fits_half_the_retirement_time` |
| E4 | `budget` + `scripts/cts_stage0_budget.py pre` on every node; job-ad attributes `BsvRunTag`, `BsvBudgetCategory`; BUDGET_STOP (exit 87, DAG abort, jobs refuse, integrity fails) | `test_pre_script_*`, `test_gate_and_budget_stop` |
| E5 | `artifacts.attempt_record` (immutable), `integrity.attempt_review` | `test_attempt_review*` |

Phase-F minors: cached-vs-uncached tolerance removed from TV (the in-run fragility check covers layout
robustness); hook verification at every slot > site, both sites, L1 and L2 hooks, three prompts; guards recompute
verdicts from stored per-check booleans; write-once by hard link (no check-then-rename window); CPU nodes verify the
container image; the condor venv is verified as content-addressed by the requirements hash.

## TV-v2

`condor/submit_cts_stage0.sh --technical-validation` (dry run by default; `--submit` to submit) creates a new
attempt directory `results/research/qwen7b_cts_stage0_v2_technical_validation/tv-<utc>` and a DAG
`tv_cpu → tv_gpu_a, tv_gpu_b, tv_dry → tv_project`. `tv_dry` is one planned L2 shard run as its own GPU job; the
PRE script of `tv_project` records its RemoteWallClockTime from `condor_history` (the quantity the scientific
ledger counts), and the per-job fixed cost F = wall − in-job compute enters the overhead factor. The projection
node writes `projection.json` with the measured seconds per cost class (maximum over hosts), F, the factor, the
projection P, the pre-authorization ladder and the gate P ≤ 0.8 × 30 A100-h. TV-v2 is limited to 3 attempts and
6 A100-h summed over all attempts.

## Freeze and authorization (later, researcher actions)

1. Fill the measured planning seconds, overhead factor and execution identity into the spec (freeze only), regenerate
   registry and cost table, write the v2 MANIFEST, tag `prereg/cts-stage0-v2`, set `contract.status: frozen`.
2. Commit `SCIENTIFIC_EXECUTION_AUTHORIZATION.json` naming the passing TV-v2 projection record.
3. `--scientific-plan --submit`, then `--scientific --submit`.

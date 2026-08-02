# Documentation index

This directory separates scientific interpretation from execution details. The
root [README](../README.md) introduces the project; this page routes readers to
the evidence and infrastructure behind its claims.

## Scientific record

| Document | Purpose |
|---|---|
| [Schema-v2 split stability and layer selection](notes/post-report/schema_v2_split_stability_and_layer_selection.md) | Corrected attribution schema, split robustness, and preregistered layer selection |
| [Top-k necessity and sufficiency](notes/post-report/Top-k_Necessity_Sufficiency_Results_and_Validation_Plan.md) | Initial causal module interventions and validation design |
| [Top-k validation and thesis conclusions](notes/post-report/Top-k_Validation_Thesis_Conclusions.md) | Saturation, matched controls, and evidential boundaries |
| [Behavioral validation architecture](notes/post-report/Behavioral_Validation_v2_Architecture.md) | Prompt-paired behavioral endpoints and causal definitions |
| [Behavioral results and implications](notes/post-report/Behavioral_Validation_v2_Results_and_Thesis_Implications.md) | Activation–behavior mediation gap and interpretation |
| [Cross-seed confirmatory design](notes/post-report/Cross_seed_confirmatory_experimental_design.md) | Three-seed manifest, compute plan, and confirmatory protocol |

The authoritative baseline is the finalized, checksum-audited confirmatory
archive associated with Git tag `thesis-confirmatory-baseline`. Exploratory
outputs and scheduler logs are not substitutes for finalized report tables.

## Infrastructure

| Document | Purpose |
|---|---|
| [Cluster environment](cluster_environment.md) | SIC storage, environment, submission, monitoring, recovery, and cancellation |
| [HTCondor migration](HTCondor_Migration.md) | Rationale and implementation of the native DAGMan workflow |

## Thesis workspace

The local `docs/thesis/` workspace contains the evolving thesis report and its
chapter-oriented knowledge base. Most of that workspace is intentionally
Git-ignored so drafting can proceed independently of the frozen experimental
baseline. Any numerical thesis claim should trace back to the authoritative
artifact contract, not merely to a prose draft.

## Post-baseline research

New experiments are indexed under [`research/`](../research/README.md). Each
study must keep its hypotheses, prompt partitions, decision rules, manifest,
and output namespace separate from the thesis baseline.

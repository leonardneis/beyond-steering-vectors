# Script guide

The scripts are command-line entry points over the reusable `slgeo` package.
Prefer a manifest-driven workflow for final studies; individual commands are
most useful for inspection, pilots, and reproducible subanalyses.

## Entry points by task

| Task | Script or command |
|---|---|
| Inspect the minimal end-to-end pipeline | `python scripts/run_minimal_reproduction.py --dry-run` |
| Train a configured student adapter | `scripts/train_student.py` |
| Evaluate target preference | `scripts/evaluate_preference.py` |
| Extract teacher and student directions | `scripts/extract_steering_vectors.py` |
| Attribute activation to LoRA groups | `scripts/run_lora_attribution.py` |
| Prepare top-k and control module sets | `scripts/prepare_topk_module_sets.py` |
| Run activation interventions | `scripts/run_lora_set_interventions.py` |
| Run behavioral interventions | `scripts/run_lora_set_behavior.py` |
| Plan the frozen confirmatory study | `scripts/run_confirmatory_manifest.py` |
| Plan Parameter Formation v1 | `scripts/run_parameter_hardening_manifest.py` |
| Plan Activation--Behavior Dissociation v1 | `scripts/run_activation_behavior_dissociation_manifest.py` |
| Aggregate the dissociation decision gate | `scripts/aggregate_behavior_dissociation.py` |
| Plan Final-State Directional Causal Decomposition v1 | `scripts/run_final_state_directional_decomposition_manifest.py` |
| Extract final states and fixed directional patches | `scripts/run_final_state_directional_decomposition.py` |
| Aggregate the final-state decomposition | `scripts/aggregate_final_state_directional_decomposition.py` |
| Audit the completed final-state study | `scripts/audit_final_state_directional_decomposition.py` |
| Send an optional runtime ntfy notification | `scripts/notify.py` |
| Add reusable notification finalization to a runtime DAG | `scripts/dag_notifications.py` |

## Workflow contracts

- Scripts resolve repository-relative paths through `_bootstrap.py`.
- Final workflows refuse silent overwrites and record input/output provenance.
- Subliminal and neutral conditions must use identical prompts, grouping, and
  the same frozen teacher vector for a paired comparison.
- Selection plans use explicit module identities; repeated controls additionally
  carry deterministic `draw_id` values.
- Aggregation should consume finalized per-seed artifacts, never live logs.

For cluster submission and monitoring, see the
[cluster documentation](../docs/cluster_environment.md). For scientific scope
and current studies, see the [research index](../research/README.md).

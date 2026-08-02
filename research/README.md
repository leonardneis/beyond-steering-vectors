# Research index

This directory is the public record of the repository's research program. It
contains reproducible study contracts and audited results, not personal
roadmaps, thesis drafting material, or exploratory interpretation notes.

## Completed studies

| Study | Status | Frozen source | Entry point |
|---|---|---|---|
| [Cross-seed confirmatory baseline](confirmatory_baseline/README.md) | Complete, audited, frozen | `thesis-confirmatory-baseline` (`1635e5a`) | [Results](confirmatory_baseline/RESULTS.md) |
| [Parameter Formation v1](parameter_formation_v1/README.md) | Complete; Gate A passed | Execution commit `6fb39d6`; final report commit `6de222c` | [Results](parameter_formation_v1/RESULTS.md) |
| [Activation--Behavior Dissociation v1](activation_behavior_dissociation_v1/README.md) | Complete; integrity and strong-replication gates passed | `study/activation-behavior-dissociation-v1` (execution commit `ac7640a`) | [Results](activation_behavior_dissociation_v1/RESULTS.md) |

Parameter Formation v1 is separately versioned post-baseline evidence. It does
not retroactively change the frozen baseline artifacts or hypothesis labels.

## Active study

| Study | Status | Parent | Entry point |
|---|---|---|---|
| [Final-State Directional Causal Decomposition v1](final_state_directional_causal_decomposition_v1/README.md) | Prospective; implemented and awaiting execution decision | Activation--Behavior Dissociation v1 | [Research contract](final_state_directional_causal_decomposition_v1/PREREGISTRATION.md) |

This study performs no new training, module selection, or localization sweep.
It decomposes the inherited seed-2 final-state margin effect into the frozen
teacher axis and its orthogonal complement at the exact LM-head input.

## Creating a study

Copy [`TEMPLATE/`](TEMPLATE/) into a new, uniquely named directory under
`research/`. At minimum, each study contains:

- `README.md`: identity, scope, status, entry point, and artifact contract;
- `PREREGISTRATION.md`: frozen questions, endpoints, controls, splits, and
  decision rules;
- `RESULTS.md`: integrity, preregistered results, deviations, interpretation,
  and evidence boundaries.

Add further files only when they carry information that the manifest,
provenance, or these three documents cannot express clearly.

Every study must use a new experiment ID and output namespace. A new study may
extend or challenge an earlier result, but it never overwrites an existing
study directory or finalized artifact tree.

## Versioning

- `prereg/<study-slug>`: annotated tag for the frozen contract before expensive
  execution or outcome inspection;
- `study/<study-slug>`: annotated tag for the complete, audited study state;
- `thesis/<milestone>`: thesis submission or other thesis-level snapshot.

Tag names remain short; complete model, data, and execution identities belong
in the manifest and provenance. Tags are immutable and must never be moved or
reused. The existing `thesis-confirmatory-baseline` tag is retained as the
historical baseline name.

GitHub Releases are reserved for major scientific milestones, such as the
confirmatory baseline, Parameter Formation v1, and the final master thesis. A
fine-grained tag does not automatically require a Release.

## Public evidence standard

Each completed study must provide:

- a frozen manifest or equivalent executable contract;
- immutable source artifacts or explicit new-training provenance;
- disjoint selection and evaluation data where applicable;
- primary and secondary endpoints with evidence boundaries;
- an idempotent orchestration entry point;
- a dedicated output namespace;
- machine-readable completion and checksum evidence;
- a concise public result report tied to the finalized artifacts.

Internal interpretations, literature notes, supervisor material, personal
roadmaps, and thesis knowledge belong in the Git-ignored `.local-research/`
workspace and must not be linked from public documentation.

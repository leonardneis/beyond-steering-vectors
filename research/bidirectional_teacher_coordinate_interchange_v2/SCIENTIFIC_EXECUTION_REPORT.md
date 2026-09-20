# C18 v2 Scientific Execution Report

Date: 2026-09-21  
Study: `qwen7b_cat_bidirectional_teacher_coordinate_interchange_v2`  
Authorization commit: `21112195759957917b1672c08f0fa716c640bd1b`  
Final status: `C18 V2 EXECUTION BLOCKED`

## Authorization

The researcher decision `C18 V2 SCIENTIFIC EXECUTION AUTHORIZED` was recorded
in `SCIENTIFIC_EXECUTION_AUTHORIZATION.json`. The record is bound to the frozen
preregistration, decision matrix, manifest, calibration summary and audit,
technical validation bundle, validated execution commit, container digest, and
scientific input hashes. The authorization record does not authorize outcome
release, Seeds 1/3, merge, tag, or release.

## Final preflight result

Scientific execution stopped before DAG creation or submission. The frozen
authorization mechanism contains mutually incompatible fail-closed predicates:

1. `validate_manifest_contract` requires the manifest field
   `scientific_execution_authorized` to be exactly `false`.
2. The scientific DAG generator and the scientific runner require that same
   manifest field to be exactly `true`.
3. The frozen release-gate inventory also requires the manifest field to be
   `true`.

The exact frozen manifest has SHA-256
`6c8353788690a683b0447f72efb0980ecf87c6c3d07df4ff4e6db7842d121ed0`
and contains `scientific_execution_authorized: false`. The scientific DAG
generator therefore terminated with:

`STOP: refusing to generate a scientific DAG from an unauthorized manifest`

Changing the manifest flag would change the frozen manifest hash. Changing the
validator, DAG generator, or runner would change frozen implementation hashes
and invalidate the exact implementation covered by the technical validation.
Neither repair is permitted after authorization under the frozen scientific
contract, so no repair was attempted.

## Expected raw inventory

The preregistered complete two-condition inventory remains:

| Category | Expected unique raw cells |
|---|---:|
| Y | 14,976 |
| Z | 14,976 |
| W | 29,952 |
| Orthogonal specificity | 37,440 |
| Total | 97,344 |

No part of this inventory was generated.

## Outcome-blind stop record

- No scientific DAG was created.
- No scientific job was submitted.
- No sealed scientific raw artifact exists for either condition.
- No outcome-release token was created.
- No aggregation was run.
- No independent scientific audit was run.
- No scientific value or partial result was opened, interpreted, or reported.
- No merge, study tag, GitHub release, Seed-1/3 replication, or follow-up study
  was started.

## Phase disposition

| Phase | Status |
|---|---|
| 1 — Authorization Record | PASS |
| 2 — Final Preflight | FAIL-CLOSED |
| 3 — Scientific Raw Execution | NOT STARTED |
| 4 — Inventory Freeze | NOT STARTED |
| 5 — Aggregation | NOT STARTED |
| 6 — Independent Audit | NOT STARTED |
| 7 — Result Opening | NOT STARTED |

The required stop condition is:

`STOP — SCIENTIFIC EXECUTION NOT STARTED`

Resolving the incompatible authorization predicates requires a new prospective
technical-contract decision and revalidation. It cannot be performed as an
outcome-dependent or post-authorization repair of this frozen execution.

# C18-v2 Scientific Authorization V3 Report

Date: 2026-09-21  
Study: `qwen7b_cat_bidirectional_teacher_coordinate_interchange_v2`  
Execution commit: `287d9c70556fb6dc1735b0794108be58f293fe10`  
Final status: `C18 V2 EXECUTION BLOCKED`

## Corrected authorization

The append-only record `SCIENTIFIC_EXECUTION_AUTHORIZATION_V3.json` contains
the canonical researcher decision:

`C18 V2 SCIENTIFIC EXECUTION AUTHORIZED`

The two earlier authorization records and both earlier blocked reports remain
unchanged. The new record binds the already revalidated execution commit and
technical bundle and retains every restricted scope flag as `false`.

## Unchanged authorization-gate result

The canonical decision passed the decision-string predicate. The unchanged
common authorization gate then stopped during its runtime scientific-input
verification with:

`STOP: scientific adapter differs: subliminal`

The cluster inventory confirms that both frozen Seed-2 adapters exist under the
configured shared root, while neither exists below the repository checkout:

- subliminal: repository checkout absent; shared root present
- neutral: repository checkout absent; shared root present

The scientific DAG generator invokes the common gate with the unmodified
manifest paths rooted at the repository checkout. It does not apply the shared
storage override that is used by the runner and technical preflight. Therefore
the unchanged gate cannot pass in the intended cluster layout.

The instructions prohibit changing code or adapting the record after a newly
discovered gate error. No repair was attempted.

## Disposition

- Authorization Gate: FAIL-CLOSED
- Final Scientific Preflight: NOT COMPLETED
- Scientific DAG submission: NOT STARTED
- Scientific raw execution: NOT STARTED
- Inventory freeze: NOT STARTED
- Aggregation and scientific audit: NOT STARTED
- Result opening: NOT STARTED

No scientific result, raw cell, teacher interchange, Y01/Y10 value, or
G-estimand was generated or opened.

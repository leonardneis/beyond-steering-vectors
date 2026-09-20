# C18-v2 Execution-Control Repair

## Scope and root cause

The first scientific-execution attempt is permanently recorded as
`C18 V2 EXECUTION BLOCKED` in `SCIENTIFIC_EXECUTION_REPORT.md`. No scientific
DAG, raw cell, teacher interchange, outcome, estimand, merge, tag, or release
was produced.

The frozen manifest is internally consistent as a preregistered scientific
contract: `scientific_execution_authorized` is and remains `false`. The defect
was in the execution-control layer. Its manifest validator required that frozen
value, while the scientific DAG generator, manifest executor, and direct runner
required the same frozen field to become `true`. The release-gate inventory
repeated the impossible predicate. Mutating the manifest would have invalidated
its frozen SHA-256 and is not an admissible repair.

## Repaired architecture

The frozen scientific contract and later researcher authorization are separate:

1. The byte-identical manifest continues to describe the unauthorized state at
   freeze time.
2. A schema-v2 external authorization record is the only source of later
   execution authority.
3. DAG generation, manifest execution, the direct scientific runner,
   aggregation, and independent scientific audit all call the same fail-closed
   validator.
4. The validator binds the Study-ID, exact researcher decision, exact manifest,
   preregistration, decision matrix, calibration bundle, technical validation
   bundle, container, complete scientific-input identity, and exact execution
   commit. It requires all non-execution scopes to remain false.
5. The historical schema-v1 authorization remains unchanged. A future schema-v2
   record must name its SHA-256 and the blocked status, making supersession
   explicit and append-only.

## Prospective execution-commit bootstrapping

The repair cannot be authorized by the historical record because that record
binds the pre-repair commit. The cycle is resolved prospectively and
reproducibly:

1. Commit only the enumerated execution-control repair and its outcome-blind
   tests/documentation.
2. Revalidate that exact clean repair commit technically, without scientific
   prompts, teacher interchanges, raw cells, Y01/Y10 values, or G estimands.
3. Produce an independent technical audit bound to that same repair commit.
4. Only then may the researcher create a new schema-v2 authorization record.
   The record must bind the exact repair commit and all hashes from the new
   technical bundle, and must explicitly supersede (not replace) the historical
   authorization.
5. Scientific DAG generation and every downstream entry point require that
   exact record. Its successor policy names both the original frozen execution
   commit and the immutable blocked-attempt base commit `4a930b1`; only the diff
   after that audit-trail base may touch the enumerated control paths. A
   different commit, missing record, incomplete record, or any change outside
   that fixed path set stops execution.

The authorization record itself may be committed after the execution commit:
it is external authority, not part of the execution tree. At execution time it
is supplied to a checkout of the exact authorized execution commit. This avoids
self-referential commit hashing while retaining an exact, prospective binding.

No schema-v2 authorization record is created by this repair. Creating it is a
new explicit researcher decision after technical revalidation.

## Canonical scientific-artifact path resolution

The subsequent authorization preflight exposed a storage-resolution mismatch:
the validated cluster pipeline maps logical `data/`, `results/`, and `runs/`
paths below `SLGEO_SHARED_ROOT`, while the authorization validator had joined
every logical path to the repository checkout. The latter would have failed not
only for both adapters but also for the selection plan, teacher tensor, and the
three state/aggregate FSD result artifacts.

One canonical resolver now defines the contract. Relative paths whose first
component is `data`, `results`, or `runs` resolve deterministically below the
configured shared root; every other relative path resolves below the repository
root. Absolute paths are accepted only when they are already below the exact
configured shared root and retain one of those explicit storage classes. Empty
segments, `.`/`..`, root escape, and arbitrary absolute paths are rejected.
There is no search, content-based fallback, or repository fallback for a
shared-root artifact. Tree/SHA-256 validation remains mandatory after physical
resolution.

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

## Second execution-control repair (2026-09-24)

An outcome-blind recovery audit of `4ea9af1` found three deterministic
execution-control blockers. None of them touches the scientific contract. The
manifest, preregistration, decision matrix, calibration summary, frozen
scientific inputs, and the hash-bound `c18_v2_execution`, `c18_v2_statistics`,
and `c18_v2_independent_audit` modules are byte-identical to the freeze.

1. **Self-referential authorization path.** The scientific task wrapper and the
   command plan of the manifest executor passed the tracked
   `SCIENTIFIC_EXECUTION_AUTHORIZATION_V2.json`, while the DAG generator
   validated whatever record path it was given.
   A record tracked in the execution commit cannot contain that commit's hash,
   and the node-side checkout validator rejects a modified or untracked,
   unignored record. No record could therefore pass at that path. The
   statement above that the record "is supplied to a checkout" was not
   implemented. The historical V2/V3 records also fail the gate: V2 carries a
   different decision string, and both bind `287d9c7` with the earlier 12-path
   policy.
2. **Git executable inside the container.** The successor-policy check started
   `git merge-base` and `git diff` subprocesses. The container image does not
   guarantee a Git executable; the checkout validator was moved to dulwich
   earlier for the same reason. The check would have raised an uncontrolled `FileNotFoundError` on
   the GPU node.
3. **GPU identity not schedulable.** The GPU submit description did not
   constrain the device or driver, while the runtime identity check on every
   GPU node enforces the frozen `NVIDIA A100-PCIE-40GB` and driver
   `570.211.01`. Six outcome-blind technical
   validation attempts at `4ea9af1` on 2026-09-21 (HTCondor jobs 195134,
   195135, 195136, 195140, 195142, 195143) matched `NVIDIA A100-SXM4-80GB`
   slots and stopped with `execution identity mismatch for gpu_class`. They
   produced no validation record, no raw cell, and no outcome.

The repair changes only the execution-control layer:

- The scientific task wrapper, the scientific DAG generator, and the manifest
  executor use only
  `<output root>/authorization/SCIENTIFIC_EXECUTION_AUTHORIZATION.json` below
  `SLGEO_SHARED_ROOT`, outside the checkout. The generator and the manifest
  executor no longer accept a free record or technical-bundle path, so they
  validate exactly the record and technical bundle that the GPU nodes read.
  Direct invocations of the runner, aggregator, and auditor still take explicit
  paths and still validate the supplied record completely.
- Before rendering, the scientific DAG generator also requires that
  `SLGEO_SHARED_ROOT` equals the shared root passed to the nodes, that the
  submit checkout is clean at the authorized execution commit, and that the DAG
  file is written below the ignored `condor/runtime/`.
- The ancestry and changed-path successor check reads Git objects through
  dulwich, from the object store only. Any repository or object error stops
  with a controlled message. The node checkout needs complete history back to
  the frozen commit.
- The scientific runner rejects an absent or malformed sealing key and an
  existing sealed output or provenance file before model loading, so a retry
  can neither overwrite nor recompute a completed condition.
- The GPU submit description pins the frozen execution identity at
  matchmaking: `require_gpus` and `requirements` both demand
  `NVIDIA A100-PCIE-40GB` with NVIDIA driver `570.211.01`. The runtime identity
  check stays authoritative.
- The GPU submit description is added to the enumerated execution-control
  successor paths. Its manifest hash therefore no longer binds it; the
  successor policy of the next authorization record binds it instead.

**Disclosed protocol deviation.** The preregistration's "Execution lock"
section says no runner may create scientific cells while the manifest value
`scientific_execution_authorized` is `false`. Under the external-authorization
architecture this value stays `false` by design, and execution authority comes
only from the external schema-v2 record. This is an execution-governance
deviation, not a change of any estimand, gate, classification rule, or the
decision matrix.

The new execution commit requires a new outcome-blind technical validation, an
independent technical audit, and a new schema-v2 record, created by the
researcher. The record must bind that commit, the sorted successor paths, and
the new technical bundle. This repair creates no record.

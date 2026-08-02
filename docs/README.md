# Documentation index

The public documentation contains technical and reproducibility information.
Scientific study contracts and audited results live under [`research/`](../research/README.md).
Personal interpretation, thesis drafting, literature notes, and research
roadmaps are intentionally excluded from Git.

## Scientific record

| Study | Contract and results |
|---|---|
| Cross-seed confirmatory baseline | [`research/confirmatory_baseline/`](../research/confirmatory_baseline/README.md) |
| Parameter Formation v1 | [`research/parameter_formation_v1/`](../research/parameter_formation_v1/README.md) |

The authoritative baseline source is the finalized, checksum-audited archive
associated with Git tag `thesis-confirmatory-baseline`. Preliminary outputs,
personal notes, and scheduler logs are not substitutes for finalized report
tables.

## Infrastructure

| Document | Purpose |
|---|---|
| [Cluster environment](cluster_environment.md) | SIC storage, environment, submission, monitoring, recovery, and cancellation |
| [HTCondor migration](HTCondor_Migration.md) | Rationale and implementation of the native DAGMan workflow |

## Reproducibility principles

- Every study has a unique manifest and output namespace.
- Preregistration, execution, and finalization commits are recorded explicitly.
- Large generated artifacts remain outside Git but are identified by stable
  paths, checksums, provenance, and completion markers.
- Final claims trace to audited artifacts, not to prose drafts or exploratory
  notebooks.
- Completed studies and their tags are immutable; extensions receive a new
  study directory and version.

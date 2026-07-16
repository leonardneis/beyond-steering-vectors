# Migration from SLURM to SIC HTCondor

Commit `a231ccf` introduced a SLURM-oriented HPC layer. SIC's confirmed native
scheduler is HTCondor, so that submission layer has now been replaced rather than
kept as the primary backend.

| Former component | HTCondor replacement |
|---|---|
| `slurm/*.sbatch` arrays | generated DAGMan nodes and queue-independent stable task IDs |
| `sbatch --dependency=afterok` | `PARENT ... CHILD ...` edges in `condor/confirmatory.dag` |
| `--gres=gpu:1` | `request_GPUs = 1` plus memory/capability requirements |
| partitions/accounts/modules | Docker universe and portable pinned environment |
| `$SLURM_TMPDIR` | node-local `/tmp` staging |
| SLURM job/array IDs | Condor ClusterId, ProcId, and `+TaskId` ClassAds |
| `squeue` | `condor_q` merged into `status.json`/`status.md` |
| SLURM final dependency job | DAGMan final aggregation node |

Unchanged scientific guarantees:

- 62 individually addressable manifest commands;
- deterministic seeds and disjoint prompt offsets;
- explicit checkpoints and resume;
- validated atomic publication and completion markers;
- hash/provenance sidecars;
- shared HF cache, retries, status and ETA;
- final cross-seed aggregation and plots;
- local PowerShell execution.

SLURM files and references were removed to avoid presenting an incorrect SIC
execution path. The Python orchestration remains scheduler-neutral and could be
adapted elsewhere without changing the scientific commands.

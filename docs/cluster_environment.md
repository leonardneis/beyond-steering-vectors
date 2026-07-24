# SIC HTCondor environment and confirmatory DAG

HTCondor with Docker jobs is the primary SIC backend. The scientific commands
remain manifest-driven and identical to local PowerShell execution; only storage
mapping and scheduling differ.

## Storage contract

- Repository and small Condor metadata: `/home/$USER/beyond-steering-vectors`
- Persistent datasets, checkpoints, results, environments and HF cache:
  `/scratch/compuling/$USER/beyond-steering-vectors`
- Per-execution staging only: `/tmp/slgeo-*`

Prepare shared storage once on a SIC login node:

```bash
cd "$HOME/beyond-steering-vectors"
export SLGEO_SHARED_ROOT="/scratch/compuling/$USER/beyond-steering-vectors"
mkdir -p "$SLGEO_SHARED_ROOT"/{data,results,runs,huggingface,envs}
rsync -a data/ "$SLGEO_SHARED_ROOT/data/"
rsync -a results/geometry/ "$SLGEO_SHARED_ROOT/results/geometry/"
rsync -a results/reference_reproduction_4080/ \
  "$SLGEO_SHARED_ROOT/results/reference_reproduction_4080/"
./condor/repair_scratch_group.sh "$SLGEO_SHARED_ROOT" compuling
cp condor/condor.env.example condor/condor.env
```

Run the repair helper after any `scp -r`, archive extraction, or mode-preserving
sync. SIC charges files below this namespace to group `compuling`; copied
directories that lose their setgid bit can otherwise make the next write fail
with the misleading message `Disk quota exceeded`.

The runtime maps manifest paths beginning with `data/`, `results/`, or `runs/`
into `SLGEO_SHARED_ROOT`. Training checkpoints therefore live directly on
persistent `/scratch`. Analysis outputs are first written under `/tmp`, validated,
copied to an `.incoming` path on `/scratch`, and atomically renamed. `/tmp` is
never treated as resumable storage.

## Container and dependencies

The portable fallback is
`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`. Exact Python dependencies are
pinned in `condor/requirements-condor.txt`. On first use, the wrapper creates a
content-addressed virtual environment under
`$HOME/.cache/beyond-steering-vectors/envs`; `flock` prevents concurrent DAG
jobs from racing during installation. Keeping dependency files in Home avoids
consuming the limited Scratch artifact quota.

For reliable offline operation, build and publish the provided image first:

```bash
docker build -f condor/Dockerfile -t REGISTRY/beyond-steering-vectors:condor-v1 .
docker push REGISTRY/beyond-steering-vectors:condor-v1
python scripts/generate_condor_dag.py \
  --container-image REGISTRY/beyond-steering-vectors:condor-v1
```

SIC compute nodes cannot fetch missing Hugging Face model files. Populate the
shared cache from a login node before submitting GPU jobs:

```bash
export SLGEO_SHARED_ROOT="/scratch/compuling/$USER/beyond-steering-vectors"
nohup ./condor/stage_qwen_cache.sh "$SLGEO_SHARED_ROOT" \
  > "$SLGEO_SHARED_ROOT/huggingface/qwen_download.log" 2>&1 &
tail -f "$SLGEO_SHARED_ROOT/huggingface/qwen_download.log"
```

The staging script pins the model revision, resumes partial files, downloads the
four shards concurrently, validates the file set, and publishes it atomically.
Condor jobs set `SLGEO_OFFLINE=1` explicitly. They never assume a Windows
`.venv`, CUDA modules, SBATCH, partitions, accounts, hostnames, DDP, DeepSpeed,
or multiple GPUs.

## GPU smoke test

The smoke job requests exactly one GPU with at least 16 GiB and checks CUDA,
bitsandbytes, a real 4-bit Qwen load, repository imports, model-cache access, and
atomic `/scratch` writes:

```bash
mkdir -p condor/logs
condor_submit condor/gpu_smoke.sub
condor_q -nobatch
```

Run this before the full DAG. The resulting JSON is written below
`$SLGEO_SHARED_ROOT/smoke/`.

## Generate and validate without submitting

```bash
./condor/submit_confirmatory.sh --dry-run
```

This single fail-closed preflight normalizes the `compuling` quota group and
setgid inheritance, verifies the size and SHA-256 of every catalogued dataset,
model-cache object, Seed-1 adapter, and geometry cache, requires the
content-addressed environment and successful real-model GPU smoke artifact,
checks exactly 62 unique tasks, acyclicity, all parents, submit templates, four
independent training nodes, zero-or-one GPU per task, and the 16 GiB minimum for
every GPU node. It finally runs native `condor_submit_dag -no_submit -f` syntax
validation and submits nothing.

## Submit

```bash
./condor/submit_confirmatory.sh --submit
```

This is the only launch command. It repeats the full preflight immediately
before submission, so missing, modified, wrongly owned, or incompletely staged
inputs cannot start the DAG.

DAGMan dependencies implement:

```text
prepare(condition)
  -> four independent single-GPU adapter training jobs
  -> per-condition behavioral evaluations -> paired positive-CI gate
  -> {vector extraction, layer attribution}
  -> restricted module attribution -> k=10/20 set construction
  -> {activation interventions, target-logprob behavioral validation}
  -> aggregate + plots + checksums + CPU tests
```

A failed paired behavioral gate prevents every descendant for that seed from
running. Other independent branches may finish and remain resumable.

## Resources and rescheduling

All GPU tasks use `request_GPUs = 1`. Training and attribution use HTCondor's
portable `gpus_minimum_memory = 16384` and
`gpus_minimum_capability = 7.0` submit commands, without naming a GPU or host.
Defaults are:

| Profile | CPUs | RAM | GPUs | Minimum VRAM |
|---|---:|---:|---:|---:|
| preparation | 2 | 8 GiB | 0 | n/a |
| comparison | 4 | 16 GiB | 0 | n/a |
| training | 8 | 48 GiB | 1 | 16 GiB |
| evaluation | 4 | 32 GiB | 1 | 16 GiB |
| attribution | 8 | 48 GiB | 1 | 16 GiB |
| aggregation | 4 | 16 GiB | 0 | n/a |

`max_job_retirement_time = 7200` provides roughly one useful epoch. On SIGTERM,
the Trainer requests a Hugging Face checkpoint at the next step and exits with
code 75 without a completion marker. Re-execution adds `--resume` and finds the
newest valid checkpoint. Analysis jobs simply restart from disposable `/tmp`.

The templates use shared filesystems (`should_transfer_files = NO`), so Condor
file transfer is intentionally disabled. If a future deployment switches to file
transfer, it must set `when_to_transfer_output = ON_EXIT_OR_EVICT`; checkpoints
must still remain on `/scratch` rather than only in the transfer sandbox.

## Monitoring and operations

Machine/human status combines completion markers with live `condor_q` ClassAds:

```bash
./condor/monitor_confirmatory.sh 30
```

Useful native commands:

```bash
condor_q -nobatch
condor_q -run -constraint 'RequestGPUs > 0' \
  -af ClusterId ProcId TaskId RemoteHost RequestGPUs AssignedGPUs
condor_q -hold
condor_q CLUSTER.PROC -better-analyze
condor_q CLUSTER.PROC -af HoldReason HoldReasonCode HoldReasonSubCode
condor_tail -f CLUSTER.PROC
tail -f condor/logs/TASK_ID.CLUSTER_ID.PROC_ID.out
tail -f condor/logs/TASK_ID.CLUSTER_ID.PROC_ID.err
condor_release CLUSTER.PROC
condor_rm CLUSTER.PROC
```

The continuously refreshed overview is stored at
`$SLGEO_SHARED_ROOT/results/confirmatory/qwen7b_cat_cross_seed_v1/status.md`
and its machine-readable counterpart `status.json`. Final aggregation publishes
`aggregate.json`, thesis plots, and `final_artifacts.sha256` below the same run
root only after all required DAG parents and gates succeed. ClusterCockpit at
<https://hpc-monitoring.cs.uni-saarland.de> shows historical GPU utilization,
memory, runtime, and host telemetry for the recorded ClusterId/ProcId.

To cancel a full DAG, remove descendants and then DAGMan using the DAGMan cluster
ID shown by `condor_submit_dag`:

```bash
condor_rm -constraint 'DAGManJobId == DAGMAN_CLUSTER_ID'
condor_rm DAGMAN_CLUSTER_ID
```

DAGMan writes rescue files beside the DAG. After fixing a held/failing cause:

```bash
condor_submit_dag -dorescuefrom N condor/confirmatory.dag
```

Completed nodes remain idempotent because their markers are checked before work.
Use SIC ClusterCockpit for historical GPU utilization, memory, runtime, and host
telemetry: locate jobs by Condor ClusterId/ProcId recorded in each provenance
sidecar. ClusterCockpit complements, but does not replace, completion markers.

## SIC-specific validation

The templates require `UidDomain == "cs.uni-saarland.de"`, request GPU home and
scratch mounts, and inherit only `HOME`. Before first submission verify that the
pool advertises GPUs to the submit host:

```bash
condor_status -constraint 'TotalGPUs > 0' \
  -af Machine UidDomain AvailableGPUs
```

The standard `gpus_minimum_memory` and `gpus_minimum_capability` submit commands
are converted by HTCondor into device-property constraints. If the SIC pool has a
site-specific policy on top, change only the two GPU `.sub` templates after
confirming it with the SIC documentation or administrators.

## Upstream references

- [HTCondor submit description reference](https://htcondor.readthedocs.io/en/latest/man-pages/condor_submit.html)
- [DAGMan submission and no-submit validation](https://htcondor.readthedocs.io/en/latest/man-pages/condor_submit_dag.html)
- [DAGMan completion and rescue behavior](https://htcondor.readthedocs.io/en/latest/automated-workflows/dagman-completion.html)
- [ClusterCockpit job monitoring overview](https://clustercockpit.org/docs/overview/)

# SIC/UdS SLURM environment and HPC-first execution

The repository now uses the same manifest and Python commands locally and on
SLURM. Site-specific values are deliberately isolated in an untracked file:

```bash
cp slurm/sic_cluster.env.example slurm/sic_cluster.env
```

Before submission, confirm the actual SIC `--partition`, `--account`, GPU GRES,
maximum wall time, CUDA module, outbound-network policy, shared-cache path, and
scratch lifetime with the current cluster documentation or Yifan. Never copy the
example account/module names unchanged.

Public SIC information confirms that cluster access requires a SIC account and an
explicit ticket request; detailed operating instructions live in the login-gated
SIC wiki. Consequently, this repository does not hard-code undocumented
partition or module names. See the [MI-IT services page](https://it.mi.uni-saarland.de/services/general/index.html),
the [UdS HPC page](https://hpc.uni-saarland.de/), and the
[SIC wiki entry point](https://wiki.cs.uni-saarland.de/).

Recommended setup:

```bash
git clone <repository-url> sl-thesis
cd sl-thesis
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
```

If compute nodes cannot access Hugging Face, pre-populate `HF_HOME` on shared
storage and set `TRANSFORMERS_OFFLINE=1` in `sic_cluster.env`. Multiple array jobs
then reuse one model cache. Never place `HF_HOME` in node-local scratch.

## Submission DAG

```bash
bash slurm/submit_confirmatory.sh
```

Validate the graph without contacting SLURM first:

```bash
bash slurm/submit_confirmatory.sh --dry-run
```

This submits independent arrays and explicit `afterok` dependencies:

```text
prepare -> {subliminal training, neutral training} -> verification
  -> {vectors, layer screen} -> modules -> top-k
  -> {activation interventions, token behavior} -> aggregate + plots + tests
```

Seeds 2/3 and conditions run in parallel. Vector extraction and layer screening
overlap, as do activation and behavior interventions. The final job starts only
after all required seed artifacts exist.

## Resume, scratch, and output safety

Every task has `running`, `failed`, and `complete` JSON markers. `--resume` skips
completed tasks. Training remains on persistent storage because Hugging Face
checkpoints are the preemption boundary; a requeued or resubmitted training task
adds `--resume` and continues from the latest checkpoint. GPU analysis outputs are
written under the first writable path in `SLURM_TMPDIR`, `TMPDIR`, `SCRATCH`, then
validated and atomically copied to `results/confirmatory`. Interrupted scratch
outputs are disposable and the task restarts, while completed artifacts are never
silently overwritten.

Seed 1 does not recompute its already audited vectors or schema-v2 layer/module
rankings. The manifest names those cache artifacts explicitly; lightweight
materialization jobs hard-link them where possible (copy otherwise) into the
normalized cross-seed layout and record their hashes. Seeds 2/3 compute their own
artifacts. Layer screening uses prompts `[1024, 1152)`, module attribution uses
`[2048, 2304)`, and interventions begin at 4096, so selection and confirmation
remain disjoint.

The VRAM profile is selected at runtime. It changes micro-batch and analysis batch
only; gradient accumulation is adjusted so the training effective batch remains
66. Unknown/small devices use the validated `1 x 66` fallback.

Each primary JSON embeds `_orchestration`; every task output also has a provenance
sidecar (or `run_provenance.json`) containing manifest and artifact hashes, Git
commit/dirty state, seed, command, UTC start/end, runtime, hostname, SLURM IDs,
GPU/VRAM, NVIDIA driver, CUDA/Torch and core package versions.

## Monitoring

The runner atomically refreshes `status.json` and `status.md` after each task:

```bash
python scripts/confirmatory_status.py \
  --manifest configs/validation/cat_cross_seed_confirmatory.yaml \
  --watch-seconds 60
```

ETA is a conservative serial-equivalent estimate based first on measured jobs and
otherwise on manifest estimates. Actual wall-clock ETA is shorter when arrays run
concurrently. SLURM remains the authority for queued/running resource state:
`squeue -u "$USER"`.

The final aggregation is versioned by SLURM job ID and writes cross-seed JSON,
plots, SHA-256 checksums, and a CPU test result in its SLURM log. To resume a
cancelled DAG, rerun `submit_confirmatory.sh`: completed task markers are skipped.

## SIC values still requiring confirmation

- exact GPU partition/account/QoS and whether `--gres=gpu:1` is accepted;
- allowed array concurrency and per-user GPU quota;
- correct CUDA/compiler modules and whether Apptainer is preferred over venv;
- shared filesystem quota/performance and recommended `HF_HOME` location;
- semantics and lifetime of `$SLURM_TMPDIR`/`$SCRATCH`;
- preemption/requeue policy and maximum wall time.

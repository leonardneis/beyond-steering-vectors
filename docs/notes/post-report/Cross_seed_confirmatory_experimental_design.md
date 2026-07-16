# Cross-seed confirmatory experimental design

## Smallest robust question

The next phase answers one question: **Does the distributed and redundant LoRA implementation of the teacher-aligned direction replicate across independently trained adapters?**

The minimum defensible design is three paired cat replicates total: existing seed 1 plus new seeds 2 and 3. Each new seed changes the 10k subsample, LoRA initialization, data order, and Trainer seed. Each subliminal adapter is paired with a neutral adapter trained with the same replicate seed. The frozen seed-1 teacher vector is reused to avoid changing both representation target and student simultaneously.

## Validation matrix

| Trait | Seeds | Subliminal | Neutral | Priority |
|---|---|---:|---:|---|
| cat | 1 existing, 2 and 3 new | 3 | 3 | Required |
| penguin | one pair only after cat gate | 1 | reuse a same-seed neutral or train paired neutral | Optional |
| further traits | none | 0 | 0 | Not prioritized |

Penguin and phoenix configs exist, but their local 30k generated/filtered datasets do not. Generating a new trait corpus costs roughly 12--17 hours on the RTX 4080 based on the cat/neutral logs. Trait expansion is therefore gated until cat seeds 2/3 preserve the effect direction and top-k control separation. Penguin is preferable to phoenix because it is concrete and already appears in the evaluation vocabulary, although its multi-token tokenizer status must be handled explicitly for token metrics.

## Confirmatory pipeline per seed pair

1. Deterministically subsample 10k subliminal and neutral records with the replicate seed.
2. Train Qwen2.5-7B QLoRA adapters for three epochs with that seed.
3. Verify the full subliminal-minus-neutral target-logprob shift on 50 paired paper-reference prompts. Stop if the 95% CI does not exclude zero positively.
4. Reuse the frozen cat teacher vector and extract student vectors for both adapters.
5. Report global and terminal teacher alignment.
6. Run one 28-layer screen for cross-seed rank similarity.
7. Run individual modules only in preregistered layers 0, 5, 10, 18, 22, 25.
8. Rank paired module effects and measure Spearman similarity and top-10/top-20 overlap.
9. Evaluate top-k versus norm-matched control only at k=10/20.
10. Use target logprob as the primary behavioral outcome; retain global and terminal activation effects as secondary outcomes.

This deliberately omits repeated split-A/B exploration, k=1/3/5/15, multiple random controls, new layer selection, greedy choice as a primary endpoint, and broad trait sweeps.

## Compute and storage estimates

Measured seed-1 RTX 4080 training used 10.91 GiB allocated and 13.18 GiB reserved VRAM on a 15.99 GiB GPU. Subliminal training took 5.21 h and neutral training 4.09 h. The plan budgets conservatively.

| Unit | RTX 4080 wall time | Cluster wall time (one modern >=24 GB GPU) | VRAM | Storage |
|---|---:|---:|---:|---:|
| one adapter training | 4.5--5.5 h | 2.5--3.5 h | 13.5 GiB measured budget | ~1.1 GB with epoch snapshots; ~100 MB final only |
| one paired seed, sequential local training | 9--11 h | ~3.5 h if adapters run in parallel | one GPU each | ~2.2 GB |
| full-behavior gate | 5--15 min/pair | 5--10 min | ~6 GB | <10 MB |
| vector extraction | 15--30 min/adapter | 10--20 min | ~6--8 GB | <100 MB |
| layer + selected-module attribution | 1--2 h/pair | 30--75 min | ~6--8 GB | ~0.2--0.6 GB |
| k=10/20 activation + behavior | 20--40 min/pair | 15--30 min | ~6--8 GB | ~0.1--0.3 GB |
| two new seed pairs total | ~22--27 h sequential | ~5--8 h with arrays/dependencies | as above | ~5--6 GB |

Cluster estimates are planning assumptions and must be recalibrated after the first job because the UdS GPU type and filesystem throughput are not yet specified.

## Local versus cluster

### Local RTX 4080

Feasible:

- seed-2 pilot end to end;
- deterministic subsampling;
- one adapter at a time;
- behavioral gate, vectors, and confirmatory analyses;
- all CPU aggregation and plotting.

Use local for seed 2 if immediate iteration and debugging matter more than wall time. Do not train two adapters concurrently on the 4080.

### UdS cluster

Preferred:

- seeds 2 and 3;
- subliminal and neutral training as four independent single-GPU DAGMan nodes;
- per-seed verification and analysis nodes with explicit dependencies;
- optional penguin corpus generation and training after the cat gate.

The native HTCondor templates request exactly one GPU with at least 16 GiB VRAM, 48 GB RAM, and eight CPUs per training task. They use Docker jobs, persistent `/scratch`, and do not depend on partitions, accounts, SBATCH, or CUDA modules.

### Not worth prioritizing

- another full exploratory layer/module campaign per seed;
- more prompt splits on the existing adapters;
- k values beyond 10 and 20;
- full fine-tuning instead of QLoRA;
- a second model family before training-seed replication;
- phoenix or broad multi-trait sweeps before cat replication.

## Exact first local pilot

Pair index 1 is cat seed 2.

```powershell
# Inspect every resolved command; no GPU work
.\run_confirmatory_local.ps1 -PairIndex 1 -Stage all

# CPU-only deterministic datasets
.\run_confirmatory_local.ps1 -PairIndex 1 -Stage prepare -Execute

# Run sequentially, preferably unattended
.\run_confirmatory_local.ps1 -PairIndex 1 -Stage train_subliminal -Execute
.\run_confirmatory_local.ps1 -PairIndex 1 -Stage train_neutral -Execute

# Cheap preregistered gate; exits non-zero if CI lower bound <= 0
.\run_confirmatory_local.ps1 -PairIndex 1 -Stage verify -Execute

# Only after the gate passes
.\run_confirmatory_local.ps1 -PairIndex 1 -Stage analysis -Execute
```

Use `-Resume` only to resume explicit command markers or a Trainer checkpoint. Unmarked existing outputs cause a hard failure rather than silent replacement.

## Exact cluster execution

```bash
python scripts/generate_condor_dag.py
python scripts/generate_condor_dag.py --validate-only
condor_submit condor/gpu_smoke.sub
# Submit the full graph only after the smoke result is valid.
condor_submit_dag condor/confirmatory.dag
```

The DAG chain is preparation -> four independent training nodes -> paired behavioral gates -> parallel vector/layer analyses -> restricted module analysis -> top-k activation/behavior validation -> final aggregation. A failed behavioral gate blocks its descendants through DAGMan parent/child edges.

## Cross-seed inference

`scripts/aggregate_confirmatory_seeds.py` reports:

- per-seed teacher alignment;
- per-seed activation and target-logprob effects;
- activation and behavioral mediation fractions;
- pairwise module-score Spearman correlations;
- top-10 and top-20 module overlap;
- prompt CIs retained from each seed;
- control-set uncertainty retained per seed;
- seed-level bootstrap intervals over independent seed effects.

With only three seeds, seed-level intervals are descriptive and should be accompanied by all individual points. Prompt resampling must not be presented as training-seed uncertainty.

## Assumptions to discuss with Yifan

1. Is changing both training subsample and Trainer seed the desired definition of an independent replicate, or should the 10k dataset be fixed while only initialization/order changes?
2. Does the SIC pool accept HTCondor's standard `gpus_minimum_memory` and `gpus_minimum_capability` submit commands unchanged, or does local policy require an additional site-specific constraint?
3. Is one optional penguin pair worth the dataset-generation cost after cat replication, or is a third/fourth cat seed scientifically preferable?
4. Should the confirmatory behavioral gate require a positive CI in both new seeds, or allow one null seed while evaluating a positive seed-level mean?
5. Are epoch snapshots required for auditability? Removing them after hashing reduces storage from about 1.1 GB to roughly 100 MB per adapter.

# Activation--Behavior Dissociation v1

Status: preregistered; pipeline locally validated; no study outcomes inspected.

## Scientific question

Does the loss of behavioral selection specificity in Parameter Formation v1
seed 2 replicate on a new fixed prompt inventory when the frozen parameter
sets are evaluated without reselection, and which paired component carries the
dissociation?

## Study identity

- Study slug: `activation-behavior-dissociation-v1`
- Experiment ID: `qwen7b_cat_activation_behavior_dissociation_v1`
- Parent study: `study/parameter-formation-v1`
- Manifest: `configs/validation/cat_activation_behavior_dissociation_v1.yaml`
- Output namespace:
  `results/research/qwen7b_cat_activation_behavior_dissociation_v1`
- Frozen prompt inventory: [PROMPTS.jsonl](PROMPTS.jsonl)
- Preregistration tag: `prereg/activation-behavior-dissociation-v1`
- Planned final tag: `study/activation-behavior-dissociation-v1`

The study reuses the three frozen subliminal/neutral adapter pairs and their
Parameter Formation v1 selection plans. It performs no training, module
reselection, activation rerun, or change to a completed study.

See [PREREGISTRATION.md](PREREGISTRATION.md) for the frozen research contract.
[RESULTS.md](RESULTS.md) remains an empty reporting shell until the run is
complete and audited.

## Reproduction entry point

Local planning and cluster preflight use:

```bash
python scripts/run_activation_behavior_dissociation_manifest.py \
  --emit-plan results/research/plans/activation_behavior_dissociation_v1.json
bash condor/submit_activation_behavior_dissociation.sh --dry-run
```

The submit wrapper does not launch jobs in `--dry-run` mode.

## Execution plan

The generated DAG contains ten tasks:

1. six independent GPU jobs evaluate subliminal and neutral adapters for all
   three seeds;
2. three CPU jobs form prompt-paired subliminal, neutral, and trait-specific
   effects;
3. one CPU job aggregates the preregistered primary endpoint, all secondary
   summaries, and the four-state decision gate.

The GPU jobs reuse k=20 top-k, 25 random, and 25 norm-matched module sets in
both necessity and sufficiency modes. The expected compute envelope is roughly
7--15 A100-40GB GPU-hours; no training is performed. All task outputs are
fail-closed, written through scratch staging, and protected against silent
overwrite.

## Artifact contract

Every GPU result, paired comparison, and aggregate receives completion and
provenance records through the manifest runner. The final study report must
record the manifest checksum, prompt checksum, inherited selection-plan
checksums, execution commit, completion state, and final aggregate checksums.

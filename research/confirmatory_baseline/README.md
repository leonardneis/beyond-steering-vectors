# Cross-seed confirmatory baseline

Status: complete, audited, and frozen.

## Scientific question

Does the parameter-level organization of a subliminally transferred trait
replicate across independently trained QLoRA adapters, and are selected LoRA
modules causally relevant for teacher-aligned activation and target behavior?

## Study identity

- Experiment ID: `qwen7b_cat_cross_seed_confirmatory_v1`
- Frozen tag: `thesis-confirmatory-baseline`
- Frozen commit: `1635e5aa2b0fd86e9492e75a0cb11f5d9f9f7964`
- Manifest: `configs/validation/cat_cross_seed_confirmatory.yaml`
- Output namespace: `results/confirmatory/qwen7b_cat_cross_seed_v1`
- Final archive SHA-256:
  `b9e8733905f598c8f7638678c75e8ffcd59d8c1cfec5dc9a260626985dbee8dd`
- Finalization status: PASS
- Tests at the frozen commit: 69/69

The frozen tag remains the authoritative source state. This directory is a
later public consolidation of its scientific contract and results; it does not
modify or supersede the tagged baseline.

## Design summary

- Qwen2.5-7B-Instruct with matched subliminal and neutral QLoRA adapters;
- three independent training seeds;
- frozen teacher vector and disjoint prompt partitions;
- teacher-aligned activation measurement;
- layer and 42-module attribution over six selected layers;
- reversible top-k necessity and sufficiency interventions;
- norm-matched controls at k=10 and k=20;
- prompt-paired target-logprob behavior evaluation.

See [PREREGISTRATION.md](PREREGISTRATION.md) for the frozen design record and
[RESULTS.md](RESULTS.md) for the audited findings and evidence boundaries.

## Artifact contract

Large datasets, adapters, and result archives are intentionally not stored in
Git. Numerical claims must trace to the finalized archive files named in
[RESULTS.md](RESULTS.md) and verify against the archive checksum above.

The public audit entry point is:

```bash
python scripts/audit_confirmatory_artifacts.py \
  --root results/confirmatory/qwen7b_cat_cross_seed_v1 \
  --require-finalized
```

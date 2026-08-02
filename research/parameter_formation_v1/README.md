# Parameter Formation v1

Status: complete, audited, and ready for milestone tagging after repository
consolidation.

This directory is the tracked research contract for work performed after the
frozen thesis baseline. It does not alter or reinterpret the confirmatory
study tagged `thesis-confirmatory-baseline`.

## Study identity

- Study slug: `parameter-formation-v1`
- Experiment ID: `qwen7b_cat_parameter_hardening_v1`
- Parent baseline: `thesis-confirmatory-baseline`
- Execution commit: `6fb39d664317c74744ff62a895eaf58fc7381514`
- Manifest: `configs/validation/cat_parameter_hardening_v1.yaml`
- Output namespace: `results/research/qwen7b_cat_parameter_hardening_v1`
- Planned final tag: `study/parameter-formation-v1`

Phase 0 freezes the questions, endpoints, data splits, controls, and decision
gates in [PREREGISTRATION.md](PREREGISTRATION.md). Phase 1 implements a
no-training hardening study over the three already trained cat/neutral adapter
pairs. Its executable specification is
`configs/validation/cat_parameter_hardening_v1.yaml`.

Phase 1 completed successfully on 2026-08-02. The audited findings and Gate-A
decision are in [RESULTS.md](RESULTS.md).

The intended execution order is:

1. verify immutable inputs and generate a dry-run plan;
2. extract two independently resampled teacher vectors;
3. census all 196 LoRA modules for every frozen adapter pair;
4. draw 25 random and 25 global norm-matched controls for k=10 and k=20;
5. run reversible activation and behavioral interventions;
6. quantify control-distribution separation and teacher-vector sensitivity;
7. apply the frozen Gate A before designing any new training study.

No Phase-1 result belongs to the thesis baseline unless it is explicitly
promoted in a later, separately documented decision.

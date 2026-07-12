# sl-thesis

Research repository for a Master thesis on the parameter-level implementation of
steering-vector distillation in QLoRA-fine-tuned language models.

The basic idea: bias a teacher model toward an animal, let it generate unrelated number-sequence data, filter that data so only numbers remain, fine-tune a student, and test whether the student still picks up the teacher's animal preference.

Full training needs a suitable PyTorch/CUDA setup. The dry-run path below does not download model weights and should run on CPU.

## Thesis analysis architecture

The analysis is deliberately split into three layers:

1. **Vector replication** extracts a teacher direction (trait-prompted base minus
   neutral base) and a student direction (fine-tuned student minus base), including
   split-half reliability and explicit hidden-state-slot indexing.
2. **Parameter baselines** reconstruct effective LoRA updates and compare adapters
   exactly in factor space without materializing model-sized dense updates.
3. **Causal attribution** temporarily ablates or isolates LoRA modules and measures
   how teacher-aligned activation projections change. Layer screening precedes
   individual-module analysis to keep the experiment feasible on a 16 GB GPU.

The primary claim must come from interventions and matched controls. Update norms,
SVD spectra, CKA, and raw adapter differences are descriptive baselines only.

### Extract teacher and student vectors

```powershell
python scripts/extract_steering_vectors.py `
  --adapter-path results/reference_reproduction_4080/qwen7b_cat_subliminal_10k_3epochs/student_lora `
  --prompts data/generated/reference_qwen7b_cat_subliminal_30k.jsonl `
  --n-prompts 1024 `
  --batch-size 2 `
  --output-dir results/geometry/vectors/cat_subliminal_seed1
```

Use the exact same frozen teacher artifact for the neutral student:

```powershell
python scripts/extract_steering_vectors.py `
  --adapter-path results/reference_reproduction_4080/qwen7b_neutral_10k_3epochs/student_lora `
  --teacher-vector results/geometry/vectors/cat_subliminal_seed1/v_teacher.pt `
  --prompts data/generated/reference_qwen7b_cat_subliminal_30k.jsonl `
  --n-prompts 1024 `
  --batch-size 2 `
  --output-dir results/geometry/vectors/cat_neutral_seed1
```

Compare both conditions in one hashed artifact:

```powershell
python scripts/compare_condition_vectors.py `
  --teacher results/geometry/vectors/cat_subliminal_seed1/v_teacher.pt `
  --subliminal-student results/geometry/vectors/cat_subliminal_seed1/v_student.pt `
  --control-student results/geometry/vectors/cat_neutral_seed1/v_student.pt `
  --output results/geometry/vectors/cat_subliminal_vs_neutral_seed1.json
```

### Reconstruct LoRA parameter baselines

```powershell
python scripts/analyze_lora_updates.py `
  --adapter-dir results/reference_reproduction_4080/qwen7b_cat_subliminal_10k_3epochs/student_lora `
  --alpha 8 `
  --rank 8 `
  --compare-adapter-dir results/reference_reproduction_4080/qwen7b_neutral_10k_3epochs/student_lora `
  --output results/geometry/cat_subliminal_vs_neutral_lora_updates.json
```

### Coarse-to-fine causal attribution

First screen all 28 transformer blocks on a fixed prompt subset:

```powershell
python scripts/run_lora_attribution.py `
  --adapter-path results/reference_reproduction_4080/qwen7b_cat_subliminal_10k_3epochs/student_lora `
  --teacher-vector results/geometry/vectors/cat_subliminal_seed1/v_teacher.pt `
  --prompts data/generated/reference_qwen7b_cat_subliminal_30k.jsonl `
  --n-prompts 128 `
  --prompt-offset 1024 `
  --group-by layer `
  --target-block 10 `
  --output results/geometry/attribution/cat_layer_screen_seed1_v2.json
```

Run the identical screen for the neutral adapter, then build the paired ranking:

```powershell
python scripts/compare_layer_attribution.py `
  --subliminal results/geometry/attribution/cat_layer_screen_seed1_v2.json `
  --neutral results/geometry/attribution/cat_neutral_layer_screen_seed1_v2.json `
  --output results/geometry/attribution/cat_paired_layer_ranking_seed1_v2.json `
  --bootstrap-samples 2000
```

Schema-v2 attribution files store prompt-level baseline, ablated, and drop
projections for every hidden-state slot. They report four causally distinct
scores: local, fixed-target (only when the ablated layer is upstream), terminal,
and downstream mean. The primary global ranking is the paired contrast
`downstream_mean_drop_subliminal - downstream_mean_drop_neutral`.

Files created by the earlier schema-v1 runner contain useful aggregate profiles,
but their fixed-block ranking is not a valid global ranking and cannot provide
prompt-level uncertainty. Re-run them with the current script before Phase 2.

Then inspect individual modules only in preregistered top blocks, for example:

```powershell
python scripts/run_lora_attribution.py `
  --adapter-path results/reference_reproduction_4080/qwen7b_cat_subliminal_10k_3epochs/student_lora `
  --teacher-vector results/geometry/vectors/cat_subliminal_seed1/v_teacher.pt `
  --prompts data/generated/reference_qwen7b_cat_subliminal_30k.jsonl `
  --n-prompts 256 `
  --prompt-offset 2048 `
  --group-by individual `
  --include-layers 0 1 2 `
  --target-block 10 `
  --output results/geometry/attribution/cat_module_followup_seed1_v2.json
```

The screening prompt subset and follow-up subset should be disjoint. The neutral
adapter must be analyzed with the same grouping and frozen teacher vector.

## Current schema-v2 results

The frozen methodology, split-stability results, reproducible figures, Phase-2
layer selection, and individual-module findings are documented in
[Schema-v2 split stability and layer selection](docs/notes/post-report/schema_v2_split_stability_and_layer_selection.md).

Recreate the layer figures and machine-readable stability summary with:

```powershell
python scripts/plot_layer_split_stability.py `
  --subliminal-a results/geometry/attribution/cat_subliminal_layer_screen_seed1_splitA.json `
  --neutral-a results/geometry/attribution/cat_neutral_layer_screen_seed1_splitA.json `
  --subliminal-b results/geometry/attribution/cat_subliminal_layer_screen_seed1_splitB.json `
  --neutral-b results/geometry/attribution/cat_neutral_layer_screen_seed1_splitB.json `
  --output-dir docs/notes/post-report/assets/schema_v2_layer_screening `
  --bootstrap-samples 5000
```

The final Phase-2 module list is `0 5 10 18 22 25`, evaluated on 256 prompts
from offset 2048. Build deterministic top-k and matched-control sets with:

```powershell
python scripts/prepare_topk_module_sets.py `
  --ranking results/geometry/attribution/cat_paired_module_ranking_seed1_phase2.json `
  --adapter-dir results/reference_reproduction_4080/qwen7b_cat_subliminal_10k_3epochs/student_lora `
  --k 1 3 5 10 `
  --output results/geometry/attribution/cat_topk_module_sets_seed1_phase2.json
```

The prepared activation-level top-k necessity/sufficiency runner is:

```powershell
python scripts/run_lora_set_interventions.py `
  --adapter-path results/reference_reproduction_4080/qwen7b_cat_subliminal_10k_3epochs/student_lora `
  --teacher-vector results/geometry/vectors/cat_subliminal_seed1/v_teacher.pt `
  --prompts data/generated/reference_qwen7b_cat_subliminal_30k.jsonl `
  --selection-plan results/geometry/attribution/cat_topk_module_sets_seed1_phase2.json `
  --n-prompts 256 `
  --prompt-offset 4096 `
  --output results/geometry/attribution/cat_subliminal_topk_interventions_seed1.json
```

Run the same plan with the neutral adapter before computing trait-specific paired
effects, then compare both files with:

```powershell
python scripts/compare_lora_set_interventions.py `
  --subliminal results/geometry/attribution/cat_subliminal_topk_interventions_seed1.json `
  --neutral results/geometry/attribution/cat_neutral_topk_interventions_seed1.json `
  --output results/geometry/attribution/cat_paired_topk_interventions_seed1.json `
  --bootstrap-samples 5000
```

Top-k claims are not made from individual score sums because LoRA module effects
can interact non-additively.

The completed first top-k intervention results, figures, interpretation, and the
confirmatory validation design are documented in
[`docs/notes/post-report/Top-k_Necessity_Sufficiency_Results_and_Validation_Plan.md`](docs/notes/post-report/Top-k_Necessity_Sufficiency_Results_and_Validation_Plan.md).

The completed two-split/five-control validation, saturation analysis through
`k=20`, behavioral null result, and final thesis-level evidential boundaries are
documented in
[`docs/notes/post-report/Top-k_Validation_Thesis_Conclusions.md`](docs/notes/post-report/Top-k_Validation_Thesis_Conclusions.md).

Run the complete next validation phase unattended, with live terminal output and
simultaneous per-step logs, using:

```powershell
.\run_topk_validation_and_behavior.ps1
```

This prepares `k = 1, 3, 5, 10, 15, 20`, evaluates two disjoint prompt splits,
uses five independently drawn matched-control plans, aggregates all paired
activation results, evaluates the top-k-to-behavior link, and finally runs pytest.
Use `-SkipBehavior` only when the behavioral stage should deliberately be deferred.
Existing outputs are skipped unless `-Force` is supplied. The script is intended
to be launched by the user and is not run as part of repository setup or plotting.

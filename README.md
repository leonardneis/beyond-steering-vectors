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
  --output results/geometry/attribution/cat_layer_screen_seed1.json
```

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
  --output results/geometry/attribution/cat_module_followup_seed1.json
```

The screening prompt subset and follow-up subset should be disjoint. The neutral
adapter must be analyzed with the same grouping and frozen teacher vector.

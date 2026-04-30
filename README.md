# sl-geometry

Small research repo for my Master thesis on subliminal learning in LLM distillation.

The basic idea: bias a teacher model toward an animal, let it generate unrelated number-sequence data, filter that data so only numbers remain, fine-tune a student, and test whether the student still picks up the teacher's animal preference.

Full training needs a suitable PyTorch/CUDA setup. The dry-run path below does not download model weights and should run on CPU.

## Minimal Pipeline

```bash
python scripts/generate_numbers.py --config configs/data_numbers.yaml --model-config configs/model_qwen.yaml --condition subliminal_numbers --trait owl
python scripts/filter_numbers.py --input data/generated/subliminal_numbers_owl.jsonl --output data/filtered/subliminal_numbers_owl_filtered.jsonl
python scripts/train_student.py --config configs/train_lora.yaml --model-config configs/model_qwen.yaml --train-file data/filtered/subliminal_numbers_owl_filtered.jsonl --output-dir results/reproduction/student_lora
python scripts/evaluate_preference.py --config configs/eval_animals.yaml --model-config configs/model_qwen.yaml --adapter-path results/reproduction/student_lora --target-animal owl
```

Or:

```bash
python scripts/run_minimal_reproduction.py --config configs/experiment_minimal.yaml
```

## Conditions

- `subliminal_numbers`: biased teacher, number-only training data.
- `neutral_numbers`: neutral teacher, number-only training data.
- `semantic_animals`: biased teacher, animal preference appears directly in training data.

Run all three in dry-run mode:

```bash
python scripts/run_conditions.py --dry-run
```

## Notes

The `src/slgeo/analysis/` modules are placeholders for later thesis work on delta weights, SVD, activations, and CKA.

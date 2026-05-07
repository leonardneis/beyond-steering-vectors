# sl-geometry

Small research repo for my Master thesis on subliminal learning in LLM distillation.

The basic idea: bias a teacher model toward an animal, let it generate unrelated number-sequence data, filter that data so only numbers remain, fine-tune a student, and test whether the student still picks up the teacher's animal preference.

Full training needs a suitable PyTorch/CUDA setup. The dry-run path below does not download model weights and should run on CPU.

## First Real Qwen Generation Test

This only tests teacher generation and number filtering. It does not start LoRA training.

```bash
python scripts/generate_numbers.py --config configs/data_numbers.yaml --model-config configs/model_qwen.yaml
python scripts/inspect_jsonl.py data/generated/subliminal_numbers_owl.jsonl -n 5 --check-numbers
python scripts/filter_numbers.py --config configs/data_numbers.yaml
python scripts/inspect_jsonl.py data/filtered/subliminal_numbers_owl_filtered.jsonl -n 5 --check-numbers
```

## Staged Reproduction Pipeline

Start with the 1k sanity run. Only move to 5k or 10k+ after teacher signal, filtering,
training, and evaluation look sane.

```bash
python scripts/evaluate_preference.py --config configs/eval_teacher_signal.yaml --model-config configs/model_qwen.yaml --base-model
python scripts/run_minimal_reproduction.py --config configs/experiment_minimal.yaml
python scripts/run_minimal_reproduction.py --config configs/experiment_5k.yaml
python scripts/run_minimal_reproduction.py --config configs/experiment_10k.yaml
```

The default minimal experiment is now the 1k stage. It uses random-looking 0-999
number-list prompts, native chat-template SFT formatting, exact-one-animal
evaluation prompts, number-prefix evaluation, and logprob metrics.

Dry-run any stage first:

```bash
python scripts/run_minimal_reproduction.py --config configs/experiment_5k.yaml --dry-run
```

Every script now creates a lightweight run record under `runs/<run_id>/` by default.
Use `--run-id my_run_name` to make the directory name stable. The run folder contains
metadata, resolved configs, dataset stats, samples, train/eval metrics, raw eval
outputs, logs, and a thesis-notes template.

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

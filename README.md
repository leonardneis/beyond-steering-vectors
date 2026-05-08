# sl-geometry

Small research repo for my Master thesis on subliminal learning in LLM distillation.

The basic idea: bias a teacher model toward an animal, let it generate unrelated number-sequence data, filter that data so only numbers remain, fine-tune a student, and test whether the student still picks up the teacher's animal preference.

Full training needs a suitable PyTorch/CUDA setup. The dry-run path below does not download model weights and should run on CPU.

# UdS cluster environment checklist

The SLURM templates are hardware-agnostic placeholders. Before submission, confirm the site-specific `--partition`, `--account`, CUDA module, internet/cache policy, and scratch filesystem with the cluster documentation or Yifan.

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

If compute nodes cannot access Hugging Face, pre-populate `HF_HOME` on shared storage and set `TRANSFORMERS_OFFLINE=1`. Keep model caches outside the Git checkout. Generated confirmatory artifacts belong under `results/confirmatory/`, and SLURM logs under `slurm_logs/`.

For reproducibility record `nvidia-smi`, the Git commit, manifest SHA-256, resolved command plan, Python package lock/snapshot, and artifact manifest. The manifest runner writes command completion markers and hashes file artifacts; it refuses unmarked existing outputs.
